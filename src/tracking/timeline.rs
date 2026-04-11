//! Timeline semaphore based GPU completion tracking.
//!
//! Replaces per-fence polling with a single monotonic counter per queue.
//! Uses `VK_KHR_timeline_semaphore` (core in Vulkan 1.2) for O(1)
//! kernel-side blocking instead of O(N) fence polling.
//!
//! # How It Works
//!
//! Each queue has a timeline semaphore with a monotonically increasing
//! value. Every submission signals the next value. To check completion,
//! read the current counter (`vkGetSemaphoreCounterValue`). To wait,
//! use `vkWaitSemaphores` which blocks in the kernel without polling.
//!
//! The background [`TimelineWatcher`] thread calls `vkWaitSemaphores`
//! with the earliest pending value across all queues. When any queue
//! progresses, it wakes all futures whose target values have been reached.
//! This is O(queues + completed) per wake instead of O(total_pending).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::Waker;
use std::time::Duration;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// A timeline semaphore owned by a single queue.
///
/// Thread-safe: the value counter is atomic, the semaphore handle
/// is immutable after creation.
pub struct QueueTimeline {
    pub(crate) shared: Arc<SharedState>,
    pub(crate) semaphore: vk::Semaphore,
    /// The next value to signal. Incremented atomically before each submit.
    next_value: AtomicU64,
}

impl QueueTimeline {
    /// Create a timeline semaphore starting at value 0.
    pub fn new(shared: Arc<SharedState>) -> Result<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        let semaphore = unsafe { shared.device.create_semaphore(&ci, None)? };

        Ok(Self {
            shared,
            semaphore,
            next_value: AtomicU64::new(1),
        })
    }

    /// Claim the next timeline value for a submission.
    /// The caller must signal this value in their `vkQueueSubmit`.
    pub fn claim_next_value(&self) -> u64 {
        self.next_value.fetch_add(1, Ordering::Relaxed)
    }

    /// Read the current semaphore counter (non-blocking).
    pub fn current_value(&self) -> Result<u64> {
        let val = unsafe {
            self.shared
                .device
                .get_semaphore_counter_value(self.semaphore)?
        };
        Ok(val)
    }

    /// Block until the timeline reaches at least `value`.
    pub fn wait_for_value(&self, value: u64, timeout_ns: u64) -> Result<bool> {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.semaphore))
            .values(std::slice::from_ref(&value));

        match unsafe { self.shared.device.wait_semaphores(&wait_info, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(Error::Vulkan(e)),
        }
    }

    /// The raw semaphore handle.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for QueueTimeline {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_semaphore(self.semaphore, None);
        }
    }
}

/// A pending waker waiting for a specific timeline value.
struct PendingWaker {
    waker: Waker,
}

/// Internal state shared between `TimelineWatcher` thread and user threads.
struct WatcherState {
    /// Per-queue pending wakers: `timeline_semaphore_raw` -> (value -> wakers)
    queues: std::collections::HashMap<u64, BTreeMap<u64, Vec<PendingWaker>>>,
}

/// Efficient background thread for waking async tasks on GPU completion.
///
/// Uses `vkWaitSemaphores` with `ANY` flag to block until any tracked
/// queue makes progress. Then processes all completed futures in batch.
///
/// # Complexity
///
/// - Blocking: O(1) kernel-side (handled by the GPU driver)
/// - Wake processing: O(queues + `completed_futures`) per wake-up
/// - Registration: O(log N) per future (`BTreeMap` insert)
pub struct TimelineWatcher {
    state: Arc<Mutex<WatcherState>>,
    notify: Arc<WatcherNotify>,
    shutdown: Arc<AtomicBool>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

struct WatcherNotify {
    flag: Mutex<bool>,
    cvar: Condvar,
}

impl WatcherNotify {
    fn new() -> Self {
        Self {
            flag: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    fn signal(&self) {
        let mut f = self.flag.lock().unwrap();
        *f = true;
        self.cvar.notify_one();
    }

    fn wait(&self, timeout: Duration) {
        let f = self.flag.lock().unwrap();
        let (mut f, _) = self.cvar.wait_timeout(f, timeout).unwrap();
        *f = false;
    }
}

impl TimelineWatcher {
    /// Create and start a timeline watcher.
    pub fn new(shared: Arc<SharedState>) -> Self {
        let state = Arc::new(Mutex::new(WatcherState {
            queues: Default::default(),
        }));
        let notify = Arc::new(WatcherNotify::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let t_shared = Arc::clone(&shared);
        let t_state = Arc::clone(&state);
        let t_notify = Arc::clone(&notify);
        let t_shutdown = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("ignis-timeline-watcher".into())
            .spawn(move || {
                Self::watcher_loop(&t_shared, &t_state, &t_notify, &t_shutdown);
            })
            .expect("failed to spawn timeline watcher thread");

        Self {
            state,
            notify,
            shutdown,
            handle: Mutex::new(Some(handle)),
        }
    }

    /// Register a waker for a specific timeline value.
    pub(crate) fn register(
        &self,
        semaphore: vk::Semaphore,
        target_value: u64,
        waker: Waker,
    ) {
        use ash::vk::Handle;
        let mut state = self.state.lock().unwrap();
        let queue_map = state
            .queues
            .entry(semaphore.as_raw())
            .or_default();
        queue_map
            .entry(target_value)
            .or_default()
            .push(PendingWaker { waker });
        drop(state);
        self.notify.signal();
    }

    fn watcher_loop(
        shared: &SharedState,
        state: &Mutex<WatcherState>,
        notify: &WatcherNotify,
        shutdown: &AtomicBool,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Collect minimum pending value per queue.
            let wait_targets: Vec<(vk::Semaphore, u64)> = {
                let s = state.lock().unwrap();
                s.queues
                    .iter()
                    .filter_map(|(&sem_raw, map)| {
                        map.keys().next().map(|&min_val| {
                            (vk::Semaphore::from_raw(sem_raw), min_val)
                        })
                    })
                    .collect()
            };

            if wait_targets.is_empty() {
                // Nothing to watch. Sleep until new registration.
                if !shutdown.load(Ordering::Relaxed) {
                    notify.wait(Duration::from_millis(100));
                }
                continue;
            }

            let semaphores: Vec<vk::Semaphore> = wait_targets.iter().map(|(s, _)| *s).collect();
            let values: Vec<u64> = wait_targets.iter().map(|(_, v)| *v).collect();

            let wait_info = vk::SemaphoreWaitInfo::default()
                .flags(vk::SemaphoreWaitFlags::ANY)
                .semaphores(&semaphores)
                .values(&values);

            // Block in kernel until ANY semaphore reaches its target.
            // Use 50ms timeout so we can check shutdown periodically.
            let _result = unsafe {
                shared
                    .device
                    .wait_semaphores(&wait_info, 50_000_000)
            };

            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Read current values and wake completed futures.
            let mut s = state.lock().unwrap();
            for (&sem_raw, map) in &mut s.queues {
                let sem = vk::Semaphore::from_raw(sem_raw);
                let current = match unsafe {
                    shared.device.get_semaphore_counter_value(sem)
                } {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Drain all entries where target <= current.
                let split_point = current + 1;
                let completed = map.split_off(&split_point);
                let to_wake = std::mem::replace(map, completed);

                for (_, wakers) in to_wake {
                    for pw in wakers {
                        pw.waker.wake();
                    }
                }
            }

            // Prune empty queues.
            s.queues.retain(|_, map| !map.is_empty());
        }
    }

    /// Number of pending wakers across all queues.
    pub fn pending_count(&self) -> usize {
        self.state
            .lock()
            .unwrap()
            .queues
            .values()
            .map(|m| m.values().map(std::vec::Vec::len).sum::<usize>())
            .sum()
    }
}

impl Drop for TimelineWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.notify.signal();
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

// Need Handle trait for from_raw.
use ash::vk::Handle;