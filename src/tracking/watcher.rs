//! Background fence monitoring for efficient async GPU completion.
//!
//! [`FenceWatcher`] runs a dedicated thread that periodically checks
//! Vulkan fence status and wakes the corresponding [`std::task::Waker`]
//! when a fence becomes signaled. This eliminates the busy-wait loop
//! that a naive `Future::poll` implementation would require.
//!
//! # Architecture
//!
//! The watcher thread maintains a list of `(fence, waker)` entries.
//! Each iteration:
//!
//! 1. Snapshots the current entry list
//! 2. Checks each fence via `vkGetFenceStatus` (non-blocking)
//! 3. Wakes and removes signaled entries
//! 4. Sleeps via condvar with a configurable poll interval
//!
//! The condvar is notified when new fences are registered, allowing
//! immediate checking without waiting for the next interval.
//!
//! # Safety
//!
//! Each entry's fence access is guarded by a per-entry mutex. When a
//! [`GpuFuture`](crate::GpuFuture) is dropped, it sets the `dropped`
//! flag under the lock, preventing the watcher from accessing a fence
//! that is about to be destroyed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::Waker;
use std::time::Duration;

use ash::vk;

use crate::device::SharedState;

/// State shared between a single [`GpuFuture`](crate::GpuFuture) and
/// the [`FenceWatcher`] thread.
///
/// The per-entry mutex guarantees that the watcher never accesses a fence
/// after the owning future has set `dropped = true`.
pub(crate) struct WatchedFenceState {
    /// The Vulkan fence handle being monitored.
    pub fence: vk::Fence,
    /// Mutable state protected by a mutex.
    pub inner: Mutex<WatchedFenceInner>,
}

/// Mutable interior of a watched fence entry.
pub(crate) struct WatchedFenceInner {
    /// Set to `true` when the fence has been signaled.
    pub completed: bool,
    /// Set to `true` by the `GpuFuture`'s drop impl to prevent further
    /// fence access by the watcher.
    pub dropped: bool,
    /// The most recent waker from the async executor.
    pub waker: Option<Waker>,
    /// If the fence check returned a Vulkan error, it is stored here.
    pub error: Option<vk::Result>,
}

/// Background thread for monitoring Vulkan fences and waking async tasks.
///
/// Created via [`Ignis::create_fence_watcher`](crate::Ignis::create_fence_watcher).
///
/// Attach a watcher to submissions via
/// [`SubmitBuilder::with_watcher`](crate::SubmitBuilder::with_watcher).
/// Futures created with a watcher use efficient sleep-based polling
/// instead of busy-waiting.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::*; use ash::vk; use std::time::Duration;
/// # fn example(ignis: &Ignis, queue: &AsyncQueue,
/// #            cmd: vk::CommandBuffer) -> Result<()> {
/// let watcher = ignis.create_fence_watcher(Duration::from_micros(200));
///
/// let future = queue.submit()
///     .command_buffer(cmd)
///     .with_watcher(&watcher)
///     .build()?;
///
/// // The future will be woken efficiently by the watcher thread.
/// // future.await?;  // works with any async executor
///
/// // Or block:
/// future.wait()?;
/// # Ok(())
/// # }
/// ```
///
/// # Poll Interval Tuning
///
/// The `poll_interval` parameter controls how often the watcher thread
/// checks fence status. A shorter interval reduces latency but increases
/// CPU usage. For frames at 60 Hz (~16ms), 100-500 microseconds is a
/// good default.
pub struct FenceWatcher {
    _shared: Arc<SharedState>,
    entries: Arc<Mutex<Vec<Arc<WatchedFenceState>>>>,
    notify: Arc<NotifyPair>,
    shutdown: Arc<AtomicBool>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// Condvar + Mutex pair for waking the monitor thread.
struct NotifyPair {
    flag: Mutex<bool>,
    cvar: Condvar,
}

impl NotifyPair {
    fn new() -> Self {
        Self {
            flag: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    /// Signal the watcher thread to wake up.
    fn signal(&self) {
        let mut flag = self.flag.lock().unwrap();
        *flag = true;
        self.cvar.notify_one();
    }

    /// Wait for a signal or timeout.
    fn wait(&self, timeout: Duration) {
        let flag = self.flag.lock().unwrap();
        let (mut flag, _) = self.cvar.wait_timeout(flag, timeout).unwrap();
        *flag = false;
    }
}

impl FenceWatcher {
    /// Create and start a new fence watcher.
    ///
    /// # Arguments
    ///
    /// * `shared` - Device state for Vulkan API calls
    /// * `poll_interval` - How often to check fence status when idle
    pub fn new(shared: Arc<SharedState>, poll_interval: Duration) -> Self {
        let entries: Arc<Mutex<Vec<Arc<WatchedFenceState>>>> = Arc::new(Mutex::new(Vec::new()));
        let notify = Arc::new(NotifyPair::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let thread_shared = Arc::clone(&shared);
        let thread_entries = Arc::clone(&entries);
        let thread_notify = Arc::clone(&notify);
        let thread_shutdown = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("ignis-fence-watcher".into())
            .spawn(move || {
                Self::monitor_loop(
                    &thread_shared,
                    &thread_entries,
                    &thread_notify,
                    &thread_shutdown,
                    poll_interval,
                );
            })
            .expect("failed to spawn ignis fence watcher thread");

        Self {
            _shared: shared,
            entries,
            notify,
            shutdown,
            handle: Mutex::new(Some(handle)),
        }
    }

    /// The main loop running on the watcher thread.
    fn monitor_loop(
        shared: &SharedState,
        entries: &Mutex<Vec<Arc<WatchedFenceState>>>,
        notify: &NotifyPair,
        shutdown: &AtomicBool,
        interval: Duration,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Take a snapshot so we don't hold the list lock during Vulkan calls.
            let snapshot: Vec<Arc<WatchedFenceState>> = {
                let guard = entries.lock().unwrap();
                guard.clone()
            };

            let mut any_completed = false;

            for entry in &snapshot {
                // Lock the per-entry mutex to safely access the fence.
                let mut inner = entry.inner.lock().unwrap();

                if inner.dropped || inner.completed {
                    any_completed = true;
                    continue;
                }

                // SAFETY: the fence is valid because `dropped` is false and
                // we hold the per-entry lock, preventing the GpuFuture from
                // destroying the fence concurrently.
                match unsafe { shared.device.get_fence_status(entry.fence) } {
                    Ok(true) => {
                        inner.completed = true;
                        any_completed = true;
                        if let Some(waker) = inner.waker.take() {
                            waker.wake();
                        }
                    }
                    Ok(false) => {
                        // Still pending, nothing to do.
                    }
                    Err(e) => {
                        inner.completed = true;
                        inner.error = Some(e);
                        any_completed = true;
                        if let Some(waker) = inner.waker.take() {
                            waker.wake();
                        }
                    }
                }
            }

            // Prune completed and dropped entries.
            if any_completed {
                let mut guard = entries.lock().unwrap();
                guard.retain(|e| {
                    let inner = e.inner.lock().unwrap();
                    !inner.completed && !inner.dropped
                });
            }

            // Sleep until a new fence is registered or the interval expires.
            if !shutdown.load(Ordering::Relaxed) {
                notify.wait(interval);
            }
        }
    }

    /// Register a fence for monitoring.
    ///
    /// Called internally by `SubmitBuilder::build` when a watcher is attached.
    pub(crate) fn watch(&self, state: Arc<WatchedFenceState>) {
        {
            let mut entries = self.entries.lock().unwrap();
            entries.push(state);
        }
        self.notify.signal();
    }

    /// Returns the number of fences currently being monitored.
    pub fn pending_count(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

impl Drop for FenceWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.notify.signal();

        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

// SAFETY: All interior state is protected by Mutex/AtomicBool.
// The thread handle is managed and joined on drop.
unsafe impl Send for FenceWatcher {}
unsafe impl Sync for FenceWatcher {}
