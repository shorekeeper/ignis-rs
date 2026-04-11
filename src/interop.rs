//! Cross-engine interoperability primitives.
//!
//! When ignis coexists with wgpu, vulkano, or another Vulkan consumer
//! on the same device, they must coordinate on:
//!
//! - **Queue access**: two engines must not submit to the same
//!   `VkQueue` concurrently
//! - **Resource synchronization**: semaphores/fences between engines
//! - **Memory**: each engine manages its own allocations, but raw
//!   handles (`VkBuffer`, `VkImage`) can be shared
//!
//! This module provides [`QueueBroker`] for safe queue sharing and
//! [`InteropSync`] for cross-engine sync.

use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;

use crate::device::SharedState;
use crate::error::Result;

/// A guarded queue handle. Holds the mutex lock for the duration of
/// its lifetime, guaranteeing exclusive access.
///
/// Returned by [`QueueBroker::acquire`].
pub struct QueueGuard<'a> {
    queue: vk::Queue,
    _lock: MutexGuard<'a, ()>,
}

impl QueueGuard<'_> {
    /// Get the raw queue handle. Safe to submit to while this guard exists.
    pub fn handle(&self) -> vk::Queue {
        self.queue
    }
}

/// Mediates access to a shared `VkQueue` between ignis and an external
/// engine (wgpu, vulkano, etc.).
///
/// Both sides must acquire the queue through the broker before submitting.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::interop::*; use ash::vk;
/// # fn example(broker: &QueueBroker, device: &ash::Device,
/// #            cmd: vk::CommandBuffer, fence: vk::Fence) {
/// // Engine A (ignis):
/// {
///     let guard = broker.acquire();
///     let submits = [vk::SubmitInfo::default()
///         .command_buffers(std::slice::from_ref(&cmd))];
///     unsafe { device.queue_submit(guard.handle(), &submits, fence).unwrap() };
/// } // lock released here
///
/// // Engine B (wgpu) can now safely submit to the same queue.
/// # }
/// ```
///
/// # Alternative: Separate Queues
///
/// If the device supports multiple queues in the same family, prefer
/// giving each engine its own queue (`queue_index` 0 vs 1). This avoids
/// the mutex entirely. Use the broker only when a single queue must
/// be shared.
pub struct QueueBroker {
    queue: vk::Queue,
    family_index: u32,
    queue_index: u32,
    lock: Mutex<()>,
}

impl QueueBroker {
    /// Create a broker for the given queue.
    pub fn new(queue: vk::Queue, family_index: u32, queue_index: u32) -> Self {
        Self {
            queue,
            family_index,
            queue_index,
            lock: Mutex::new(()),
        }
    }

    /// Acquire exclusive access to the queue.
    ///
    /// Blocks until the queue is available. The returned guard holds
    /// the lock and releases it on drop.
    pub fn acquire(&self) -> QueueGuard<'_> {
        let lock = self.lock.lock().unwrap();
        QueueGuard {
            queue: self.queue,
            _lock: lock,
        }
    }

    /// Try to acquire without blocking. Returns `None` if the queue
    /// is currently held by another engine.
    pub fn try_acquire(&self) -> Option<QueueGuard<'_>> {
        self.lock.try_lock().ok().map(|lock| QueueGuard {
            queue: self.queue,
            _lock: lock,
        })
    }

    /// Queue family index.
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    /// Queue index.
    pub fn queue_index(&self) -> u32 {
        self.queue_index
    }
}

// SAFETY: Mutex provides synchronized access to the queue.
unsafe impl Send for QueueBroker {}
unsafe impl Sync for QueueBroker {}

/// A pair of semaphores for synchronizing work between two engines.
///
/// Engine A signals `a_done` after its work completes.
/// Engine B waits on `a_done` before starting, then signals `b_done`.
/// Engine A waits on `b_done` before reusing shared resources.
///
/// ```text
/// Engine A:  submit(signal=a_done) -----> wait on b_done ---->
/// Engine B:  wait on a_done -----> submit(signal=b_done) ---->
/// ```
pub struct InteropSync {
    shared: Arc<SharedState>,
    /// Semaphore signaled by engine A.
    pub a_done: vk::Semaphore,
    /// Semaphore signaled by engine B.
    pub b_done: vk::Semaphore,
}

impl InteropSync {
    /// Create a new interop sync pair.
    pub fn new(shared: Arc<SharedState>) -> Result<Self> {
        let ci = vk::SemaphoreCreateInfo::default();
        let a = unsafe { shared.device.create_semaphore(&ci, None)? };
        let b = unsafe { shared.device.create_semaphore(&ci, None)? };
        Ok(Self {
            shared,
            a_done: a,
            b_done: b,
        })
    }
}

impl Drop for InteropSync {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_semaphore(self.a_done, None);
            self.shared.device.destroy_semaphore(self.b_done, None);
        }
    }
}