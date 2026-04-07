//! Asynchronous queue submission and GPU futures.
//!
//! [`AsyncQueue`] wraps a `VkQueue` with a mutex for thread-safe submission
//! and provides [`SubmitBuilder`] for ergonomic work submission.
//! Each submission returns a [`GpuFuture`] that can be polled, awaited,
//! or synchronously waited on.
//!
//! # Efficient Async
//!
//! By default, [`GpuFuture`]'s `Future::poll` uses a spin-loop fallback.
//! For production async usage, attach a [`FenceWatcher`] via
//! [`SubmitBuilder::with_watcher`] to get sleep-based fence monitoring.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};
use crate::watcher::{FenceWatcher, WatchedFenceInner, WatchedFenceState};
use crate::QueueType;

/// A thread-safe asynchronous Vulkan queue.
///
/// The underlying `VkQueue` is protected by a mutex satisfying the Vulkan
/// spec's external synchronization requirement.
pub struct AsyncQueue {
    pub(crate) shared: Arc<SharedState>,
    handle: Mutex<vk::Queue>,
    family_index: u32,
    queue_index: u32,
    capabilities: vk::QueueFlags,
}

impl AsyncQueue {
    pub(crate) fn new(
        shared: Arc<SharedState>,
        handle: vk::Queue,
        family_index: u32,
        queue_index: u32,
        capabilities: vk::QueueFlags,
    ) -> Self {
        Self {
            shared,
            handle: Mutex::new(handle),
            family_index,
            queue_index,
            capabilities,
        }
    }

    /// Returns the queue family index.
    #[inline]
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    /// Returns the queue index within its family.
    #[inline]
    pub fn queue_index(&self) -> u32 {
        self.queue_index
    }

    /// Returns the capability flags of this queue's family.
    #[inline]
    pub fn capabilities(&self) -> vk::QueueFlags {
        self.capabilities
    }

    /// Check whether this queue supports the given type.
    #[inline]
    pub fn supports(&self, queue_type: QueueType) -> bool {
        self.capabilities.contains(queue_type.required_flags())
    }

    /// Begin building a queue submission.
    pub fn submit(&self) -> SubmitBuilder<'_> {
        SubmitBuilder {
            queue: self,
            command_buffers: Vec::new(),
            wait_semaphores: Vec::new(),
            wait_stages: Vec::new(),
            signal_semaphores: Vec::new(),
            watcher: None,
        }
    }

    /// Submit a single command buffer with no synchronization.
    pub fn submit_simple(&self, command_buffer: vk::CommandBuffer) -> Result<GpuFuture> {
        self.submit().command_buffer(command_buffer).build()
    }

    /// Submit with a caller-provided fence. No [`GpuFuture`] returned.
    ///
    /// # Safety
    ///
    /// The fence must be valid and unsignaled.
    pub unsafe fn submit_raw(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<()> {
        let queue = self.handle.lock().unwrap();
        self.shared.device.queue_submit(*queue, submits, fence)?;
        Ok(())
    }

    /// Wait for all submitted operations on this queue to complete.
    pub fn wait_idle(&self) -> Result<()> {
        let queue = self.handle.lock().unwrap();
        unsafe { self.shared.device.queue_wait_idle(*queue)? };
        Ok(())
    }
}

/// Builder for constructing a queue submission.
pub struct SubmitBuilder<'a> {
    queue: &'a AsyncQueue,
    command_buffers: Vec<vk::CommandBuffer>,
    wait_semaphores: Vec<vk::Semaphore>,
    wait_stages: Vec<vk::PipelineStageFlags>,
    signal_semaphores: Vec<vk::Semaphore>,
    watcher: Option<Arc<FenceWatcher>>,
}

impl<'a> SubmitBuilder<'a> {
    /// Add a command buffer to the submission.
    pub fn command_buffer(mut self, buffer: vk::CommandBuffer) -> Self {
        self.command_buffers.push(buffer);
        self
    }

    /// Add multiple command buffers.
    pub fn command_buffers(mut self, buffers: &[vk::CommandBuffer]) -> Self {
        self.command_buffers.extend_from_slice(buffers);
        self
    }

    /// Add a wait semaphore with a pipeline stage mask.
    pub fn wait_semaphore(
        mut self,
        semaphore: vk::Semaphore,
        stage_mask: vk::PipelineStageFlags,
    ) -> Self {
        self.wait_semaphores.push(semaphore);
        self.wait_stages.push(stage_mask);
        self
    }

    /// Add a signal semaphore.
    pub fn signal_semaphore(mut self, semaphore: vk::Semaphore) -> Self {
        self.signal_semaphores.push(semaphore);
        self
    }

    /// Attach a [`FenceWatcher`] for efficient async completion.
    ///
    /// When a watcher is attached, the resulting [`GpuFuture`] registers
    /// its fence with the watcher's background thread instead of busy-
    /// waiting on poll.
    pub fn with_watcher(mut self, watcher: &Arc<FenceWatcher>) -> Self {
        self.watcher = Some(Arc::clone(watcher));
        self
    }

    /// Submit the work and return a [`GpuFuture`] tracking completion.
    pub fn build(self) -> Result<GpuFuture> {
        let device = &self.queue.shared.device;

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&self.command_buffers)
            .wait_semaphores(&self.wait_semaphores)
            .wait_dst_stage_mask(&self.wait_stages)
            .signal_semaphores(&self.signal_semaphores);

        let queue_handle = self.queue.handle.lock().unwrap();

        let result = unsafe {
            device.queue_submit(*queue_handle, std::slice::from_ref(&submit_info), fence)
        };

        drop(queue_handle);

        if let Err(e) = result {
            unsafe { device.destroy_fence(fence, None) };
            return Err(Error::Vulkan(e));
        }

        // If a watcher is attached, register the fence for monitoring.
        let watched = self.watcher.as_ref().map(|w| {
            let state = Arc::new(WatchedFenceState {
                fence,
                inner: Mutex::new(WatchedFenceInner {
                    completed: false,
                    dropped: false,
                    waker: None,
                    error: None,
                }),
            });
            w.watch(Arc::clone(&state));
            state
        });

        Ok(GpuFuture {
            shared: Arc::clone(&self.queue.shared),
            fence,
            completed: false,
            watched,
        })
    }
}

/// A future representing an in-flight GPU operation.
///
/// Tracks completion via a Vulkan fence. Supports three usage patterns:
///
/// - **Blocking**: [`wait`](GpuFuture::wait) or [`wait_timeout`](GpuFuture::wait_timeout)
/// - **Polling**: [`is_complete`](GpuFuture::is_complete)
/// - **Async**: `.await` via the [`Future`] implementation
///
/// # Async Behavior
///
/// Without a [`FenceWatcher`], the `Future` implementation uses a busy-wait
/// fallback (immediately re-schedules on each poll), consuming one CPU core.
///
/// With a `FenceWatcher` (attached via [`SubmitBuilder::with_watcher`]),
/// the future registers with the watcher thread which periodically checks
/// fence status and wakes the task only when the fence signals.
///
/// # Drop Safety
///
/// On drop, the future blocks until the fence is signaled to prevent
/// destroying an in-use fence. If a watcher is attached, the `dropped`
/// flag is set first under its per-entry lock, ensuring the watcher
/// never accesses the fence after destruction.
pub struct GpuFuture {
    shared: Arc<SharedState>,
    fence: vk::Fence,
    completed: bool,
    watched: Option<Arc<WatchedFenceState>>,
}

impl GpuFuture {
    /// Check whether the GPU work has completed without blocking.
    pub fn is_complete(&self) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        let signaled = unsafe { self.shared.device.get_fence_status(self.fence)? };
        Ok(signaled)
    }

    /// Block until the GPU work completes.
    pub fn wait(&self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        unsafe {
            self.shared
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        Ok(())
    }

    /// Block with a timeout. Returns `Ok(true)` if completed, `Ok(false)`
    /// on timeout.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        let nanos = timeout.as_nanos().min(u64::MAX as u128) as u64;
        match unsafe {
            self.shared
                .device
                .wait_for_fences(&[self.fence], true, nanos)
        } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(Error::Vulkan(e)),
        }
    }

    /// Get the raw fence handle.
    #[inline]
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl Future for GpuFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            return Poll::Ready(Ok(()));
        }

        // Check the watcher state first (if attached) for early completion
        // detected by the background thread. We extract the result under
        // the lock, then release the borrow on self.watched before mutating
        // self.completed.
        let watcher_result = self.watched.as_ref().and_then(|watched| {
            let inner = watched.inner.lock().unwrap();
            if inner.completed {
                Some(inner.error)
            } else {
                None
            }
        });

        if let Some(maybe_err) = watcher_result {
            self.completed = true;
            return match maybe_err {
                Some(e) => Poll::Ready(Err(Error::Vulkan(e))),
                None => Poll::Ready(Ok(())),
            };
        }

        // Non-blocking fence check via Vulkan API.
        match unsafe { self.shared.device.get_fence_status(self.fence) } {
            Ok(true) => {
                // Mark the watcher entry as completed so it gets pruned.
                {
                    if let Some(watched) = &self.watched {
                        let mut inner = watched.inner.lock().unwrap();
                        inner.completed = true;
                    }
                }
                // Now safe to mutate self - the immutable borrow above
                // ended with the block.
                self.completed = true;
                Poll::Ready(Ok(()))
            }
            Ok(false) => {
                // Fence not yet signaled.
                if let Some(watched) = &self.watched {
                    // Register the waker with the background thread.
                    // The watcher will call waker.wake() when the fence
                    // signals, so we do NOT re-schedule ourselves here.
                    let mut inner = watched.inner.lock().unwrap();
                    inner.waker = Some(cx.waker().clone());
                    Poll::Pending
                } else {
                    // No watcher attached - busy-wait fallback.
                    // Immediately re-schedule so the executor polls again.
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Err(e) => {
                self.completed = true;
                Poll::Ready(Err(Error::Vulkan(e)))
            }
        }
    }
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        // Signal the watcher to stop touching this fence.
        if let Some(watched) = &self.watched {
            let mut inner = watched.inner.lock().unwrap();
            inner.dropped = true;
            // Release the lock BEFORE blocking on the fence so the watcher
            // thread is not starved.
            drop(inner);
        }

        if !self.completed {
            unsafe {
                let _ = self
                    .shared
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX);
            }
        }

        unsafe {
            self.shared.device.destroy_fence(self.fence, None);
        }
    }
}

unsafe impl Send for GpuFuture {}
unsafe impl Sync for GpuFuture {}
