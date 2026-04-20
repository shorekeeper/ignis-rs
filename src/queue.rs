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
use crate::tracking::timeline::{QueueTimeline, TimelineWatcher};
use crate::tracking::watcher::FenceWatcher;
use crate::QueueType;

/// A thread-safe asynchronous Vulkan queue.
pub struct AsyncQueue {
    pub(crate) shared: Arc<SharedState>,
    handle: Mutex<vk::Queue>,
    family_index: u32,
    queue_index: u32,
    capabilities: vk::QueueFlags,
    /// Timeline semaphore for this queue, if Vulkan 1.2+.
    pub(crate) timeline: Option<Arc<QueueTimeline>>,
}

impl AsyncQueue {
    pub(crate) fn new(
        shared: Arc<SharedState>,
        handle: vk::Queue,
        family_index: u32,
        queue_index: u32,
        capabilities: vk::QueueFlags,
        timeline: Option<Arc<QueueTimeline>>,
    ) -> Self {
        Self {
            shared,
            handle: Mutex::new(handle),
            family_index,
            queue_index,
            capabilities,
            timeline,
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

    /// The timeline semaphore for this queue, if available (Vulkan 1.2+).
    pub fn timeline(&self) -> Option<&Arc<QueueTimeline>> {
        self.timeline.as_ref()
    }

    /// Begin building a queue submission.
    pub fn submit(&self) -> SubmitBuilder<'_> {
        SubmitBuilder {
            queue: self,
            command_buffers: Vec::new(),
            wait_semaphores: Vec::new(),
            wait_stages: Vec::new(),
            signal_semaphores: Vec::new(),
            timeline_watcher: None,
            fence_watcher: None,
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
        // Capture backtrace so the validation layer callback can show the
        // user where submit was called from when it reports async errors.
        #[cfg(feature = "debug-tools")]
        let _bt_guard = crate::debug::validation_forensic::SubmitBacktraceGuard::new();

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
    timeline_watcher: Option<Arc<TimelineWatcher>>,
    fence_watcher: Option<Arc<FenceWatcher>>,
}

impl SubmitBuilder<'_> {
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

    /// Attach a timeline watcher for efficient async completion (Vulkan 1.2+).
    pub fn with_timeline_watcher(mut self, watcher: &Arc<TimelineWatcher>) -> Self {
        self.timeline_watcher = Some(Arc::clone(watcher));
        self
    }

    /// Attach a legacy fence watcher (Vulkan 1.1 fallback).
    /// Also available as `with_watcher` for backward compatibility.
    pub fn with_fence_watcher(mut self, watcher: &Arc<FenceWatcher>) -> Self {
        self.fence_watcher = Some(Arc::clone(watcher));
        self
    }

    /// Backward-compatible alias for [`with_fence_watcher`](Self::with_fence_watcher).
    pub fn with_watcher(self, watcher: &Arc<FenceWatcher>) -> Self {
        self.with_fence_watcher(watcher)
    }

    /// Submit the accumulated work to the queue.
    ///
    /// If the queue has a timeline semaphore (Vulkan 1.2+), uses timeline
    /// signaling with no fence. Otherwise falls back to a dedicated fence.
    ///
    /// Returns a [`GpuFuture`] tracking completion.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if fence creation or queue submission fails.
    pub fn build(self) -> Result<GpuFuture> {
        // Capture submit backtrace for validation layer cross-reference.
        #[cfg(feature = "debug-tools")]
        let _bt_guard = crate::debug::validation_forensic::SubmitBacktraceGuard::new();
        
        let device = &self.queue.shared.device;

        // Timeline path (Vulkan 1.2+).
        if let Some(timeline) = &self.queue.timeline {
            let target_value = timeline.claim_next_value();

            let mut signal_sems = self.signal_semaphores.clone();
            signal_sems.push(timeline.semaphore());

            let wait_values = vec![0u64; self.wait_semaphores.len()];
            let mut signal_values = vec![0u64; self.signal_semaphores.len()];
            signal_values.push(target_value);

            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);

            let submit_info = vk::SubmitInfo::default()
                .push_next(&mut timeline_info)
                .command_buffers(&self.command_buffers)
                .wait_semaphores(&self.wait_semaphores)
                .wait_dst_stage_mask(&self.wait_stages)
                .signal_semaphores(&signal_sems);

            let queue_handle = self.queue.handle.lock().unwrap();
            let result = unsafe {
                device.queue_submit(
                    *queue_handle,
                    std::slice::from_ref(&submit_info),
                    vk::Fence::null(),
                )
            };
            drop(queue_handle);

            if let Err(e) = result {
                return Err(Error::Vulkan(e));
            }

            return Ok(GpuFuture {
                shared: Arc::clone(&self.queue.shared),
                inner: FutureKind::Timeline {
                    timeline: Arc::clone(timeline),
                    target_value,
                    watcher: self.timeline_watcher,
                },
                completed: false,
            });
        }

        // Fence path (Vulkan 1.1 fallback).
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

        Ok(GpuFuture {
            shared: Arc::clone(&self.queue.shared),
            inner: FutureKind::Fence { fence },
            completed: false,
        })
    }
}

enum FutureKind {
    Timeline {
        timeline: Arc<QueueTimeline>,
        target_value: u64,
        watcher: Option<Arc<TimelineWatcher>>,
    },
    Fence {
        fence: vk::Fence,
    },
}

/// A future representing in-flight GPU work.
///
/// # Timeline mode (Vulkan 1.2+)
///
/// Uses a timeline semaphore. `drop()` is free - no blocking, no fence
/// cleanup. Safe to drop from any async executor.
///
/// # Fence mode (Vulkan 1.1 fallback)
///
/// Uses a dedicated fence. `drop()` blocks until the fence signals.
/// Avoid dropping from async executor worker threads.
pub struct GpuFuture {
    shared: Arc<SharedState>,
    inner: FutureKind,
    completed: bool,
}

impl GpuFuture {
    /// Non-blocking completion check.
    pub fn is_complete(&self) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        match &self.inner {
            FutureKind::Timeline {
                timeline,
                target_value,
                ..
            } => {
                let current = timeline.current_value()?;
                Ok(current >= *target_value)
            }
            FutureKind::Fence { fence } => {
                let signaled = unsafe { self.shared.device.get_fence_status(*fence)? };
                Ok(signaled)
            }
        }
    }

    /// Blocking wait.
    pub fn wait(&self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        match &self.inner {
            FutureKind::Timeline {
                timeline,
                target_value,
                ..
            } => {
                timeline.wait_for_value(*target_value, u64::MAX)?;
                Ok(())
            }
            FutureKind::Fence { fence } => {
                unsafe {
                    self.shared
                        .device
                        .wait_for_fences(&[*fence], true, u64::MAX)?;
                }
                Ok(())
            }
        }
    }

    /// Blocking wait with timeout.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        let nanos = timeout.as_nanos().min(u128::from(u64::MAX)) as u64;
        match &self.inner {
            FutureKind::Timeline {
                timeline,
                target_value,
                ..
            } => timeline.wait_for_value(*target_value, nanos),
            FutureKind::Fence { fence } => {
                match unsafe { self.shared.device.wait_for_fences(&[*fence], true, nanos) } {
                    Ok(()) => Ok(true),
                    Err(vk::Result::TIMEOUT) => Ok(false),
                    Err(e) => Err(Error::Vulkan(e)),
                }
            }
        }
    }

    /// Get the timeline target value (if timeline mode).
    pub fn timeline_value(&self) -> Option<u64> {
        match &self.inner {
            FutureKind::Timeline { target_value, .. } => Some(*target_value),
            _ => None,
        }
    }

    /// Get the timeline semaphore (if timeline mode).
    pub fn timeline_semaphore(&self) -> Option<vk::Semaphore> {
        match &self.inner {
            FutureKind::Timeline { timeline, .. } => Some(timeline.semaphore()),
            _ => None,
        }
    }

    /// Get the fence handle (if fence mode).
    pub fn fence(&self) -> Option<vk::Fence> {
        match &self.inner {
            FutureKind::Fence { fence } => Some(*fence),
            _ => None,
        }
    }
}

impl Future for GpuFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            return Poll::Ready(Ok(()));
        }

        match &self.inner {
            FutureKind::Timeline {
                timeline,
                target_value,
                watcher,
            } => {
                let current = match timeline.current_value() {
                    Ok(v) => v,
                    Err(e) => {
                        self.completed = true;
                        return Poll::Ready(Err(e));
                    }
                };

                if current >= *target_value {
                    self.completed = true;
                    return Poll::Ready(Ok(()));
                }

                // Register waker with the timeline watcher.
                if let Some(w) = watcher {
                    w.register(timeline.semaphore(), *target_value, cx.waker().clone());
                    Poll::Pending
                } else {
                    // No watcher: busy-wait fallback.
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            FutureKind::Fence { fence } => {
                match unsafe { self.shared.device.get_fence_status(*fence) } {
                    Ok(true) => {
                        self.completed = true;
                        Poll::Ready(Ok(()))
                    }
                    Ok(false) => {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                    Err(e) => {
                        self.completed = true;
                        Poll::Ready(Err(Error::Vulkan(e)))
                    }
                }
            }
        }
    }
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        match &self.inner {
            FutureKind::Timeline { .. } => {
                // No-op. Timeline value will be reached eventually.
                // No resources to clean up.
            }
            FutureKind::Fence { fence } => {
                if !self.completed {
                    unsafe {
                        let _ = self
                            .shared
                            .device
                            .wait_for_fences(&[*fence], true, u64::MAX);
                    }
                }
                unsafe {
                    self.shared.device.destroy_fence(*fence, None);
                }
            }
        }
    }
}

unsafe impl Send for GpuFuture {}
unsafe impl Sync for GpuFuture {}
