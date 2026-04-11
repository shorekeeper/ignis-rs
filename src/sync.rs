//! Per-frame synchronization management.
//!
//! [`FrameSync`] manages a ring buffer of synchronization primitives for
//! N frames in flight - the standard pattern for double/triple buffering.
//!
//! # Frame Lifecycle
//!
//! ```text
//! loop {
//!     let ctx = frame_sync.begin_frame()?;  // waits on fence[N], resets it
//!     // ... acquire swapchain image using ctx.image_available_semaphore()
//!     // ... record commands
//!     // ... submit with wait=image_available, signal=render_finished, fence=ctx.fence()
//!     // ... present with wait=render_finished
//!     frame_sync.advance();                  // N = (N + 1) % frames_in_flight
//! }
//! ```

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::Result;

/// Manages per-frame synchronization primitives for N frames in flight.
///
/// Allocates and owns:
/// - One fence per frame (created signaled so the first `begin_frame` does not deadlock)
/// - One "image available" semaphore per frame
/// - One "render finished" semaphore per frame
///
/// Thread-safe: the current frame index is tracked atomically.
pub struct FrameSync {
    shared: Arc<SharedState>,
    frames_in_flight: u32,
    current_frame: AtomicU32,
    fences: Vec<vk::Fence>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
}

impl FrameSync {
    /// Create a new frame synchronization manager.
    ///
    /// # Arguments
    ///
    /// * `shared` - Device state
    /// * `frames_in_flight` - Number of frames that may be processed concurrently
    ///   (typically 2 or 3)
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if fence or semaphore creation fails.
    pub(crate) fn new(shared: Arc<SharedState>, frames_in_flight: u32) -> Result<Self> {
        let device = &shared.device;

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        let mut fences = Vec::with_capacity(frames_in_flight as usize);
        let mut image_available = Vec::with_capacity(frames_in_flight as usize);
        let mut render_finished = Vec::with_capacity(frames_in_flight as usize);

        for _ in 0..frames_in_flight {
            // SAFETY: create info is valid, device is valid.
            unsafe {
                fences.push(device.create_fence(&fence_info, None)?);
                image_available.push(device.create_semaphore(&semaphore_info, None)?);
                render_finished.push(device.create_semaphore(&semaphore_info, None)?);
            }
        }

        Ok(Self {
            shared,
            frames_in_flight,
            current_frame: AtomicU32::new(0),
            fences,
            image_available_semaphores: image_available,
            render_finished_semaphores: render_finished,
        })
    }

    /// The number of frames in flight this sync manager was created for.
    #[inline]
    pub fn frames_in_flight(&self) -> u32 {
        self.frames_in_flight
    }

    /// The current frame index (0-based, wraps around at `frames_in_flight`).
    #[inline]
    pub fn current_frame_index(&self) -> u32 {
        self.current_frame.load(Ordering::Relaxed)
    }

    /// Begin a new frame.
    ///
    /// Waits for the current frame's fence to be signaled (meaning the GPU
    /// finished processing a previous submission that used this slot), then
    /// resets the fence to unsignaled.
    ///
    /// Returns a [`FrameContext`] providing access to this frame's
    /// synchronization primitives.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if the fence wait or reset fails.
    pub fn begin_frame(&self) -> Result<FrameContext> {
        let idx = self.current_frame.load(Ordering::Relaxed) as usize;
        let fence = self.fences[idx];

        // SAFETY: fence is valid and was created by us.
        unsafe {
            self.shared
                .device
                .wait_for_fences(&[fence], true, u64::MAX)?;
            self.shared.device.reset_fences(&[fence])?;
        }

        Ok(FrameContext {
            frame_index: idx as u32,
            fence,
            image_available: self.image_available_semaphores[idx],
            render_finished: self.render_finished_semaphores[idx],
        })
    }

    /// Advance to the next frame slot.
    ///
    /// Should be called after presenting the current frame.
    pub fn advance(&self) {
        let next = (self.current_frame.load(Ordering::Relaxed) + 1) % self.frames_in_flight;
        self.current_frame.store(next, Ordering::Relaxed);
    }

    /// Wait for all frames to finish processing.
    ///
    /// Useful before shutdown or resource recreation.
    pub fn wait_all(&self) -> Result<()> {
        // SAFETY: all fences are valid.
        unsafe {
            self.shared
                .device
                .wait_for_fences(&self.fences, true, u64::MAX)?;
        }
        Ok(())
    }

    /// Get the fence for a specific frame index.
    ///
    /// # Panics
    ///
    /// Panics if `frame_index >= frames_in_flight`.
    pub fn fence(&self, frame_index: u32) -> vk::Fence {
        self.fences[frame_index as usize]
    }
}

impl Drop for FrameSync {
    fn drop(&mut self) {
        let device = &self.shared.device;
        unsafe {
            // Wait for everything to settle.
            let _ = device.wait_for_fences(&self.fences, true, u64::MAX);

            for &fence in &self.fences {
                device.destroy_fence(fence, None);
            }
            for &sem in &self.image_available_semaphores {
                device.destroy_semaphore(sem, None);
            }
            for &sem in &self.render_finished_semaphores {
                device.destroy_semaphore(sem, None);
            }
        }
    }
}

/// Synchronization primitives for a single frame.
///
/// Returned by [`FrameSync::begin_frame`]. Contains the fence, image
/// available semaphore, and render finished semaphore for this frame slot.
#[derive(Debug, Clone, Copy)]
pub struct FrameContext {
    frame_index: u32,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
}

impl FrameContext {
    /// Frame slot index (0-based).
    #[inline]
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Fence for this frame. Pass to `vkQueueSubmit` so ignis can track
    /// when the GPU finishes processing this frame.
    #[inline]
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }

    /// Semaphore signaled when a swapchain image is available.
    /// Use as a wait semaphore in your submit info.
    #[inline]
    pub fn image_available_semaphore(&self) -> vk::Semaphore {
        self.image_available
    }

    /// Semaphore signaled when rendering commands for this frame complete.
    /// Use as a wait semaphore in your present info.
    #[inline]
    pub fn render_finished_semaphore(&self) -> vk::Semaphore {
        self.render_finished
    }
}

/// Reusable fence pool to avoid per-submission `vkCreateFence`/`vkDestroyFence`.
///
/// Fences are reset when released back to the pool, ready for immediate reuse.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::*; use ash::vk;
/// # fn example(ignis: &Ignis) -> Result<()> {
/// let pool = ignis.create_fence_pool();
/// let fence = pool.acquire()?;
/// // ... submit with fence ...
/// // ... wait for fence ...
/// pool.release(fence)?;
/// # Ok(())
/// # }
/// ```
pub struct FencePool {
    shared: Arc<SharedState>,
    available: std::sync::Mutex<Vec<vk::Fence>>,
}

impl FencePool {
    /// Create a new, empty fence pool.
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            available: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Acquire a fence from the pool, creating a new one if empty.
    ///
    /// The fence is returned in the unsignaled state.
    pub fn acquire(&self) -> Result<vk::Fence> {
        let mut pool = self.available.lock().unwrap();
        if let Some(fence) = pool.pop() {
            Ok(fence)
        } else {
            let ci = vk::FenceCreateInfo::default();
            let fence = unsafe { self.shared.device.create_fence(&ci, None)? };
            Ok(fence)
        }
    }

    /// Release a signaled fence back to the pool.
    ///
    /// The fence is reset to unsignaled before being added to the pool.
    pub fn release(&self, fence: vk::Fence) -> Result<()> {
        unsafe { self.shared.device.reset_fences(&[fence])? };
        self.available.lock().unwrap().push(fence);
        Ok(())
    }

    /// Number of fences currently available in the pool.
    pub fn available_count(&self) -> usize {
        self.available.lock().unwrap().len()
    }
}

impl Drop for FencePool {
    fn drop(&mut self) {
        let pool = self.available.get_mut().unwrap();
        for &fence in pool.iter() {
            unsafe { self.shared.device.destroy_fence(fence, None) };
        }
    }
}