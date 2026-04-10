//! Deferred GPU resource destruction using timeline semaphores.
//!
//! Resources are tagged with a (semaphore, value) pair from the queue
//! they were last used on. They are destroyed only after
//! `vkGetSemaphoreCounterValue` confirms the GPU has moved past that
//! point. No concept of "frame" - works with any number of windows,
//! async compute, and independent transfer queues.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use ash::vk;
use ash::vk::Handle;

use crate::memory::allocator::{Allocation, Allocator};
use crate::device::SharedState;
use crate::diagnostic::{self, Severity, Style};
use super::timeline::QueueTimeline;

/// How to determine when a resource is safe to destroy.
#[derive(Clone)]
pub enum DeletionGuard {
    /// Safe when the timeline semaphore reaches this value.
    Timeline {
        /// The timeline semaphore to check.
        timeline: Arc<QueueTimeline>,
        /// The value that must be reached.
        value: u64,
    },
    /// Safe when this fence is signaled. Fence is NOT owned by the
    /// deletion queue - it will check status but not destroy the fence.
    Fence(vk::Fence),
    /// Safe to destroy immediately (already waited externally).
    Immediate,
}

enum PendingResource {
    Buffer {
        handle: vk::Buffer,
        allocation: Option<(Arc<dyn Allocator>, Allocation)>,
    },
    Image {
        handle: vk::Image,
        allocation: Option<(Arc<dyn Allocator>, Allocation)>,
    },
    ImageView(vk::ImageView),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    ShaderModule(vk::ShaderModule),
    RenderPass(vk::RenderPass),
    Framebuffer(vk::Framebuffer),
    Sampler(vk::Sampler),
    DescriptorPool(vk::DescriptorPool),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    Fence(vk::Fence),
    Semaphore(vk::Semaphore),
    Custom(Box<dyn FnOnce(&ash::Device) + Send>),
}

struct DeletionEntry {
    resource: PendingResource,
    guard: DeletionGuard,
}

/// Deferred resource destruction queue.
///
/// # Usage
///
/// ```text
/// let submit_value = timeline.claim_next_value();
/// // ... submit work using the resource ...
///
/// dq.retire_buffer_after(buffer, &timeline, submit_value);
/// // buffer will be destroyed after the timeline reaches submit_value
///
/// // Periodically (e.g., start of each frame):
/// dq.poll(); // destroys resources whose GPU work has finished
/// ```
pub struct DeletionQueue {
    shared: Arc<SharedState>,
    entries: Mutex<VecDeque<DeletionEntry>>,
}

impl DeletionQueue {
    /// Create a new deletion queue.
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            entries: Mutex::new(VecDeque::new()),
        }
    }

    /// Process pending entries. Destroys resources whose guard condition
    /// has been met. Call periodically (e.g., once per frame).
    ///
    /// Returns the number of resources destroyed.
    pub fn poll(&self) -> usize {
        let mut entries = self.entries.lock().unwrap();
        let mut destroyed = 0;
        let device = &self.shared.device;

        let mut timeline_cache: std::collections::HashMap<u64, u64> =
            std::collections::HashMap::new();

        // Collect indices to remove (avoiding borrow issues).
        let mut to_remove: Vec<usize> = Vec::new();

        for (i, entry) in entries.iter().enumerate() {
            let safe = match &entry.guard {
                DeletionGuard::Timeline { timeline, value } => {
                    let sem_raw = timeline.semaphore().as_raw();
                    let current = *timeline_cache.entry(sem_raw).or_insert_with(|| {
                        timeline.current_value().unwrap_or(0)
                    });
                    current >= *value
                }
                DeletionGuard::Fence(fence) => unsafe {
                    device.get_fence_status(*fence).unwrap_or(false)
                },
                DeletionGuard::Immediate => true,
            };
            if safe {
                to_remove.push(i);
            }
        }

        // Remove in reverse order to keep indices valid.
        for &i in to_remove.iter().rev() {
            let entry = entries.remove(i).unwrap();
            destroy_resource(device, entry.resource);
            destroyed += 1;
        }

        destroyed
    }

    /// Flush all entries, waiting for each guard to complete.
    pub fn flush(&self) {
        let mut entries = self.entries.lock().unwrap();
        let device = &self.shared.device;
        for entry in entries.drain(..) {
            match &entry.guard {
                DeletionGuard::Timeline { timeline, value } => {
                    let _ = timeline.wait_for_value(*value, u64::MAX);
                }
                DeletionGuard::Fence(fence) => unsafe {
                    let _ = device.wait_for_fences(&[*fence], true, u64::MAX);
                },
                DeletionGuard::Immediate => {}
            }
            destroy_resource(device, entry.resource);
        }
    }

    /// Number of entries pending.
    pub fn pending_count(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    fn enqueue(&self, resource: PendingResource, guard: DeletionGuard) {
        self.entries.lock().unwrap().push_back(DeletionEntry {
            resource,
            guard,
        });
    }

    fn destroy_resource_inner(&self, resource: &PendingResource) {
        let device = &self.shared.device;
        unsafe {
            match resource {
                PendingResource::Buffer { handle, allocation } => {
                    device.destroy_buffer(*handle, None);
                    if let Some((alloc, a)) = allocation {
                        alloc.free(a);
                    }
                }
                PendingResource::Image { handle, allocation } => {
                    device.destroy_image(*handle, None);
                    if let Some((alloc, a)) = allocation {
                        alloc.free(a);
                    }
                }
                PendingResource::ImageView(h) => device.destroy_image_view(*h, None),
                PendingResource::Pipeline(h) => device.destroy_pipeline(*h, None),
                PendingResource::PipelineLayout(h) => {
                    device.destroy_pipeline_layout(*h, None)
                }
                PendingResource::ShaderModule(h) => device.destroy_shader_module(*h, None),
                PendingResource::RenderPass(h) => device.destroy_render_pass(*h, None),
                PendingResource::Framebuffer(h) => device.destroy_framebuffer(*h, None),
                PendingResource::Sampler(h) => device.destroy_sampler(*h, None),
                PendingResource::DescriptorPool(h) => {
                    device.destroy_descriptor_pool(*h, None)
                }
                PendingResource::DescriptorSetLayout(h) => {
                    device.destroy_descriptor_set_layout(*h, None)
                }
                PendingResource::Fence(h) => device.destroy_fence(*h, None),
                PendingResource::Semaphore(h) => device.destroy_semaphore(*h, None),
                // Custom is FnOnce, can't be called through &. Need owned.
                PendingResource::Custom(_) => {} // handled below
            }
        }
    }

    // Retire methods.

    /// Retire a buffer after a timeline value is reached.
    pub fn retire_buffer_after(
        &self,
        handle: vk::Buffer,
        allocation: Option<(Arc<dyn Allocator>, Allocation)>,
        guard: DeletionGuard,
    ) {
        self.enqueue(
            PendingResource::Buffer { handle, allocation },
            guard,
        );
    }

    /// Retire an image after a timeline value is reached.
    pub fn retire_image_after(
        &self,
        handle: vk::Image,
        allocation: Option<(Arc<dyn Allocator>, Allocation)>,
        guard: DeletionGuard,
    ) {
        self.enqueue(
            PendingResource::Image { handle, allocation },
            guard,
        );
    }

    /// Retire an image view.
    pub fn retire_image_view(&self, handle: vk::ImageView, guard: DeletionGuard) {
        self.enqueue(PendingResource::ImageView(handle), guard);
    }

    /// Retire a pipeline.
    pub fn retire_pipeline(&self, handle: vk::Pipeline, guard: DeletionGuard) {
        self.enqueue(PendingResource::Pipeline(handle), guard);
    }

    /// Retire immediately (caller guarantees GPU is done).
    pub fn retire_immediate_buffer(&self, handle: vk::Buffer) {
        self.enqueue(
            PendingResource::Buffer {
                handle,
                allocation: None,
            },
            DeletionGuard::Immediate,
        );
    }

    /// Retire a pipeline layout.
    pub fn retire_pipeline_layout(&self, handle: vk::PipelineLayout, guard: DeletionGuard) {
        self.enqueue(PendingResource::PipelineLayout(handle), guard);
    }

    /// Retire a shader module.
    pub fn retire_shader_module(&self, handle: vk::ShaderModule, guard: DeletionGuard) {
        self.enqueue(PendingResource::ShaderModule(handle), guard);
    }

    /// Retire a render pass.
    pub fn retire_render_pass(&self, handle: vk::RenderPass, guard: DeletionGuard) {
        self.enqueue(PendingResource::RenderPass(handle), guard);
    }

    /// Retire a framebuffer.
    pub fn retire_framebuffer(&self, handle: vk::Framebuffer, guard: DeletionGuard) {
        self.enqueue(PendingResource::Framebuffer(handle), guard);
    }

    /// Retire a sampler.
    pub fn retire_sampler(&self, handle: vk::Sampler, guard: DeletionGuard) {
        self.enqueue(PendingResource::Sampler(handle), guard);
    }

    /// Retire a descriptor pool.
    pub fn retire_descriptor_pool(&self, handle: vk::DescriptorPool, guard: DeletionGuard) {
        self.enqueue(PendingResource::DescriptorPool(handle), guard);
    }

    /// Retire a descriptor set layout.
    pub fn retire_descriptor_set_layout(
        &self,
        handle: vk::DescriptorSetLayout,
        guard: DeletionGuard,
    ) {
        self.enqueue(PendingResource::DescriptorSetLayout(handle), guard);
    }

    /// Retire a fence.
    pub fn retire_fence(&self, handle: vk::Fence, guard: DeletionGuard) {
        self.enqueue(PendingResource::Fence(handle), guard);
    }

    /// Retire a semaphore.
    pub fn retire_semaphore(&self, handle: vk::Semaphore, guard: DeletionGuard) {
        self.enqueue(PendingResource::Semaphore(handle), guard);
    }

    /// Retire a resource with a custom destructor.
    ///
    /// The closure receives the `ash::Device` and should perform whatever
    /// cleanup is needed. Useful for extension objects or composite
    /// resources not covered by the built-in variants.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::deletion_queue::*; use ash::vk;
    /// # fn example(dq: &DeletionQueue, accel: vk::AccelerationStructureKHR,
    /// #            guard: DeletionGuard) {
    /// dq.retire_custom(
    ///     "AccelerationStructure",
    ///     accel,
    ///     guard,
    ///     |device| { /* destroy via extension fn */ },
    /// );
    /// # }
    /// ```
    pub fn retire_custom<F>(
        &self,
        _name: &'static str,
        _handle: impl ash::vk::Handle,
        guard: DeletionGuard,
        destroy: F,
    ) where
        F: FnOnce(&ash::Device) + Send + 'static,
    {
        self.enqueue(PendingResource::Custom(Box::new(destroy)), guard);
    }
}

impl Drop for DeletionQueue {
    fn drop(&mut self) {
        let entries = self.entries.get_mut().unwrap();
        if entries.is_empty() {
            return;
        }

        let s = Style::detect();
        let mut o = String::new();
        diagnostic::write_header(
            &mut o,
            &s,
            &Severity::Warning,
            "IGN-Q001",
            &format!(
                "{} resource(s) in DeletionQueue at shutdown, flushing",
                entries.len()
            ),
        );
        eprint!("{o}");

        let device = &self.shared.device;
        for entry in entries.drain(..) {
            // Wait for the guard.
            match &entry.guard {
                DeletionGuard::Timeline { timeline, value } => {
                    let _ = timeline.wait_for_value(*value, u64::MAX);
                }
                DeletionGuard::Fence(fence) => unsafe {
                    let _ = device.wait_for_fences(&[*fence], true, u64::MAX);
                },
                DeletionGuard::Immediate => {}
            }
            // Destroy inline instead of calling self method.
            destroy_resource(device, entry.resource);
        }
    }
    
}

/// Destroy a pending resource. Free function to avoid borrow conflicts.
fn destroy_resource(device: &ash::Device, resource: PendingResource) {
    unsafe {
        match resource {
            PendingResource::Buffer { handle, allocation } => {
                device.destroy_buffer(handle, None);
                if let Some((alloc, a)) = allocation {
                    alloc.free(&a);
                }
            }
            PendingResource::Image { handle, allocation } => {
                device.destroy_image(handle, None);
                if let Some((alloc, a)) = allocation {
                    alloc.free(&a);
                }
            }
            PendingResource::ImageView(h) => device.destroy_image_view(h, None),
            PendingResource::Pipeline(h) => device.destroy_pipeline(h, None),
            PendingResource::PipelineLayout(h) => device.destroy_pipeline_layout(h, None),
            PendingResource::ShaderModule(h) => device.destroy_shader_module(h, None),
            PendingResource::RenderPass(h) => device.destroy_render_pass(h, None),
            PendingResource::Framebuffer(h) => device.destroy_framebuffer(h, None),
            PendingResource::Sampler(h) => device.destroy_sampler(h, None),
            PendingResource::DescriptorPool(h) => device.destroy_descriptor_pool(h, None),
            PendingResource::DescriptorSetLayout(h) => {
                device.destroy_descriptor_set_layout(h, None)
            }
            PendingResource::Fence(h) => device.destroy_fence(h, None),
            PendingResource::Semaphore(h) => device.destroy_semaphore(h, None),
            PendingResource::Custom(f) => f(device),
        }
    }
}