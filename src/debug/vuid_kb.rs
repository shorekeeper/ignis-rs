//! Knowledge base of Vulkan usage ID (VUID) explanations.
//!
//! Maps VUID numeric suffixes to human-readable descriptions and
//! ignis-specific fix suggestions. Populated manually as VUIDs are
//! encountered in practice; coverage is incomplete by design.
//!
//! Add a new VUID by appending to the static slice at the bottom of
//! this file, or register runtime entries via `register_runtime_entry`.
//!
//! Look up via `lookup(suffix)`, which checks runtime entries first
//! and falls back to the static base.

use std::sync::OnceLock;
use std::sync::RwLock;

use super::validation_forensic::{DiagnosticCategory, KnowledgeEntry};

/// Runtime-registrable knowledge entries. Checked before the static base.
///
/// Key is the VUID numeric suffix. Uses `String` storage because entries
/// may come from user code at runtime and cannot be `&'static`.
pub struct RuntimeEntry {
    pub vuid_suffix: String,
    pub title: String,
    pub category: DiagnosticCategory,
    pub what_happened: String,
    pub why_rejected: String,
    pub ignis_fix: String,
    pub spec_section: String,
}

static RUNTIME_ENTRIES: OnceLock<RwLock<Vec<RuntimeEntry>>> = OnceLock::new();

fn runtime_slot() -> &'static RwLock<Vec<RuntimeEntry>> {
    RUNTIME_ENTRIES.get_or_init(|| RwLock::new(Vec::new()))
}

/// Add a knowledge entry at runtime. Useful for application-specific
/// VUIDs or for shipping updated coverage without a library rebuild.
pub fn register_runtime_entry(entry: RuntimeEntry) {
    runtime_slot().write().unwrap().push(entry);
}

/// Clear all runtime entries. Static base is unaffected.
pub fn clear_runtime_entries() {
    runtime_slot().write().unwrap().clear();
}

/// Snapshot of a knowledge entry for display. Owns its strings so it
/// can be returned across the static/runtime boundary uniformly.
pub struct KnowledgeLookup {
    pub vuid_suffix: String,
    pub title: String,
    pub category: DiagnosticCategory,
    pub what_happened: String,
    pub why_rejected: String,
    pub ignis_fix: String,
    pub spec_section: String,
}

impl KnowledgeLookup {
    fn from_static(e: &'static KnowledgeEntry) -> Self {
        Self {
            vuid_suffix: e.vuid_suffix.to_string(),
            title: e.title.to_string(),
            category: e.category,
            what_happened: e.what_happened.to_string(),
            why_rejected: e.why_rejected.to_string(),
            ignis_fix: e.ignis_fix.to_string(),
            spec_section: e.spec_section.to_string(),
        }
    }

    fn from_runtime(e: &RuntimeEntry) -> Self {
        Self {
            vuid_suffix: e.vuid_suffix.clone(),
            title: e.title.clone(),
            category: e.category,
            what_happened: e.what_happened.clone(),
            why_rejected: e.why_rejected.clone(),
            ignis_fix: e.ignis_fix.clone(),
            spec_section: e.spec_section.clone(),
        }
    }
}

/// Find a knowledge entry by VUID suffix. Checks runtime entries first.
pub fn lookup(suffix: &str) -> Option<KnowledgeLookup> {
    if let Ok(guard) = runtime_slot().read() {
        for e in guard.iter() {
            if e.vuid_suffix == suffix {
                return Some(KnowledgeLookup::from_runtime(e));
            }
        }
    }
    STATIC_BASE
        .iter()
        .find(|e| e.vuid_suffix == suffix)
        .map(KnowledgeLookup::from_static)
}

/// Number of entries across static and runtime tables.
pub fn total_entries() -> usize {
    let rt = runtime_slot().read().map(|g| g.len()).unwrap_or(0);
    STATIC_BASE.len() + rt
}

/// Access the static portion for documentation or export.
pub fn static_base() -> &'static [KnowledgeEntry] {
    STATIC_BASE
}

// Move the large static slice into this file. In forensic.rs it will be
// removed and lookup() used instead.
static STATIC_BASE: &[KnowledgeEntry] = &[
       KnowledgeEntry {
        vuid_suffix: "00002",
        title: "image usage flag missing TRANSFER_DST",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "a command that writes to the image (clear, copy, blit, resolve) was recorded,\n\
             but the image was not created with VK_IMAGE_USAGE_TRANSFER_DST_BIT in its usage.",
        why_rejected:
            "Vulkan rejects writes to images whose usage flags do not include the\n\
             operation being performed. The driver may have chosen an image layout\n\
             optimized for other uses and cannot serve as a transfer destination.",
        ignis_fix:
            "when creating the image, include TRANSFER_DST in the usage flags:\n\n\
             \x20  ctx.create_image(&ImageInfo::texture_2d(\n\
             \x20      w, h, fmt,\n\
             \x20      ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,\n\
             \x20  ))?;\n\n\
             if this is a swapchain image: set SwapchainConfig::image_usage to include\n\
             TRANSFER_DST before create_swapchain. The default is COLOR_ATTACHMENT only.",
        spec_section: "§12.3 Images / §19.4 Copy Commands",
    },
    KnowledgeEntry {
        vuid_suffix: "01213",
        title: "image layout incompatible with usage flags",
        category: DiagnosticCategory::LayoutTransition,
        what_happened:
            "a pipeline barrier transitions an image into a layout that the image's\n\
             usage flags do not permit.",
        why_rejected:
            "each optimal layout requires specific usage bits to have been set when\n\
             the image was created. The driver cannot conjure a layout for a usage\n\
             that was never declared.\n\n\
             common layout/usage pairings:\n\
             \x20  TRANSFER_DST_OPTIMAL    requires TRANSFER_DST_BIT\n\
             \x20  TRANSFER_SRC_OPTIMAL    requires TRANSFER_SRC_BIT\n\
             \x20  COLOR_ATTACHMENT_OPT    requires COLOR_ATTACHMENT_BIT\n\
             \x20  DEPTH_STENCIL_ATT_OPT   requires DEPTH_STENCIL_ATTACHMENT_BIT\n\
             \x20  SHADER_READ_ONLY_OPT    requires SAMPLED_BIT or INPUT_ATTACHMENT_BIT\n\
             \x20  PRESENT_SRC_KHR         requires COLOR_ATTACHMENT_BIT on swapchain",
        ignis_fix:
            "either include the required usage flag at image creation, or use a\n\
             different layout.\n\n\
             if you use ResourceTracker: the tracker picks layouts from\n\
             ImageUsageContext, so a mismatch here means the image was created\n\
             without enough usage flags for its declared access pattern. Union\n\
             the usages the image will actually see across all frames:\n\n\
             \x20  ImageUsageFlags::COLOR_ATTACHMENT\n\
             \x20    | ImageUsageFlags::TRANSFER_DST\n\
             \x20    | ImageUsageFlags::SAMPLED",
        spec_section: "§7.1.3 Image Memory Barriers",
    },
    KnowledgeEntry {
        vuid_suffix: "00629",
        title: "VkInstance destroyed while child objects still alive",
        category: DiagnosticCategory::ObjectLifetime,
        what_happened:
            "vkDestroyInstance was called but a child object (VkSurfaceKHR,\n\
             VkDebugUtilsMessengerEXT, VkDevice) is still alive.",
        why_rejected:
            "all objects derived from an instance must be destroyed before the\n\
             instance itself, otherwise their backing resources leak or become\n\
             dangling.",
        ignis_fix:
            "ignis in managed mode owns and destroys instance on SharedState drop.\n\
             this error typically means one of:\n\n\
             \x20  1. a VkSurfaceKHR created via raw ash calls outside ignis was\n\
             \x20     never destroyed. ignis Swapchain drop destroys the swapchain\n\
             \x20     but not the surface (the caller owns the surface).\n\n\
             \x20  2. a debug utils messenger was installed manually and never\n\
             \x20     destroyed. ignis's own messenger is destroyed before the\n\
             \x20     instance automatically.\n\n\
             \x20  3. a queue broker or external VkDevice reference kept the\n\
             \x20     device alive past its intended scope.\n\n\
             use LifetimeTracker to catch such leaks at shutdown:\n\n\
             \x20  let tracker = ctx.create_lifetime_tracker();\n\
             \x20  tracker.register(ObjectType::SURFACE_KHR, surface.as_raw(), Some(\"main\"));",
        spec_section: "§4.1 Instances",
    },
    KnowledgeEntry {
        vuid_suffix: "05137",
        title: "VkDevice destroyed while child objects still alive",
        category: DiagnosticCategory::ObjectLifetime,
        what_happened:
            "vkDestroyDevice was called but a child object (VkBuffer, VkImage,\n\
             VkPipeline, VkCommandPool, VkFence, ...) is still alive.",
        why_rejected:
            "same rule as instance: all objects derived from a device must be\n\
             destroyed before the device itself.",
        ignis_fix:
            "resources owned by ignis (Buffer, Image, CommandPool, ShaderModule,\n\
             RenderPassHandle, PipelineLayoutHandle, etc) destroy themselves on\n\
             Drop. common sources of this error:\n\n\
             \x20  1. raw vk::Pipeline handles returned by builders. The builder\n\
             \x20     returns a raw handle and the caller must call\n\
             \x20     ctx.device().destroy_pipeline(handle, None) before ignis\n\
             \x20     drops, OR retire via DeletionQueue with a timeline guard.\n\n\
             \x20  2. raw vk::Fence / vk::Semaphore allocated manually. Use\n\
             \x20     FencePool for fences, and timeline semaphores via\n\
             \x20     AsyncQueue::timeline() when possible.\n\n\
             \x20  3. descriptor set layouts created through the descriptor\n\
             \x20     builder hold raw handles that must be destroyed.\n\n\
             create a LifetimeTracker and register raw handles you allocate:\n\n\
             \x20  tracker.register(ObjectType::PIPELINE, pipe.as_raw(), Some(\"gbuffer\"));\n\n\
             at shutdown, ctx drop triggers the tracker which names every leak.",
        spec_section: "§4.2 Devices",
    },
    KnowledgeEntry {
        vuid_suffix: "00070",
        title: "queue submit with fence already in use",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkQueueSubmit was called with a fence that is still associated\n\
             with a previous pending submission.",
        why_rejected:
            "a fence can be associated with only one pending submission at a time.\n\
             Reusing it without waiting and resetting first creates an ambiguous\n\
             synchronization contract.",
        ignis_fix:
            "fences must be waited and reset before reuse. ignis provides three\n\
             patterns for this:\n\n\
             \x20  1. FrameSync manages per-frame-in-flight fences:\n\
             \x20     let frame = frame_sync.begin_frame()?;\n\
             \x20     // frame.fence() is already waited and reset\n\n\
             \x20  2. FencePool: acquire/release semantics with automatic reset:\n\
             \x20     let fence = pool.acquire()?;\n\
             \x20     // submit, wait\n\
             \x20     pool.release(fence)?;\n\n\
             \x20  3. GpuFuture owns its fence and destroys it on drop:\n\
             \x20     let future = queue.submit_simple(cmd)?;\n\
             \x20     future.wait()?;\n\n\
             if using Vulkan 1.2+, prefer timeline semaphores via\n\
             AsyncQueue::submit().build() which avoids per-submission fences entirely.",
        spec_section: "§7.3 Fences",
    },
    KnowledgeEntry {
        vuid_suffix: "02697",
        title: "draw without bound graphics pipeline",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "vkCmdDraw, vkCmdDrawIndexed, or a related draw command was recorded\n\
             without a graphics pipeline being bound first.",
        why_rejected:
            "draw commands need a graphics pipeline to define vertex input, shader\n\
             stages, rasterization, blending, and render pass compatibility.",
        ignis_fix:
            "bind the pipeline before any draw:\n\n\
             \x20  rec.bind_pipeline(PipelineBindPoint::GRAPHICS, pipeline);\n\
             \x20  rec.draw(vertex_count, instance_count, 0, 0);\n\n\
             if you use ValidatedRecorder, this check runs CPU-side at record\n\
             time with a better error pointing at the missing bind:\n\n\
             \x20  let mut vrec = ValidatedRecorder::wrap(rec);\n\
             \x20  vrec.draw(...);  // fails with [IGN-S002] missing graphics pipeline\n\n\
             also verify the pipeline is compatible with the current render pass\n\
             (or matches your DynamicRenderPassBuilder setup).",
        spec_section: "§22 Drawing Commands",
    },
    KnowledgeEntry {
        vuid_suffix: "06788",
        title: "VkBufferCreateInfo::usage is zero",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened: "a buffer was created with no usage flags set.",
        why_rejected:
            "every buffer must declare at least one usage flag so the driver can\n\
             allocate appropriate memory and configure hardware descriptors.",
        ignis_fix:
            "use one of the BufferInfo constructors which provide sensible defaults:\n\n\
             \x20  BufferInfo::vertex(size, MemoryLocation::GpuOnly)\n\
             \x20  BufferInfo::index(size, MemoryLocation::GpuOnly)\n\
             \x20  BufferInfo::uniform(size)                 // CpuToGpu, UNIFORM_BUFFER\n\
             \x20  BufferInfo::storage(size, location)       // STORAGE_BUFFER\n\
             \x20  BufferInfo::staging(size)                 // CpuToGpu, TRANSFER_SRC\n\n\
             or build BufferInfo manually with non-empty usage flags.",
        spec_section: "§12.1 Buffers",
    },
    KnowledgeEntry {
        vuid_suffix: "04009",
        title: "descriptor set incompatible with pipeline layout",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "a descriptor set bound at a given slot was created from a layout\n\
             that is not compatible with the pipeline layout expecting that slot.",
        why_rejected:
            "the pipeline layout describes exactly what each descriptor set slot\n\
             must contain (count per binding, descriptor types, stage flags). A\n\
             set from a mismatching layout cannot be bound there.",
        ignis_fix:
            "create the descriptor set layout and pipeline layout from the same\n\
             binding description, and use PipelineAuditor to verify compatibility:\n\n\
             \x20  let auditor = PipelineAuditor::new();\n\
             \x20  auditor.register_layout(layout, &[set0_hash, set1_hash], &push_ranges);\n\
             \x20  auditor.register_pipeline(pipe, Some(\"main\"), layout, &shader_hashes);\n\n\
             \x20  // at bind time\n\
             \x20  for issue in auditor.validate_bind(pipe, 2) {\n\
             \x20      eprintln!(\"{}\", auditor.report(&[issue]));\n\
             \x20  }\n\n\
             if the pipeline expects 3 sets and you only bind 2, the auditor\n\
             catches it before the GPU ever sees the draw.",
        spec_section: "§14.2.2 Pipeline Layouts / §14.2.3 Descriptor Set Updates",
    },
    KnowledgeEntry {
        vuid_suffix: "00011",
        title: "memory allocation exceeds heap size or type incompatibility",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkAllocateMemory was called with a size exceeding the heap capacity,\n\
             or with a memory type index incompatible with the resource's\n\
             memoryRequirements.memoryTypeBits.",
        why_rejected:
            "the driver must back every allocation with a concrete heap and type.\n\
             Requests that exceed physical limits or mismatch type constraints\n\
             have no valid backing.",
        ignis_fix:
            "ignis allocators pick memory types via find_memory_type_index which\n\
             respects memoryTypeBits. Seeing this error usually means:\n\n\
             \x20  1. the requested allocation is larger than the DEVICE_LOCAL\n\
             \x20     heap. Monitor with BudgetMonitor:\n\n\
             \x20       let budget = ctx.create_budget_monitor(Default::default());\n\
             \x20       if let Some(report) = budget.check() { ... }\n\n\
             \x20  2. a custom Allocator implementation is picking the wrong\n\
             \x20     type index. Use BlockAllocator or SlabAllocator which\n\
             \x20     share the correct selection logic.\n\n\
             \x20  3. the driver's memory_properties.memoryHeaps was misread.\n\
             \x20     Check ctx.memory_properties() and confirm heap sizes.",
        spec_section: "§11.6 Device Memory Allocation",
    },
    KnowledgeEntry {
        vuid_suffix: "01997",
        title: "feature or extension used but not enabled at device creation",
        category: DiagnosticCategory::FeatureNotEnabled,
        what_happened:
            "an operation requires a device feature or extension that was not\n\
             enabled when vkCreateDevice was called.",
        why_rejected:
            "Vulkan is strict: features are opt-in. Using a feature without\n\
             enabling it is undefined behavior, and the layer blocks it.",
        ignis_fix:
            "common cases and their ignis fix:\n\n\
             \x20  ray tracing:\n\
             \x20    ManagedConfig::new(..).enable_raytracing(true)\n\n\
             \x20  timeline semaphores:\n\
             \x20    requires Vulkan 1.2+, auto-enabled by ignis\n\n\
             \x20  shader printf (VK_KHR_shader_non_semantic_info):\n\
             \x20    ManagedConfig::new(..).enable_shader_printf(true)\n\n\
             \x20  pipeline statistics queries:\n\
             \x20    ManagedConfig::new(..).enable_pipeline_stats(true)\n\n\
             \x20  descriptor indexing for bindless:\n\
             \x20    ManagedConfig::new(..).enable_descriptor_indexing(true)\n\n\
             \x20  custom extensions:\n\
             \x20    .device_extension(MY_EXT_NAME)\n\n\
             if using external device mode, you must enable the feature\n\
             yourself before handing the device to Ignis::external.",
        spec_section: "§41 Features / §35 Extensions",
    },
    KnowledgeEntry {
        vuid_suffix: "01208",
        title: "COLOR_ATTACHMENT layout requires COLOR_ATTACHMENT usage",
        category: DiagnosticCategory::LayoutTransition,
        what_happened:
            "a pipeline barrier transitions an image into VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL\n\
             but the image was not created with VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT.",
        why_rejected:
            "COLOR_ATTACHMENT_OPTIMAL is a layout the driver reserves for images\n\
             that will be used as color render targets. Without COLOR_ATTACHMENT_BIT\n\
             at creation time, the driver has not allocated metadata required to\n\
             back that layout (tile tracking, compression state, etc).",
        ignis_fix:
            "either add COLOR_ATTACHMENT to the image usage flags, or use a layout\n\
             compatible with the usage you actually declared.\n\n\
             for storage images rendered by compute, use GENERAL:\n\n\
             \x20  ImageUsageContext::ComputeShaderWrite  // resolves to GENERAL\n\n\
             for sampling-only images, use SHADER_READ_ONLY_OPTIMAL:\n\n\
             \x20  ImageUsageContext::FragmentShaderRead\n\n\
             if the image really is a render target, include COLOR_ATTACHMENT:\n\n\
             \x20  ctx.create_image(&ImageInfo::texture_2d(\n\
             \x20      w, h, fmt,\n\
             \x20      ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,\n\
             \x20  ))?;",
        spec_section: "§7.1.3 Image Memory Barriers",
    },
    KnowledgeEntry {
        vuid_suffix: "01210",
        title: "DEPTH_STENCIL layout requires DEPTH_STENCIL_ATTACHMENT usage",
        category: DiagnosticCategory::LayoutTransition,
        what_happened:
            "a barrier transitions an image into VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL\n\
             or VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL without the image\n\
             having VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT set at creation time.",
        why_rejected:
            "depth-stencil layouts require the usage bit that enables depth-stencil\n\
             hardware path (hi-Z, stencil compression, fast clears). Without the\n\
             bit, the memory layout is incompatible with those features.",
        ignis_fix:
            "use ImageInfo::depth which sets the required usage automatically:\n\n\
             \x20  ctx.create_image(&ImageInfo::depth(w, h, Format::D32_SFLOAT))?;\n\n\
             or if building ImageInfo manually, include the usage bit:\n\n\
             \x20  usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT\n\
             \x20         | ImageUsageFlags::SAMPLED  // if you also sample the depth",
        spec_section: "§7.1.3 Image Memory Barriers",
    },
    KnowledgeEntry {
        vuid_suffix: "00115",
        title: "copy region size exceeds source buffer bounds",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkCmdCopyBuffer was called with a region whose size + srcOffset\n\
             exceeds the size of the source buffer.",
        why_rejected:
            "the driver cannot read past the end of a buffer. The region\n\
             must satisfy: srcOffset + size <= src_buffer.size.",
        ignis_fix:
            "clamp the copy size against the source buffer:\n\n\
             \x20  let copy_size = desired.min(src.size() - src_offset);\n\
             \x20  rec.copy_buffer(src.handle(), dst.handle(), &[vk::BufferCopy {\n\
             \x20      src_offset, dst_offset, size: copy_size,\n\
             \x20  }]);\n\n\
             if the copy is supposed to cover the whole buffer, use src.size():\n\n\
             \x20  size: src.size()\n\n\
             for staging uploads where source is sized to the data, use the\n\
             same constant in both the staging allocation and the copy region.",
        spec_section: "§19.2 Buffer Copy Commands",
    },
    KnowledgeEntry {
        vuid_suffix: "00116",
        title: "copy region size exceeds destination buffer bounds",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkCmdCopyBuffer was called with a region whose size + dstOffset\n\
             exceeds the size of the destination buffer.",
        why_rejected:
            "the driver cannot write past the end of a buffer. The region\n\
             must satisfy: dstOffset + size <= dst_buffer.size.",
        ignis_fix:
            "clamp the copy size against the destination buffer:\n\n\
             \x20  let copy_size = desired.min(dst.size() - dst_offset);\n\n\
             common pattern: allocating destination with the exact size of the\n\
             data to be copied:\n\n\
             \x20  let dst = ctx.create_buffer(&BufferInfo {\n\
             \x20      size: data.len() as u64,\n\
             \x20      usage: BufferUsageFlags::TRANSFER_DST | ...,\n\
             \x20      location: MemoryLocation::GpuOnly,\n\
             \x20      sharing_mode: SharingMode::EXCLUSIVE,\n\
             \x20  })?;\n\n\
             if both src and dst come from the same upload pipeline, consider\n\
             using StagingRing which sizes them in lockstep automatically.",
        spec_section: "§19.2 Buffer Copy Commands",
    },
    // Image copy / blit / clear 

    KnowledgeEntry {
        vuid_suffix: "00219",
        title: "cannot blit from or to a multisampled image",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "vkCmdBlitImage was called with a source or destination image whose\n\
            sample count is greater than VK_SAMPLE_COUNT_1_BIT.",
        why_rejected:
            "blit is defined as a filtered copy that samples from a single texel\n\
            per source location. Multisampled images hold N samples per texel\n\
            and have no well-defined scalar value at any given coordinate, so\n\
            the operation is prohibited without an explicit resolve first.",
        ignis_fix:
            "resolve the multisampled image into a single-sample image, then\n\
            blit from that.\n\n\
            using a render pass with a resolve attachment is the idiomatic path:\n\n\
            \x20  ctx.render_pass_builder()\n\
            \x20     .attachment(AttachmentConfig {\n\
            \x20         format: Format::R8G8B8A8_UNORM,\n\
            \x20         samples: SampleCountFlags::TYPE_4,\n\
            \x20         store_op: AttachmentStoreOp::DONT_CARE,\n\
            \x20         ..Default::default()\n\
            \x20     })\n\
            \x20     .attachment(AttachmentConfig {  // resolve target\n\
            \x20         format: Format::R8G8B8A8_UNORM,\n\
            \x20         samples: SampleCountFlags::TYPE_1,\n\
            \x20         final_layout: ImageLayout::TRANSFER_SRC_OPTIMAL,\n\
            \x20         ..Default::default()\n\
            \x20     })\n\n\
            outside a render pass, use vkCmdResolveImage directly on the raw\n\
            device handle (ignis does not wrap this yet).",
        spec_section: "§19.5 Image Blit Commands",
    },

    KnowledgeEntry {
        vuid_suffix: "01728",
        title: "image copy aspect mask mismatch",
        category: DiagnosticCategory::LayoutTransition,
        what_happened:
            "vkCmdCopyImage was called with source and destination subresource\n\
            aspects that do not correspond. Most commonly this means copying\n\
            from a depth image to a color image (or vice versa), or mixing\n\
            DEPTH and STENCIL aspects.",
        why_rejected:
            "image copies are per-aspect. The driver needs matching aspect\n\
            masks so it can translate tile layouts between source and\n\
            destination. Copying COLOR into DEPTH is semantically undefined\n\
            because the bit patterns decode differently.",
        ignis_fix:
            "use format_aspect_mask to derive the correct aspect automatically:\n\n\
            \x20  use ignis::format;\n\
            \x20  let aspect = format::format_aspect_mask(image.format());\n\n\
            when both images are the same aspect (typical case), use that\n\
            aspect on both subresource layers. For depth+stencil images,\n\
            copy DEPTH and STENCIL as separate regions:\n\n\
            \x20  for aspect in [ImageAspectFlags::DEPTH, ImageAspectFlags::STENCIL] {\n\
            \x20      rec.copy_image(src, src_layout, dst, dst_layout, &[ImageCopy {\n\
            \x20          src_subresource: ImageSubresourceLayers {\n\
            \x20              aspect_mask: aspect, ...\n\
            \x20          },\n\
            \x20          dst_subresource: ImageSubresourceLayers {\n\
            \x20              aspect_mask: aspect, ...\n\
            \x20          },\n\
            \x20          ...\n\
            \x20      }]);\n\
            \x20  }",
        spec_section: "§19.3 Image Copy Commands",
    },

    KnowledgeEntry {
        vuid_suffix: "00171",
        title: "copy region extends past image boundary",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkCmdCopyBufferToImage or vkCmdCopyImageToBuffer was called with\n\
            an imageOffset + imageExtent that exceeds the image's actual\n\
            dimensions at the given mip level.",
        why_rejected:
            "each mip level has a smaller extent than level 0. Writing past\n\
            the edge of a mip level is undefined, and on tiled architectures\n\
            can corrupt adjacent mip levels stored in the same memory page.",
        ignis_fix:
            "compute the mip extent before copying:\n\n\
            \x20  let mip_w = (image.extent().width  >> mip).max(1);\n\
            \x20  let mip_h = (image.extent().height >> mip).max(1);\n\
            \x20  rec.copy_buffer_to_image(buf, image.handle(), TRANSFER_DST_OPTIMAL,\n\
            \x20      &[BufferImageCopy {\n\
            \x20          image_subresource: ImageSubresourceLayers {\n\
            \x20              aspect_mask: ImageAspectFlags::COLOR,\n\
            \x20              mip_level: mip,\n\
            \x20              base_array_layer: 0,\n\
            \x20              layer_count: 1,\n\
            \x20          },\n\
            \x20          image_offset: Offset3D { x: 0, y: 0, z: 0 },\n\
            \x20          image_extent: Extent3D { width: mip_w, height: mip_h, depth: 1 },\n\
            \x20          ..Default::default()\n\
            \x20      }]);\n\n\
            if uploading a full mip chain, use the mipmap helper which\n\
            handles this automatically:\n\n\
            \x20  ignis::tracking::mipmap::generate_mipmaps(&rec, &mut tracker,\n\
            \x20      image.handle(), image.format(),\n\
            \x20      image.extent().width, image.extent().height,\n\
            \x20      image.mip_levels(), Filter::LINEAR);",
        spec_section: "§19.4 Copying Data Between Buffers and Images",
    },

    KnowledgeEntry {
        vuid_suffix: "01556",
        title: "oldLayout must be UNDEFINED or match current layout",
        category: DiagnosticCategory::LayoutTransition,
        what_happened:
            "an image memory barrier specified oldLayout as something other\n\
            than UNDEFINED or the image's actual current layout.",
        why_rejected:
            "Vulkan tracks image layouts implicitly. If the barrier lies about\n\
            the source layout, the driver generates wrong transitions, which\n\
            can leave the image in a decompressed or corrupt state.",
        ignis_fix:
            "two correct patterns:\n\n\
            \x20  1. oldLayout = UNDEFINED: use when the contents of the image\n\
            \x20     are no longer needed. The driver is free to discard them,\n\
            \x20     which is the fastest path for render targets being reused.\n\n\
            \x20  2. oldLayout = actual previous layout: use when the contents\n\
            \x20     must be preserved across the transition.\n\n\
            tracking which layout each image is in manually is error-prone.\n\
            use ResourceTracker which maintains per-subresource state:\n\n\
            \x20  let mut tracker = ctx.create_resource_tracker();\n\
            \x20  tracker.track_image(img.handle(), ImageLayout::UNDEFINED,\n\
            \x20                      img.mip_levels(), img.array_layers(),\n\
            \x20                      ImageAspectFlags::COLOR);\n\
            \x20  // later: tracker.transition_image(img.handle(), context)\n\
            \x20  //         returns a barrier with correct oldLayout automatically.",
        spec_section: "§7.1.3 Image Memory Barriers",
    },

    KnowledgeEntry {
        vuid_suffix: "00931",
        title: "clear color image requires TRANSFER_DST or STORAGE usage",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "vkCmdClearColorImage was called on an image whose usage flags\n\
            include neither TRANSFER_DST_BIT nor STORAGE_BIT.",
        why_rejected:
            "clear is a write operation. The driver needs either the transfer\n\
            or storage path enabled at image creation to know the image can\n\
            be written from a clear command.",
        ignis_fix:
            "include TRANSFER_DST in the usage flags at creation:\n\n\
            \x20  ImageInfo {\n\
            \x20      usage: ImageUsageFlags::SAMPLED\n\
            \x20           | ImageUsageFlags::TRANSFER_DST,  // for clears/uploads\n\
            \x20      ..Default::default()\n\
            \x20  }\n\n\
            for render targets, COLOR_ATTACHMENT already implies the ability\n\
            to clear at render pass begin via AttachmentLoadOp::CLEAR. Use\n\
            that instead of vkCmdClearColorImage when possible:\n\n\
            \x20  AttachmentConfig {\n\
            \x20      load_op: AttachmentLoadOp::CLEAR,\n\
            \x20      ..Default::default()\n\
            \x20  }",
        spec_section: "§19.6 Clear Commands",
    },

    // Buffer binding and usage 

    KnowledgeEntry {
        vuid_suffix: "00628",
        title: "vertex buffer missing VERTEX_BUFFER usage flag",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "vkCmdBindVertexBuffers was called with a buffer whose usage flags\n\
            do not include VK_BUFFER_USAGE_VERTEX_BUFFER_BIT.",
        why_rejected:
            "the driver exposes vertex fetch through a dedicated hardware path\n\
            that requires specific memory layout and access patterns. Without\n\
            the usage bit at creation, the buffer is not configured for this\n\
            path and vertex fetch would read garbage.",
        ignis_fix:
            "use BufferInfo::vertex which sets the flag for you:\n\n\
            \x20  let vbo = ctx.create_buffer(&BufferInfo::vertex(\n\
            \x20      mesh.byte_size(),\n\
            \x20      MemoryLocation::GpuOnly,\n\
            \x20  ))?;\n\n\
            if combining with other uses (e.g. compute reads the buffer to\n\
            compute normals), union the flags manually:\n\n\
            \x20  BufferInfo {\n\
            \x20      size: mesh.byte_size(),\n\
            \x20      usage: BufferUsageFlags::VERTEX_BUFFER\n\
            \x20           | BufferUsageFlags::STORAGE_BUFFER\n\
            \x20           | BufferUsageFlags::TRANSFER_DST,\n\
            \x20      location: MemoryLocation::GpuOnly,\n\
            \x20      sharing_mode: SharingMode::EXCLUSIVE,\n\
            \x20  }",
        spec_section: "§21.2 Vertex Input Description",
    },

    KnowledgeEntry {
        vuid_suffix: "00433",
        title: "index buffer missing INDEX_BUFFER usage flag",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "vkCmdBindIndexBuffer was called with a buffer whose usage flags\n\
            do not include VK_BUFFER_USAGE_INDEX_BUFFER_BIT.",
        why_rejected:
            "same reasoning as vertex buffers: the driver has a specialized\n\
            index fetch path enabled by the usage bit at creation time.",
        ignis_fix:
            "use BufferInfo::index:\n\n\
            \x20  let ibo = ctx.create_buffer(&BufferInfo::index(\n\
            \x20      indices.len() as u64 * 4,  // UINT32 indices\n\
            \x20      MemoryLocation::GpuOnly,\n\
            \x20  ))?;\n\n\
            when binding, match the IndexType to your actual index width:\n\n\
            \x20  rec.bind_index_buffer(ibo.handle(), 0, IndexType::UINT32);  // u32 indices\n\
            \x20  rec.bind_index_buffer(ibo.handle(), 0, IndexType::UINT16);  // u16 indices",
        spec_section: "§21.3 Index Buffers",
    },

    KnowledgeEntry {
        vuid_suffix: "02708",
        title: "indirect draw buffer missing INDIRECT_BUFFER usage flag",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "vkCmdDrawIndirect, vkCmdDrawIndexedIndirect, vkCmdDispatchIndirect,\n\
            or a related command was called with a buffer whose usage flags\n\
            do not include VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT.",
        why_rejected:
            "indirect commands pull draw parameters from the buffer at the\n\
            front of the command processor, which is a specialized path\n\
            the driver only enables when the usage flag is declared.",
        ignis_fix:
            "include INDIRECT_BUFFER in the buffer usage:\n\n\
            \x20  BufferInfo {\n\
            \x20      size: std::mem::size_of::<DrawIndirectCommand>() as u64 * max_draws,\n\
            \x20      usage: BufferUsageFlags::INDIRECT_BUFFER\n\
            \x20           | BufferUsageFlags::STORAGE_BUFFER  // if compute writes it\n\
            \x20           | BufferUsageFlags::TRANSFER_DST,\n\
            \x20      location: MemoryLocation::GpuOnly,\n\
            \x20      sharing_mode: SharingMode::EXCLUSIVE,\n\
            \x20  }\n\n\
            if a compute shader writes the indirect args, insert a barrier\n\
            between the compute write and the indirect draw:\n\n\
            \x20  let t = tracker.transition_buffer(\n\
            \x20      buf.handle(), BufferUsageContext::IndirectDraw\n\
            \x20  );\n\
            \x20  if let Some(t) = t { rec.apply_buffer_transitions(&[t]); }",
        spec_section: "§21.11 Draw Commands / §27.3 Dispatching Commands",
    },

    KnowledgeEntry {
        vuid_suffix: "00932",
        title: "buffer view requires UNIFORM_TEXEL_BUFFER or STORAGE_TEXEL_BUFFER usage",
        category: DiagnosticCategory::UsageFlagMismatch,
        what_happened:
            "vkCreateBufferView was called on a buffer whose usage flags do\n\
            not include either VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT or\n\
            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT.",
        why_rejected:
            "texel buffer views present a typed view over raw buffer memory.\n\
            The driver needs to allocate descriptor hardware for that view,\n\
            which is only done when the usage flag signals the intent.",
        ignis_fix:
            "add the appropriate texel buffer flag:\n\n\
            \x20  BufferInfo {\n\
            \x20      usage: BufferUsageFlags::UNIFORM_TEXEL_BUFFER    // read-only\n\
            \x20         // or BufferUsageFlags::STORAGE_TEXEL_BUFFER  // read/write\n\
            \x20           | BufferUsageFlags::TRANSFER_DST,\n\
            \x20      ..Default::default()\n\
            \x20  }\n\n\
            if you do not need typed access, use a regular storage buffer\n\
            (STORAGE_BUFFER) which is simpler, supports larger ranges, and\n\
            avoids buffer views entirely.",
        spec_section: "§12.2 Buffer Views",
    },

    KnowledgeEntry {
        vuid_suffix: "00805",
        title: "uniform buffer binding exceeds maxUniformBufferRange",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "a descriptor update bound a uniform buffer with a range larger\n\
            than VkPhysicalDeviceLimits::maxUniformBufferRange (typically\n\
            16 KiB to 64 KiB depending on the device).",
        why_rejected:
            "uniform buffers use a fast constant-register path with limited\n\
            addressable range. Exceeding the limit is a hard hardware cap,\n\
            not a software choice.",
        ignis_fix:
            "check the device limit at startup:\n\n\
            \x20  let max = ctx.device_properties().limits.max_uniform_buffer_range;\n\
            \x20  println!(\"max uniform range: {} bytes\", max);\n\n\
            options when data exceeds the limit:\n\n\
            \x20  1. split into multiple uniform buffer bindings, each within\n\
            \x20     the limit.\n\n\
            \x20  2. switch to a storage buffer (STORAGE_BUFFER), which supports\n\
            \x20     maxStorageBufferRange (commonly 128 MiB or more):\n\n\
            \x20       BufferInfo::storage(size, MemoryLocation::CpuToGpu)\n\n\
            \x20     and change the descriptor type in the shader from\n\
            \x20     `uniform Block { ... }` to `readonly buffer Block { ... }`.\n\n\
            \x20  3. use dynamic uniform buffers with offsets if the data is\n\
            \x20     a ring of small blocks (see FrameAllocator).",
        spec_section: "§14.5.4 Uniform Buffers",
    },

    // Descriptor sets 

    KnowledgeEntry {
        vuid_suffix: "00324",
        title: "descriptor type mismatch in write",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "vkUpdateDescriptorSets wrote a descriptor whose type did not match\n\
            the layout binding's descriptor type. Common mistake: writing\n\
            SAMPLED_IMAGE into a binding declared as COMBINED_IMAGE_SAMPLER.",
        why_rejected:
            "descriptor types have different memory layouts and backing hardware\n\
            resources. A mismatched write produces descriptor memory that\n\
            decodes incorrectly when the shader samples it.",
        ignis_fix:
            "when using DescriptorWriter, pass the exact type declared by the\n\
            layout binding:\n\n\
            \x20  // layout has: binding 0 = COMBINED_IMAGE_SAMPLER\n\
            \x20  DescriptorWriter::new(set)\n\
            \x20      .image(0, DescriptorType::COMBINED_IMAGE_SAMPLER,\n\
            \x20             view, sampler, ImageLayout::SHADER_READ_ONLY_OPTIMAL)\n\
            \x20      .write(ctx.device());\n\n\
            common pairings:\n\n\
            \x20  SAMPLED_IMAGE          -> only image view, sampler = NULL\n\
            \x20  SAMPLER                -> only sampler, image view = NULL\n\
            \x20  COMBINED_IMAGE_SAMPLER -> both image view and sampler\n\
            \x20  STORAGE_IMAGE          -> view in GENERAL layout, sampler = NULL\n\
            \x20  UNIFORM_BUFFER         -> buffer + offset + range (uniform)\n\
            \x20  STORAGE_BUFFER         -> buffer + offset + range (storage)\n\n\
            PipelineAuditor can catch layout vs pipeline mismatches at record\n\
            time before the draw reaches the GPU.",
        spec_section: "§14.2.3 Updates to Descriptor Sets",
    },

    KnowledgeEntry {
        vuid_suffix: "00325",
        title: "descriptor write array index out of range",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "a descriptor write specified a dstArrayElement + descriptorCount\n\
            that exceeds the descriptorCount declared at the binding in the\n\
            set layout.",
        why_rejected:
            "each binding has a fixed array capacity. Writing beyond it would\n\
            overflow into the next binding's descriptor memory.",
        ignis_fix:
            "verify the binding's array size matches your write range.\n\n\
            for bindless designs with large arrays, use BindlessHeap which\n\
            manages slot allocation safely:\n\n\
            \x20  let heap = ctx.create_bindless_heap(BindlessConfig {\n\
            \x20      sampled_images: 16384,\n\
            \x20      ..Default::default()\n\
            \x20  })?;\n\
            \x20  let handle = heap.register_sampled_image(view, layout)?;\n\
            \x20  // handle.raw() is a slot index guaranteed within bounds.\n\n\
            for fixed-size bindings, declare the count at layout creation:\n\n\
            \x20  ctx.descriptor_set_layout_builder()\n\
            \x20     .binding(0, DescriptorType::COMBINED_IMAGE_SAMPLER,\n\
            \x20              16,  // array size\n\
            \x20              ShaderStageFlags::FRAGMENT)\n\
            \x20     .build()?;",
        spec_section: "§14.2.3 Updates to Descriptor Sets",
    },

    KnowledgeEntry {
        vuid_suffix: "00340",
        title: "descriptor buffer offset exceeds buffer size",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "a buffer descriptor was written with VkDescriptorBufferInfo::offset\n\
            equal to or greater than the buffer's size.",
        why_rejected:
            "the offset must point to a valid location inside the buffer from\n\
            which the range begins. An out-of-bounds offset has no valid\n\
            target memory.",
        ignis_fix:
            "when slicing a buffer for multiple descriptor bindings (common\n\
            with FrameAllocator), track offsets explicitly:\n\n\
            \x20  let mut alloc = ctx.create_frame_allocator(\n\
            \x20      1 << 20, 2, BufferUsageFlags::UNIFORM_BUFFER)?;\n\
            \x20  alloc.advance();\n\n\
            \x20  let (camera_off, _) = alloc.push_bytes(256, 256)?;\n\
            \x20  let (material_off, _) = alloc.push_bytes(128, 256)?;\n\n\
            \x20  DescriptorWriter::new(set)\n\
            \x20      .buffer(0, DescriptorType::UNIFORM_BUFFER,\n\
            \x20              alloc.buffer(), camera_off, 256)\n\
            \x20      .buffer(1, DescriptorType::UNIFORM_BUFFER,\n\
            \x20              alloc.buffer(), material_off, 128)\n\
            \x20      .write(ctx.device());\n\n\
            both offsets are guaranteed to be within the buffer because\n\
            FrameAllocator tracks remaining capacity and rejects oversized\n\
            pushes.",
        spec_section: "§14.2.3 Updates to Descriptor Sets",
    },

    KnowledgeEntry {
        vuid_suffix: "00341",
        title: "descriptor buffer range is zero or exceeds buffer bounds",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "a buffer descriptor was written with range = 0, or range + offset\n\
            exceeds the buffer's size.",
        why_rejected:
            "a zero range is meaningless. An out-of-bounds range would let\n\
            the shader read past the buffer, exposing arbitrary memory.",
        ignis_fix:
            "when the binding covers the whole buffer, use VK_WHOLE_SIZE:\n\n\
            \x20  DescriptorWriter::new(set)\n\
            \x20      .buffer(0, DescriptorType::STORAGE_BUFFER,\n\
            \x20              buf.handle(), 0, vk::WHOLE_SIZE)\n\
            \x20      .write(ctx.device());\n\n\
            for typed buffers, use TypedBuffer::byte_size() as the range:\n\n\
            \x20  let tbuf: TypedBuffer<Particle> = ctx.create_typed_buffer(\n\
            \x20      1024, BufferUsageFlags::STORAGE_BUFFER,\n\
            \x20      MemoryLocation::GpuOnly)?;\n\
            \x20  .buffer(0, DescriptorType::STORAGE_BUFFER,\n\
            \x20          tbuf.handle(), 0, tbuf.byte_size())",
        spec_section: "§14.2.3 Updates to Descriptor Sets",
    },

    KnowledgeEntry {
        vuid_suffix: "02824",
        title: "descriptor bound but never written",
        category: DiagnosticCategory::DescriptorMismatch,
        what_happened:
            "a draw or dispatch used a descriptor set binding that was never\n\
            written via vkUpdateDescriptorSets. Accessing an uninitialized\n\
            descriptor is undefined behavior.",
        why_rejected:
            "descriptor memory starts uninitialized. Reading it produces\n\
            arbitrary handle values, which the shader then tries to use as\n\
            textures or buffers.",
        ignis_fix:
            "DescriptorAuditor catches this before the GPU sees the draw:\n\n\
            \x20  let mut auditor = DescriptorAuditor::new();\n\
            \x20  auditor.register_resource(buf.handle().as_raw());\n\
            \x20  auditor.record_write(set, 0, BoundResource::Buffer {\n\
            \x20      handle: buf.handle().as_raw(),\n\
            \x20      offset: 0, range: 256,\n\
            \x20  });\n\
            \x20  // later, before submit:\n\
            \x20  let issues = auditor.validate_set(set);\n\
            \x20  if !issues.is_empty() {\n\
            \x20      eprintln!(\"{}\", auditor.report(&issues));\n\
            \x20  }\n\n\
            alternatively, use partially-bound descriptors via the descriptor\n\
            indexing feature so unbound slots do not cause errors. This\n\
            requires enable_descriptor_indexing(true) in ManagedConfig and\n\
            PARTIALLY_BOUND binding flags on the layout.",
        spec_section: "§14.5.2 Descriptor Set Access",
    },

    // Pipelines 

    KnowledgeEntry {
        vuid_suffix: "02699",
        title: "bound pipeline incompatible with active render pass",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "a draw command used a graphics pipeline whose render pass is not\n\
            compatible with the render pass currently begun on the command\n\
            buffer.",
        why_rejected:
            "pipelines are compiled against a specific render pass layout\n\
            (attachment formats, sample counts, subpass dependencies). Using\n\
            a pipeline inside an incompatible render pass would produce\n\
            hardware state mismatches.",
        ignis_fix:
            "two render passes are \"compatible\" when their attachments have\n\
            matching format and sample count, and the subpass structure\n\
            matches. For many applications this means building a pipeline\n\
            per render pass.\n\n\
            when using dynamic rendering (Vulkan 1.3), the pipeline carries\n\
            the color/depth formats directly, eliminating render pass\n\
            compatibility rules entirely:\n\n\
            \x20  // pipeline is created against formats, not a render pass:\n\
            \x20  PipelineRenderingCreateInfo::default()\n\
            \x20      .color_attachment_formats(&[Format::B8G8R8A8_SRGB])\n\
            \x20      .depth_attachment_format(Format::D32_SFLOAT);\n\n\
            \x20  // recording uses DynamicRenderPassBuilder:\n\
            \x20  DynamicRenderPassBuilder::new()\n\
            \x20      .color_attachment(ColorAttachmentInfo { image_view, ... })\n\
            \x20      .depth_attachment(DepthStencilAttachmentInfo { ... })\n\
            \x20      .begin(&rec);",
        spec_section: "§8.2 Render Pass Compatibility",
    },

    KnowledgeEntry {
        vuid_suffix: "07751",
        title: "dynamic state required by pipeline was not set",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "a graphics pipeline declared a dynamic state (e.g. VIEWPORT,\n\
            SCISSOR, LINE_WIDTH) but the draw command was issued without a\n\
            corresponding vkCmdSet*() call to provide the value.",
        why_rejected:
            "dynamic state is decoupled from the pipeline: the pipeline\n\
            promises the value will be supplied at record time. If it is\n\
            not, the GPU has no value to use.",
        ignis_fix:
            "every dynamic state declared in the pipeline must have a\n\
            matching set_* call before draw:\n\n\
            \x20  rec.bind_pipeline(PipelineBindPoint::GRAPHICS, pipe);\n\
            \x20  rec.set_viewport(0, &[viewport]);\n\
            \x20  rec.set_scissor(0, &[scissor]);\n\
            \x20  rec.draw(3, 1, 0, 0);\n\n\
            ignis's GraphicsPipelineBuilder defaults to VIEWPORT and SCISSOR\n\
            as dynamic states, so always call set_viewport / set_scissor\n\
            before your first draw in each command buffer.\n\n\
            ValidatedRecorder can detect missing dynamic state at record\n\
            time if you extend its state machine.",
        spec_section: "§10.7 Dynamic State",
    },

    KnowledgeEntry {
        vuid_suffix: "00746",
        title: "graphics pipeline created with null render pass",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "vkCreateGraphicsPipelines was called with pCreateInfo->renderPass\n\
            = VK_NULL_HANDLE, but dynamic rendering was not enabled via a\n\
            VkPipelineRenderingCreateInfo chained on pNext.",
        why_rejected:
            "a graphics pipeline must target either a classical render pass\n\
            or a dynamic rendering configuration. Null with neither is an\n\
            incomplete specification.",
        ignis_fix:
            "attach a render pass when creating the pipeline:\n\n\
            \x20  ctx.graphics_pipeline_builder()\n\
            \x20     .shader_stage(ShaderStageFlags::VERTEX,   vs.handle(), \"main\")\n\
            \x20     .shader_stage(ShaderStageFlags::FRAGMENT, fs.handle(), \"main\")\n\
            \x20     .layout(layout.handle())\n\
            \x20     .render_pass(pass.handle(), 0)  // required\n\
            \x20     .build()?;\n\n\
            or, for Vulkan 1.3 dynamic rendering, attach the format info\n\
            via ash directly (ignis's builder does not yet expose this; use\n\
            .push_next on the CreateInfo returned by the builder at a lower\n\
            level).",
        spec_section: "§10.2 Graphics Pipelines",
    },

    // Render passes and framebuffers 

    KnowledgeEntry {
        vuid_suffix: "02352",
        title: "framebuffer attachments incompatible with render pass",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "vkCmdBeginRenderPass was called with a framebuffer whose\n\
            attachment image views do not match the render pass's attachment\n\
            descriptions (format, sample count, or count).",
        why_rejected:
            "the framebuffer binds concrete image views to the abstract\n\
            attachment slots described by the render pass. Mismatches\n\
            would cause the driver to route writes and reads incorrectly.",
        ignis_fix:
            "build the framebuffer with views whose formats match the render\n\
            pass attachments one-to-one in the same order:\n\n\
            \x20  let rp = ctx.render_pass_builder()\n\
            \x20      .attachment(AttachmentConfig {\n\
            \x20          format: swap.format().format,  // swapchain format\n\
            \x20          final_layout: ImageLayout::PRESENT_SRC_KHR,\n\
            \x20          ..Default::default()\n\
            \x20      })\n\
            \x20      .attachment(AttachmentConfig {\n\
            \x20          format: Format::D32_SFLOAT,\n\
            \x20          final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,\n\
            \x20          ..Default::default()\n\
            \x20      })\n\
            \x20      .subpass(...)\n\
            \x20      .build()?;\n\n\
            \x20  // framebuffer must list views in the same order:\n\
            \x20  let attachments = [color_view, depth_view];\n\
            \x20  let fb = FramebufferCreateInfo::default()\n\
            \x20      .render_pass(rp.handle())\n\
            \x20      .attachments(&attachments)\n\
            \x20      .width(w).height(h).layers(1);",
        spec_section: "§8.4 Framebuffer Compatibility",
    },

    KnowledgeEntry {
        vuid_suffix: "06003",
        title: "dynamic rendering started inside an active render pass",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "vkCmdBeginRendering was called while a classical render pass\n\
            was still active, or vice versa.",
        why_rejected:
            "the two rendering models are mutually exclusive within a command\n\
            buffer scope: you must end one before starting the other.",
        ignis_fix:
            "track the current recording state explicitly. ValidatedRecorder\n\
            enforces this automatically:\n\n\
            \x20  let rec = pool.begin_primary(cmd)?;\n\
            \x20  let mut vrec = ValidatedRecorder::wrap(rec)\n\
            \x20      .on_error(StateErrorAction::Panic);\n\n\
            \x20  // begin_render_pass transitions state to InRenderPass\n\
            \x20  vrec.begin_render_pass(rp, fb, area, clears, SubpassContents::INLINE);\n\
            \x20  // calling notify_begin_rendering() here would fire an error.\n\
            \x20  vrec.end_render_pass();\n\
            \x20  // now state is Recording; dynamic rendering is allowed.\n\n\
            if not using ValidatedRecorder, ensure begin_rendering/end_rendering\n\
            pairs only occur outside begin_render_pass/end_render_pass pairs.",
        spec_section: "§8.3 Dynamic Render Pass Instances",
    },

    // Command buffer state 

    KnowledgeEntry {
        vuid_suffix: "00049",
        title: "vkBeginCommandBuffer called on buffer already recording",
        category: DiagnosticCategory::Other,
        what_happened:
            "vkBeginCommandBuffer was called on a command buffer whose state\n\
            is Recording or Executable and whose pool was not created with\n\
            RESET_COMMAND_BUFFER_BIT.",
        why_rejected:
            "once recording begins, the buffer holds a captured state. Starting\n\
            a second recording without reset would mix commands and produce\n\
            an invalid stream.",
        ignis_fix:
            "ignis CommandPool is created with RESET_COMMAND_BUFFER_BIT by\n\
            default, so begin_primary implicitly resets. The error means\n\
            you are calling it through raw ash.\n\n\
            correct pattern with ignis:\n\n\
            \x20  let cmd = pool.allocate_primary()?;\n\
            \x20  let rec = pool.begin_primary(cmd)?;    // resets if needed\n\
            \x20  // ... record ...\n\
            \x20  let cmd = rec.end()?;\n\
            \x20  queue.submit_simple(cmd)?.wait()?;\n\n\
            to reuse a command buffer across frames, pair it with a fence\n\
            so you only re-record after the previous submission has finished:\n\n\
            \x20  let frame = frame_sync.begin_frame()?;\n\
            \x20  // frame.fence() is waited and reset; safe to re-record.\n\
            \x20  let rec = pool.begin_primary(cmd_for_this_slot)?;",
        spec_section: "§6.1 Command Buffer Lifecycle",
    },

    KnowledgeEntry {
        vuid_suffix: "00094",
        title: "secondary command buffer missing inheritance info",
        category: DiagnosticCategory::Other,
        what_happened:
            "a secondary command buffer was recorded inside a render pass but\n\
            its VkCommandBufferBeginInfo did not provide a\n\
            VkCommandBufferInheritanceInfo with matching render pass data.",
        why_rejected:
            "secondary buffers intended for use inside a render pass must\n\
            declare which render pass and subpass they target, so the driver\n\
            can compile their contents against that layout.",
        ignis_fix:
            "ParallelRecorder passes inheritance info automatically:\n\n\
            \x20  let pr = ctx.create_parallel_recorder(QueueType::Graphics, 4)?;\n\n\
            \x20  let inherit = CommandBufferInheritance {\n\
            \x20      render_pass: rp.handle(),\n\
            \x20      subpass: 0,\n\
            \x20      framebuffer: fb,  // can be null; hint only\n\
            \x20  };\n\n\
            \x20  let secondaries = pr.record(&inherit, &[\n\
            \x20      |rec| { rec.bind_pipeline(...); rec.draw(...); },\n\
            \x20      |rec| { rec.bind_pipeline(...); rec.draw(...); },\n\
            \x20  ])?;\n\n\
            \x20  // primary records vkCmdExecuteCommands:\n\
            \x20  rec.execute_commands(&secondaries);",
        spec_section: "§6.3 Secondary Command Buffers",
    },

    KnowledgeEntry {
        vuid_suffix: "00086",
        title: "command buffer reset while still in pending execution",
        category: DiagnosticCategory::Other,
        what_happened:
            "vkResetCommandBuffer, vkBeginCommandBuffer (on a pool allowing\n\
            individual resets), or vkResetCommandPool was called while one\n\
            of the affected command buffers was still executing on the GPU.",
        why_rejected:
            "resetting a buffer whose commands the GPU is still processing\n\
            races with the hardware and produces undefined results.",
        ignis_fix:
            "wait for the fence that tracks the submission before reusing\n\
            the command buffer:\n\n\
            \x20  let future = queue.submit_simple(cmd)?;\n\
            \x20  future.wait()?;                 // blocking\n\
            \x20  // now cmd is safe to reset\n\n\
            for per-frame recycling, FrameSync handles the wait automatically:\n\n\
            \x20  let frame = frame_sync.begin_frame()?;\n\
            \x20  // begin_frame() has already waited for this slot's fence.\n\n\
            for deletion-queue style cleanup where the buffer must be\n\
            destroyed after the GPU finishes, retire it with a timeline\n\
            guard:\n\n\
            \x20  dq.retire_custom(\"CommandBuffer\", cmd,\n\
            \x20      DeletionGuard::Timeline { timeline, value },\n\
            \x20      |dev| { /* buffers are freed via pool reset, not destroy */ });",
        spec_section: "§6.1 Command Buffer Lifecycle",
    },

    // Queue operations 

    KnowledgeEntry {
        vuid_suffix: "00072",
        title: "command buffer submitted to queue from incompatible family",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkQueueSubmit was called on a queue whose family index differs\n\
            from the queue family the command pool was created for.",
        why_rejected:
            "command buffers are compiled for a specific queue family's\n\
            capability set. A graphics command buffer contains opcodes a\n\
            transfer-only queue cannot execute.",
        ignis_fix:
            "ensure the pool and the queue share a family index:\n\n\
            \x20  let gfx = ctx.queue(QueueType::Graphics)?;\n\
            \x20  let pool = ctx.create_command_pool(QueueType::Graphics)?;\n\
            \x20  // pool.family_index() == gfx.family_index()\n\
            \x20  gfx.submit_simple(cmd)?.wait()?;\n\n\
            if your command buffer mixes graphics and compute work, record\n\
            it on a graphics pool (graphics queues implicitly support both).\n\n\
            for explicit queue-family ownership transfers between async\n\
            compute and graphics, use image/buffer memory barriers with\n\
            srcQueueFamilyIndex and dstQueueFamilyIndex set, and submit the\n\
            release/acquire halves on the appropriate queues.",
        spec_section: "§5.1 Queues",
    },

    KnowledgeEntry {
        vuid_suffix: "01292",
        title: "queue does not support presentation to the given surface",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkQueuePresentKHR was called on a queue whose family does not\n\
            report presentation support for the target VkSurfaceKHR (as\n\
            queried via vkGetPhysicalDeviceSurfaceSupportKHR).",
        why_rejected:
            "presentation requires the queue family to be connected to the\n\
            platform's window system. Not all queue families have that path.",
        ignis_fix:
            "most desktop drivers report presentation support on the graphics\n\
            queue family. ignis assumes this when selecting queues.\n\n\
            if you are building an external-mode context or using a custom\n\
            device selector, verify support explicitly:\n\n\
            \x20  let surface_fn = khr::surface::Instance::new(&entry, &instance);\n\
            \x20  let supports = unsafe {\n\
            \x20      surface_fn.get_physical_device_surface_support(\n\
            \x20          physical_device,\n\
            \x20          graphics_family_index,\n\
            \x20          surface,\n\
            \x20      )?\n\
            \x20  };\n\
            \x20  assert!(supports, \"graphics queue cannot present\");\n\n\
            for headless rendering (no presentation), do not call present;\n\
            read back rendered images via ReadbackRequest.",
        spec_section: "§32.5 Presenting Images",
    },

    KnowledgeEntry {
        vuid_suffix: "01432",
        title: "swapchain image acquired while still owned by presentation",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkAcquireNextImageKHR returned an image index for an image that\n\
            the application has already acquired and not yet returned via\n\
            vkQueuePresentKHR. This happens when the application acquires\n\
            more images than swapchain.minImageCount - 1.",
        why_rejected:
            "a swapchain has a finite number of images. Acquiring all of them\n\
            without presenting any leaves the presentation engine with\n\
            nothing to display from.",
        ignis_fix:
            "structure the frame loop so that each acquire is paired with a\n\
            present:\n\n\
            \x20  loop {\n\
            \x20      let frame = frame_sync.begin_frame()?;\n\
            \x20      let (image_idx, _) = swap.acquire_next_image(\n\
            \x20          u64::MAX, frame.image_available_semaphore(),\n\
            \x20          vk::Fence::null())?;\n\n\
            \x20      // ... record and submit, signaling render_finished ...\n\n\
            \x20      swap.present(queue_handle, image_idx,\n\
            \x20          &[frame.render_finished_semaphore()])?;\n\
            \x20      frame_sync.advance();\n\
            \x20  }\n\n\
            create the swapchain with enough images to support your\n\
            frames-in-flight count (typically 3 images for 2 frames in\n\
            flight, giving the presentation engine one to display while you\n\
            work on two).",
        spec_section: "§32.6 WSI Swapchain",
    },

    // Synchronization 

    KnowledgeEntry {
        vuid_suffix: "03238",
        title: "timeline semaphore wait value not increasing monotonically",
        category: DiagnosticCategory::SynchronizationHazard,
        what_happened:
            "vkQueueSubmit was called with a timeline semaphore signal value\n\
            less than or equal to the semaphore's current value, or to a\n\
            previously-signaled value.",
        why_rejected:
            "timeline semaphores expose a monotonically increasing counter.\n\
            Signaling backwards or repeating a value would break the\n\
            happens-before ordering the counter provides.",
        ignis_fix:
            "let QueueTimeline allocate values atomically:\n\n\
            \x20  let timeline = queue.timeline().unwrap();\n\
            \x20  let value = timeline.claim_next_value();\n\
            \x20  // every call returns a fresh, larger value.\n\n\
            or use SubmitBuilder which handles the timeline internally:\n\n\
            \x20  let future = queue.submit()\n\
            \x20      .command_buffer(cmd)\n\
            \x20      .with_timeline_watcher(&watcher)\n\
            \x20      .build()?;\n\
            \x20  // the future encapsulates the signaled value; no manual tracking.",
        spec_section: "§7.4 Timeline Semaphores",
    },

    KnowledgeEntry {
        vuid_suffix: "06461",
        title: "pipeline stage mask not supported by queue family",
        category: DiagnosticCategory::SynchronizationHazard,
        what_happened:
            "a pipeline barrier used a srcStageMask or dstStageMask that\n\
            includes stages the current queue family does not support (e.g.\n\
            VERTEX_SHADER on a compute-only queue).",
        why_rejected:
            "barriers translate to hardware synchronization primitives on the\n\
            specific queue. Referring to stages absent on that queue is\n\
            undefined.",
        ignis_fix:
            "use stages compatible with the queue. For async compute, stick\n\
            to compute and transfer stages:\n\n\
            \x20  PipelineStageFlags::COMPUTE_SHADER\n\
            \x20  PipelineStageFlags::TRANSFER\n\
            \x20  PipelineStageFlags::HOST\n\
            \x20  PipelineStageFlags::TOP_OF_PIPE / BOTTOM_OF_PIPE  // always allowed\n\n\
            ResourceTracker / ImageUsageContext pick stages automatically\n\
            matching the usage:\n\n\
            \x20  // on an async compute queue:\n\
            \x20  tracker.transition_image(img, ImageUsageContext::ComputeShaderWrite);\n\
            \x20  // -> uses COMPUTE_SHADER stage, always valid.\n\n\
            for cross-queue ownership transfers, the release on the source\n\
            queue and the acquire on the destination queue each use stages\n\
            valid for their respective queue.",
        spec_section: "§7.1 Synchronization Primitives",
    },

    // Memory 

    KnowledgeEntry {
        vuid_suffix: "01030",
        title: "buffer memory offset not aligned to requirements",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkBindBufferMemory was called with a memoryOffset that is not a\n\
            multiple of VkMemoryRequirements::alignment for that buffer.",
        why_rejected:
            "hardware memory controllers require specific alignment for\n\
            efficient access. The driver reports the minimum alignment via\n\
            memory requirements; violating it either fails to bind or leads\n\
            to corrupted reads/writes.",
        ignis_fix:
            "BlockAllocator and SlabAllocator handle alignment automatically\n\
            from the VkMemoryRequirements returned by the driver. You hit\n\
            this error only when binding memory manually.\n\n\
            if implementing a custom Allocator, align the offset up:\n\n\
            \x20  fn allocate(&self, req: &MemoryRequirements, ...) -> Result<Allocation> {\n\
            \x20      let aligned = (self.cursor + req.alignment - 1)\n\
            \x20                     & !(req.alignment - 1);\n\
            \x20      // ... bind at `aligned` ...\n\
            \x20  }\n\n\
            the built-in align_up helper is available crate-internally; for\n\
            external allocators, write the same formula inline.",
        spec_section: "§12.4 Resource Memory Association",
    },

    KnowledgeEntry {
        vuid_suffix: "01047",
        title: "image memory binding attempted twice",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkBindImageMemory or vkBindBufferMemory was called on a resource\n\
            that already has memory bound. Rebinding is not allowed unless\n\
            the resource was created with the SPARSE_BINDING flag.",
        why_rejected:
            "a non-sparse resource is permanently associated with the memory\n\
            it was first bound to. Rebinding would leave the original memory\n\
            orphaned and violate the driver's internal tracking.",
        ignis_fix:
            "create a new resource instead of rebinding. ignis Buffer and\n\
            Image handle this correctly on construction; you should not see\n\
            this error unless mixing raw ash calls.\n\n\
            if you need to \"change memory backing\" semantically, destroy\n\
            the old resource and create a new one:\n\n\
            \x20  drop(old_image);            // destroys handle + frees memory\n\
            \x20  let new_image = ctx.create_image(&new_info)?;\n\n\
            if the old image is still in use by an in-flight command buffer,\n\
            retire it via DeletionQueue instead:\n\n\
            \x20  let value = timeline.claim_next_value();\n\
            \x20  // submit final uses, signaling `value`.\n\
            \x20  old_image.retire(&dq, DeletionGuard::Timeline { timeline, value });",
        spec_section: "§12.4 Resource Memory Association",
    },

    KnowledgeEntry {
        vuid_suffix: "00689",
        title: "vkUnmapMemory called on unmapped memory",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkUnmapMemory was called on a VkDeviceMemory whose map count is\n\
            zero (never mapped, or already unmapped).",
        why_rejected:
            "unmap is stateful. Calling it without a matching map produces\n\
            an invalid state transition.",
        ignis_fix:
            "ignis allocators persistently map host-visible memory at\n\
            allocation time and unmap at free time. You should never call\n\
            unmap_memory manually on ignis-allocated buffers.\n\n\
            if interfacing with raw ash and mapped memory, pair each map\n\
            with exactly one unmap, and never unmap memory you did not map:\n\n\
            \x20  let ptr = unsafe { device.map_memory(mem, 0, size, flags)? };\n\
            \x20  // ... use ptr ...\n\
            \x20  unsafe { device.unmap_memory(mem); }\n\n\
            for repeated access, keep the memory mapped persistently and\n\
            only unmap once at destruction. Persistent mapping is cheap on\n\
            all modern drivers.",
        spec_section: "§12.6.4 Host Access to Device Memory",
    },

    KnowledgeEntry {
        vuid_suffix: "01390",
        title: "flush/invalidate range not aligned to nonCoherentAtomSize",
        category: DiagnosticCategory::MemoryBinding,
        what_happened:
            "vkFlushMappedMemoryRanges or vkInvalidateMappedMemoryRanges was\n\
            called with an offset or size that is not a multiple of\n\
            VkPhysicalDeviceLimits::nonCoherentAtomSize (typically 64 bytes).",
        why_rejected:
            "cache flush/invalidate operates on cache lines. Partial lines\n\
            cannot be flushed; the driver requires the whole-line granularity\n\
            the atom size represents.",
        ignis_fix:
            "prefer HOST_COHERENT memory (MemoryLocation::CpuToGpu) which does\n\
            not require explicit flush/invalidate. ignis defaults to coherent\n\
            when available and you should never see this error with standard\n\
            BufferInfo usage.\n\n\
            if you must use non-coherent memory, align ranges:\n\n\
            \x20  let atom = ctx.device_properties().limits.non_coherent_atom_size;\n\
            \x20  let aligned_offset = (offset / atom) * atom;\n\
            \x20  let aligned_size   = ((offset + size + atom - 1) / atom) * atom\n\
            \x20                        - aligned_offset;\n\
            \x20  unsafe {\n\
            \x20      device.flush_mapped_memory_ranges(&[MappedMemoryRange::default()\n\
            \x20          .memory(mem)\n\
            \x20          .offset(aligned_offset)\n\
            \x20          .size(aligned_size)])?;\n\
            \x20  }",
        spec_section: "§12.6.4 Host Access to Device Memory",
    },

    // Swapchain / surface 

    KnowledgeEntry {
        vuid_suffix: "01430",
        title: "present called on image index not currently acquired",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkQueuePresentKHR was called with a pImageIndices entry that\n\
            does not correspond to an image currently held by the application\n\
            (i.e. never acquired, already presented, or from a different\n\
            swapchain).",
        why_rejected:
            "the presentation engine reclaims images on present. Presenting\n\
            an image the engine already owns is double-presentation and\n\
            breaks the acquire/present contract.",
        ignis_fix:
            "pair every acquire with exactly one present:\n\n\
            \x20  let (idx, suboptimal) = swap.acquire_next_image(\n\
            \x20      u64::MAX, sem_acquire, Fence::null())?;\n\n\
            \x20  // ... render using swap.image_views()[idx as usize] ...\n\n\
            \x20  swap.present(queue_handle, idx, &[sem_render_done])?;\n\n\
            on swapchain recreation (window resize, ERROR_OUT_OF_DATE_KHR),\n\
            the old image indices are invalid. Recreate before presenting\n\
            the next frame:\n\n\
            \x20  match result {\n\
            \x20      Err(Error::SwapchainOutOfDate) => {\n\
            \x20          swap.recreate(new_w, new_h)?;\n\
            \x20          continue;  // skip present this frame\n\
            \x20      }\n\
            \x20      Err(e) => return Err(e),\n\
            \x20      Ok(_) => {}\n\
            \x20  }",
        spec_section: "§32.6 WSI Swapchain",
    },

    KnowledgeEntry {
        vuid_suffix: "01780",
        title: "acquire semaphore already signaled or in use",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkAcquireNextImageKHR was called with a semaphore that is already\n\
            signaled, or that is currently being waited on by another\n\
            submission.",
        why_rejected:
            "the semaphore used for acquire must be in an unsignaled state\n\
            with no pending waits, because acquire signals it fresh once\n\
            the image is available.",
        ignis_fix:
            "use one semaphore per frame-in-flight, never reuse across the\n\
            frame:\n\n\
            \x20  let frame = frame_sync.begin_frame()?;\n\
            \x20  let sem = frame.image_available_semaphore();\n\
            \x20  // FrameSync maintains N_frames_in_flight distinct semaphores.\n\n\
            \x20  swap.acquire_next_image(u64::MAX, sem, Fence::null())?;\n\n\
            if rolling your own frame sync, allocate one acquire semaphore\n\
            per swapchain image slot, and ensure the previous frame's wait\n\
            on that semaphore has completed before reusing it (typically by\n\
            waiting its fence).",
        spec_section: "§32.6 WSI Swapchain",
    },

    // Features and extensions 

    KnowledgeEntry {
        vuid_suffix: "03916",
        title: "ray tracing pipeline used but extension not enabled",
        category: DiagnosticCategory::FeatureNotEnabled,
        what_happened:
            "vkCmdTraceRaysKHR was called but the device was not created with\n\
            VK_KHR_ray_tracing_pipeline and its rayTracingPipeline feature\n\
            enabled.",
        why_rejected:
            "ray tracing is an opt-in hardware path that requires the\n\
            extension and feature at device creation.",
        ignis_fix:
            "enable ray tracing in ManagedConfig:\n\n\
            \x20  let ctx = Ignis::managed(\n\
            \x20      ManagedConfig::new(\"app\", API_VERSION_1_2)\n\
            \x20          .enable_raytracing(true)\n\
            \x20  )?;\n\n\
            this automatically enables VK_KHR_ray_tracing_pipeline,\n\
            VK_KHR_acceleration_structure, VK_KHR_deferred_host_operations,\n\
            buffer_device_address, and descriptor_indexing.\n\n\
            verify at runtime before using RT APIs:\n\n\
            \x20  if !ctx.supports_ray_tracing() {\n\
            \x20      return fallback_without_rt();\n\
            \x20  }\n\
            \x20  let builder = ctx.raytracing_pipeline_builder()?;\n\n\
            not every device supports ray tracing even when the extension\n\
            is advertised; always check supports_ray_tracing() at startup.",
        spec_section: "§37 Ray Tracing",
    },

    KnowledgeEntry {
        vuid_suffix: "03478",
        title: "buffer device address used but feature not enabled",
        category: DiagnosticCategory::FeatureNotEnabled,
        what_happened:
            "vkGetBufferDeviceAddress was called (or shader code used buffer\n\
            device addresses) without bufferDeviceAddress being enabled at\n\
            device creation.",
        why_rejected:
            "buffer device addresses expose raw 64-bit pointers to GPU memory,\n\
            which is a feature the device and driver must actively enable.",
        ignis_fix:
            "enable the Vulkan 1.2 core feature explicitly (ignis does not\n\
            currently expose a dedicated flag, so use ManagedConfig device\n\
            extensions plus features):\n\n\
            \x20  let config = ManagedConfig::new(\"app\", API_VERSION_1_2)\n\
            \x20      .enable_raytracing(true);  // BDA comes with RT\n\n\
            or, if you need BDA without RT, use external-mode device creation\n\
            where you control the VkPhysicalDeviceVulkan12Features chain:\n\n\
            \x20  let mut features12 = PhysicalDeviceVulkan12Features::default()\n\
            \x20      .buffer_device_address(true);\n\
            \x20  // ... build device with features12 chained ...\n\
            \x20  let ctx = Ignis::external(ExternalDeviceInfo { ... })?;\n\n\
            buffers backed by BDA need BufferUsageFlags::SHADER_DEVICE_ADDRESS\n\
            set at creation:\n\n\
            \x20  BufferInfo {\n\
            \x20      usage: BufferUsageFlags::STORAGE_BUFFER\n\
            \x20           | BufferUsageFlags::SHADER_DEVICE_ADDRESS,\n\
            \x20      ..Default::default()\n\
            \x20  }\n\
            \x20  // then: buf.device_address() returns the u64 pointer.",
        spec_section: "§39 Buffer Device Address",
    },

    KnowledgeEntry {
        vuid_suffix: "02715",
        title: "dynamic rendering used but feature not enabled",
        category: DiagnosticCategory::FeatureNotEnabled,
        what_happened:
            "vkCmdBeginRendering was called without dynamicRendering being\n\
            enabled, either via Vulkan 1.3 core or the\n\
            VK_KHR_dynamic_rendering extension.",
        why_rejected:
            "dynamic rendering replaces the classical render pass system and\n\
            must be enabled explicitly.",
        ignis_fix:
            "use Vulkan 1.3 in ManagedConfig which enables dynamic rendering\n\
            automatically when available:\n\n\
            \x20  let ctx = Ignis::managed(\n\
            \x20      ManagedConfig::new(\"app\", API_VERSION_1_3)\n\
            \x20  )?;\n\n\
            \x20  if ctx.device_properties().api_version < API_VERSION_1_3 {\n\
            \x20      // fallback to classical render passes\n\
            \x20  }\n\n\
            for Vulkan 1.2 + extension, use external-mode device creation\n\
            with VK_KHR_dynamic_rendering in the enabled extensions list\n\
            and VkPhysicalDeviceDynamicRenderingFeatures in the feature chain.",
        spec_section: "§8.3 Dynamic Render Pass Instances",
    },

    // Shader binding / push constants 

    KnowledgeEntry {
        vuid_suffix: "01687",
        title: "push constant stage flags do not cover access",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "vkCmdPushConstants was called with stageFlags that do not include\n\
            all the stages declared to read the push constant range in the\n\
            pipeline layout.",
        why_rejected:
            "push constant stage flags determine which shader stages see the\n\
            update. A mismatch means some stages read stale values.",
        ignis_fix:
            "match stage flags at push time to the declaration in the layout:\n\n\
            \x20  // layout declares push constants visible to both VS and FS:\n\
            \x20  let layout = ctx.pipeline_layout_builder()\n\
            \x20      .push_constant_range(\n\
            \x20          ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,\n\
            \x20          0, 128)\n\
            \x20      .build()?;\n\n\
            \x20  // push must use the same combined stage mask:\n\
            \x20  rec.push_constants(\n\
            \x20      layout.handle(),\n\
            \x20      ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,\n\
            \x20      0, bytemuck::bytes_of(&uniforms));\n\n\
            PipelineAuditor::validate_push_constants checks this at record\n\
            time and produces a rich diagnostic pointing at the mismatched\n\
            stage sets.",
        spec_section: "§14.6.2 Push Constant Updates",
    },

    KnowledgeEntry {
        vuid_suffix: "01688",
        title: "push constant size exceeds pipeline layout range",
        category: DiagnosticCategory::PipelineMismatch,
        what_happened:
            "vkCmdPushConstants was called with offset + size extending past\n\
            the end of a push constant range declared in the pipeline layout.",
        why_rejected:
            "push constants are fixed-size hardware registers. Writing past\n\
            the end corrupts unrelated state or is clamped silently depending\n\
            on the driver.",
        ignis_fix:
            "check maxPushConstantsSize at startup and size your struct\n\
            accordingly:\n\n\
            \x20  let max = ctx.device_properties().limits.max_push_constants_size;\n\
            \x20  println!(\"max push constant size: {max} bytes\");\n\
            \x20  // commonly 128-256 bytes depending on vendor.\n\n\
            if your per-draw data does not fit in push constants, use a\n\
            dynamic uniform buffer with an offset instead:\n\n\
            \x20  let mut alloc = ctx.create_frame_allocator(\n\
            \x20      1 << 18, 2,\n\
            \x20      BufferUsageFlags::UNIFORM_BUFFER,\n\
            \x20  )?;\n\
            \x20  alloc.advance();\n\
            \x20  let offset = unsafe { alloc.push(&large_struct)? };\n\
            \x20  rec.bind_descriptor_sets(\n\
            \x20      PipelineBindPoint::GRAPHICS, layout, 0,\n\
            \x20      &[set], &[offset as u32]);",
        spec_section: "§14.6.2 Push Constant Updates",
    },
    KnowledgeEntry {
        vuid_suffix: "06457",
        title: "srcStageMask contains stages not supported by the queue family",
        category: DiagnosticCategory::SynchronizationHazard,
        what_happened:
            "a pipeline barrier or event command specified a stage mask that\n\
             includes pipeline stages not supported by the queue the command\n\
             buffer will be submitted to.",
        why_rejected:
            "each queue family supports a specific subset of pipeline stages.\n\
             A transfer-only queue cannot see COLOR_ATTACHMENT_OUTPUT. A compute\n\
             queue cannot see VERTEX_SHADER. The layer rejects stages that will\n\
             never fire on the target queue.",
        ignis_fix:
            "check the queue family your command pool targets:\n\n\
             \x20  let pool = ctx.create_command_pool(QueueType::Graphics)?;\n\n\
             graphics queues see all stages. Compute queues see only\n\
             COMPUTE_SHADER and TRANSFER (plus TOP_OF_PIPE / BOTTOM_OF_PIPE /\n\
             HOST / ALL_COMMANDS). Transfer queues see only TRANSFER.\n\n\
             if you record barriers in ResourceTracker, the tracker emits\n\
             stages derived from ImageUsageContext / BufferUsageContext. Use\n\
             only contexts valid for the queue: do not ask for FragmentShaderRead\n\
             on a compute-only queue.",
        spec_section: "§7.1.2 Pipeline Stages / §5.1 Queues",
    },
    KnowledgeEntry {
        vuid_suffix: "02285",
        title: "src and dst stage masks must have at least one stage set",
        category: DiagnosticCategory::SynchronizationHazard,
        what_happened:
            "a pipeline barrier was issued with an empty src or dst stage mask.",
        why_rejected:
            "barriers with an empty mask have no synchronization effect and are\n\
             almost always a bug. The spec requires at least one stage (typically\n\
             TOP_OF_PIPE or BOTTOM_OF_PIPE if nothing else applies).",
        ignis_fix:
            "never pass PipelineStageFlags::empty() to a barrier. Use explicit\n\
             stages matching the operations you synchronize:\n\n\
             \x20  rec.pipeline_barrier(\n\
             \x20      PipelineStageFlags::TRANSFER,              // src\n\
             \x20      PipelineStageFlags::FRAGMENT_SHADER,       // dst\n\
             \x20      DependencyFlags::empty(),\n\
             \x20      &[], &[], &image_barriers,\n\
             \x20  );\n\n\
             if you do not know the stages, use ResourceTracker which computes\n\
             them from declared access contexts:\n\n\
             \x20  let transition = tracker.transition_image(img, usage);\n\
             \x20  rec.apply_image_transitions(&[transition]);",
        spec_section: "§7.1.2 Pipeline Stages",
    },
    KnowledgeEntry {
        vuid_suffix: "02791",
        title: "image layout at command time does not match barrier expectation",
        category: DiagnosticCategory::LayoutTransition,
        what_happened:
            "an image was used in a layout different from what the most recent\n\
             barrier transitioned it to, or the barrier's oldLayout did not\n\
             match the image's actual current layout.",
        why_rejected:
            "Vulkan tracks image layouts. Each barrier must declare the correct\n\
             source layout; operations must run in a layout compatible with\n\
             their access type.",
        ignis_fix:
            "centralize layout tracking. Do not compute layouts manually across\n\
             multiple code paths. Use ResourceTracker as the single source of\n\
             truth:\n\n\
             \x20  let mut tracker = ResourceTracker::new();\n\
             \x20  tracker.track_image(img.handle(), ImageLayout::UNDEFINED, ...);\n\
             \x20  let t = tracker.transition_image(\n\
             \x20      img.handle(),\n\
             \x20      ImageUsageContext::FragmentShaderRead,\n\
             \x20  );\n\
             \x20  rec.apply_image_transitions(&[t]);  // tracker knows old layout\n\n\
             the tracker also keeps per-mip and per-layer layouts for images\n\
             with mip chains or array layers, avoiding the common class of bugs\n\
             where one mip is in a different layout than the others.",
        spec_section: "§7.1.3 Image Memory Barriers",
    },
    KnowledgeEntry {
        vuid_suffix: "04068",
        title: "queue submit wait semaphore not signalable",
        category: DiagnosticCategory::QueueSubmission,
        what_happened:
            "vkQueueSubmit waits on a semaphore that will never be signaled,\n\
             usually because the code forgot to signal it from a prior submit\n\
             or the expected signaler ran on a different queue.",
        why_rejected:
            "Vulkan semaphores are strict single-signal: they must be signaled\n\
             exactly once before being waited on. Waiting on an unsignalable\n\
             semaphore deadlocks the queue.",
        ignis_fix:
            "audit your submit graph. For every wait, there must be a matching\n\
             signal earlier in the same or a prior submission.\n\n\
             on Vulkan 1.2+ prefer timeline semaphores: they accept arbitrary\n\
             wait/signal values and cannot deadlock the same way:\n\n\
             \x20  let timeline = queue.timeline().unwrap();\n\
             \x20  let v = timeline.claim_next_value();\n\
             \x20  queue.submit()\n\
             \x20      .command_buffer(cmd)\n\
             \x20      .build()?;\n\n\
             AsyncQueue::submit() automatically signals the timeline semaphore\n\
             at the claimed value; no manual wiring required.",
        spec_section: "§6.4 Semaphores",
    },
];