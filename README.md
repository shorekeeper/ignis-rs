# ignis

Vulkan queue orchestration, memory management, and GPU debugging toolkit built on [ash](https://crates.io/crates/ash) 0.38.

ignis is a mid-level layer: thicker than raw `ash` (it owns synchronization policy, allocation strategy, and diagnostic infrastructure), thinner than a renderer (it never decides what to draw, never owns a frame loop, and exposes every raw Vulkan handle it wraps). The crate compiles to a core of roughly zero-cost wrappers; everything with runtime overhead is feature-gated and off by default.

- **Rust**: 1.75 or later (edition 2021)
- **Vulkan**: 1.1 minimum (fence-based completion path); 1.2 recommended (timeline semaphores); 1.3 for dynamic rendering helpers
- **Dependencies**: `ash` only. No `tokio`, no `winit`, no allocator crates, no logging framework.
- **Platforms**: Windows, Linux, macOS (via MoltenVK portability enumeration, handled automatically). The `debug-window` and `live-link` features are Windows-only.

---

## Contents

1. [Design constraints](#design-constraints)
2. [Feature flags](#feature-flags)
3. [Quick start](#quick-start)
4. [Architecture](#architecture)
5. [Device modes](#device-modes)
6. [Queues and GPU futures](#queues-and-gpu-futures)
7. [Frame synchronization](#frame-synchronization)
8. [Memory subsystem](#memory-subsystem)
9. [Resource state tracking](#resource-state-tracking)
10. [Deferred destruction](#deferred-destruction)
11. [Pipelines, render passes, descriptors](#pipelines-render-passes-descriptors)
12. [Frame graph](#frame-graph)
13. [SPIR-V reflection](#spir-v-reflection)
14. [Ray tracing](#ray-tracing)
15. [Swapchain](#swapchain)
16. [Diagnostic system](#diagnostic-system)
17. [Debug toolkit reference](#debug-toolkit-reference)
18. [Validation layer pipeline](#validation-layer-pipeline)
19. [Live link protocol](#live-link-protocol)
20. [Debug window](#debug-window)
21. [Interoperability](#interoperability)
22. [Development shell](#development-shell)
23. [Examples](#examples)
24. [Error model](#error-model)
25. [Thread safety](#thread-safety)

---

## Design constraints

The crate is built around five invariants. Violating any of them in a contribution is grounds for rejection.

1. **No hidden global state affecting Vulkan behavior.** The only process-global state is diagnostic (counters, VL handler registries, the printf registry), and none of it alters what is submitted to the device.
2. **Raw handle escape hatches everywhere.** Every wrapper exposes `handle()` or an equivalent. Any operation the crate does not model can be performed through `Ignis::device()` on the objects the crate created.
3. **RAII with explicit lifetimes.** Resources destroy themselves on drop. Where drop order cannot express correctness (GPU still reading a buffer), the crate provides `DeletionQueue` with timeline guards rather than reference-counted magic.
4. **Diagnostics are structural, not string-grep.** Failures are detected by construction (guard bands, bitmaps, sequence counters, topological analysis), reported with stable machine-searchable codes (`IGN-*`), and formatted through one shared framing layer.
5. **Debug overhead is opt-in per module.** A build without `debug-tools` contains none of the toolkit's code. Within `debug-tools`, each facility costs nothing until instantiated.

Non-goals: windowing, input, scene management, shader compilation (SPIR-V is consumed as `&[u32]`), and any form of implicit per-frame allocation.

## Feature flags

| Feature | Adds | Default |
|---|---|---|
| (none) | Device management, queues, `GpuFuture`, `FrameSync`, command recording, `ParallelRecorder`, `BlockAllocator`/`DedicatedAllocator`, `Buffer`/`Image`, staging/frame/typed/readback helpers, pipeline and render pass builders, pipeline cache, `ShaderModule`, SPIR-V reflection, frame graph, format utilities, `ResourceTrace`, acceleration structure builders, bindless heap | yes |
| `tracking` | `ResourceTracker` (per-subresource layout tracking), `DeletionQueue`, mipmap blit-chain helper, `create_*_with_intent` validation | no |
| `descriptors` | Descriptor set layout/pool builders, `DescriptorWriter`, auto-growing `DescriptorArena`, per-frame `DescriptorRing` | no |
| `slab-allocator` | Production hardened `SlabAllocator` | no |
| `swapchain` | `Swapchain` lifecycle management over an externally created `VkSurfaceKHR` | no |
| `interop` | `QueueBroker`, `InteropSync` for cross-engine queue and semaphore sharing | no |
| `debug-tools` | The full debug toolkit: ~25 modules, VL forensic pipeline, VUID knowledge base, crash reporter, profilers, hang detector, determinism checker, and more (see [reference](#debug-toolkit-reference)) | no |
| `live-link` | Shared-memory IPC producer for out-of-process viewers (`ignis-viz`, the PowerShell `live` workspace). Windows only; no-op elsewhere | no |
| `debug-window` | Self-contained CPU-rasterized diagnostic window. Implies `swapchain` + `debug-tools`. Windows only | no |
| `full` | Everything above | no |

Feature gating is enforced by a CI audit (`wintests/test_audit.ps1`) that rejects any `use crate::` import crossing a feature boundary without a matching `#[cfg]`.

## Quick start

```rust
use ignis::prelude::*;
use ash::vk;

fn main() -> ignis::Result<()> {
    // Managed mode: ignis owns instance, device, and queues.
    let ctx = Ignis::managed(
        ManagedConfig::new("my_app", vk::API_VERSION_1_3)
            .enable_validation(true),
    )?;

    let gfx = ctx.queue(QueueType::Graphics)?;
    let pool = ctx.create_command_pool(QueueType::Graphics)?;

    // A host-visible buffer through the default shared block allocator.
    let staging = ctx.create_buffer(&BufferInfo::staging(4096))?;
    staging.write(0, &[0u8; 4096]);

    // Record, submit, await.
    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;
    rec.fill_buffer(staging.handle(), 0, 4096, 0xDEAD_BEEF);
    let cmd = rec.end()?;

    let future = gfx.submit_simple(cmd)?;
    future.wait()?;   // or `.await` under any async executor
    Ok(())
}
```

## Architecture

Every ignis object holds an `Arc<SharedState>`, the crate's single internal aggregate:

```text
SharedState
  entry, instance, device, physical_device
  device_properties, memory_properties, queue_family_props
  rt_pipeline_fn / accel_struct_fn / rt_properties   (optional, ray tracing)
  supports_timelines                                  (Vulkan >= 1.2)
  debug_messenger                                     (managed + debug-tools)
  is_managed                                          (ownership flag)
```

In managed mode, dropping the last `Arc<SharedState>` performs `vkDeviceWaitIdle`, destroys the device, destroys the debug messenger (before the instance, satisfying VUID-vkDestroyInstance-instance-00629), then destroys the instance. In external mode the drop is a no-op; the caller owns everything.

Consequence: the `Ignis` context itself may be dropped while resources live. A `Buffer` keeps its allocator alive, the allocator keeps `SharedState` alive, and teardown happens when the last resource goes.

## Device modes

### Managed: `Ignis::managed(ManagedConfig)`

Creates instance, selects a physical device (discrete > integrated > virtual > CPU, overridable via `device_selector`), creates the logical device with one queue from the graphics family plus dedicated compute and transfer families when they exist, and loads extension function tables.

`ManagedConfig` builder surface:

| Method | Effect |
|---|---|
| `enable_validation(bool)` | `VK_LAYER_KHRONOS_validation`; with `debug-tools`, also installs the ignis debug messenger |
| `enable_raytracing(bool)` | `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure` + `VK_KHR_deferred_host_operations`, plus `buffer_device_address` and `descriptor_indexing` features. Requires 1.2 |
| `enable_shader_printf(bool)` | `debugPrintfEXT` plumbing: validation feature enable + `VK_KHR_shader_non_semantic_info`. Implies validation |
| `enable_pipeline_stats(bool)` | `pipeline_statistics_query` base feature (required by `PipelineStatsPool`) |
| `enable_descriptor_indexing(bool)` | Full update-after-bind / partially-bound / runtime-array feature set (required by `BindlessHeap`) |
| `enable_device_fault(bool)` | `VK_EXT_device_fault`, `VK_NV_device_diagnostic_checkpoints`, `VK_AMD_buffer_marker`; each requested only if the physical device advertises it |
| `instance_extension` / `device_extension` | Arbitrary additional extensions |
| `device_selector(Fn(&[PhysicalDeviceInfo]) -> usize)` | Custom device choice |

Timeline semaphores are enabled unconditionally when the requested API version is >= 1.2. macOS portability enumeration and `VK_KHR_portability_subset` are appended automatically on that platform.

### External: `Ignis::external(ExternalDeviceInfo)`

Wraps caller-owned `ash::Instance`, `ash::Device`, a physical device, and a list of `QueueAllocation` entries. ignis loads its own `ash::Entry` for extension queries but destroys nothing.

Safety contract: all provided handles must be valid and must outlive every ignis object; if `enable_raytracing` is set, the device must have been created with the RT extension chain.

The `DeviceHandle` trait is the inverse direction: implement it on your engine's device type to feed ignis utilities without constructing an `Ignis` at all.

## Queues and GPU futures

`AsyncQueue` wraps a `VkQueue` behind a `Mutex` (Vulkan queues require external synchronization) plus, on 1.2+, a per-queue monotonic timeline semaphore (`QueueTimeline`).

### Submission

```rust
let future = queue.submit()
    .command_buffer(cmd)
    .wait_semaphore(sem, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
    .signal_semaphore(other)
    .with_timeline_watcher(&watcher)   // optional: efficient async wake
    .build()?;
```

`SubmitBuilder::build` selects one of two completion mechanisms:

| | Timeline mode (Vulkan >= 1.2) | Fence mode (fallback) |
|---|---|---|
| Mechanism | Queue timeline signals `claim_next_value()` | Fresh `VkFence` per submit |
| `GpuFuture::drop`, incomplete | Free. No blocking, nothing to destroy | **Blocks** until signaled, then destroys the fence |
| Async `poll` without watcher | Busy-wake (self-wake per poll) | Busy-wake |
| Async `poll` with watcher | `TimelineWatcher`: one thread, one `vkWaitSemaphores(ANY)` across all queues, O(queues + completed) per wake | `FenceWatcher`: polling thread, O(pending) per interval |

`GpuFuture` additionally exposes `is_complete()`, `wait()`, `wait_timeout(Duration)`, and raw accessors (`timeline_semaphore`/`timeline_value`/`fence`) for wiring into foreign synchronization schemes.

Complexity note: `TimelineWatcher::register` is O(log n) (`BTreeMap` insert keyed by target value); the wake path drains all entries `<= current` per semaphore via `split_off`.

### Multi-threaded recording

`ParallelRecorder` owns one transient `VkCommandPool` per worker. `record(&inheritance, &tasks)` resets all pools, then records one secondary command buffer per task under `std::thread::scope`, propagating the first error or `Error::ThreadPanic`. Returned secondaries feed `CommandRecorder::execute_commands`.

## Frame synchronization

`FrameSync::new(shared, frames_in_flight)` allocates per-slot: a fence (created signaled), an image-available semaphore, and a render-finished semaphore.

```text
loop:
    frame = sync.begin_frame()?        // vkWaitForFences + vkResetFences on slot fence
    acquire image with frame.image_available_semaphore()
    record; submit with fence = frame.fence()
    present
    sync.advance()
```

**Present-wait caveat (VUID-vkQueueSubmit-pSignalSemaphores-00067).** `FrameContext::render_finished_semaphore()` is per frame slot and is valid only for intra-frame signal chains. The presentation engine tracks binary-semaphore occupancy per swapchain image; whenever image count differs from frames-in-flight, reusing a slot-indexed semaphore as the present wait eventually signals a semaphore still owned by an outstanding present. Signal and wait on `Swapchain::render_complete_semaphore(image_index)` instead, which is allocated per swapchain image and rebuilt on `recreate`. The VUID knowledge base entry `00067` documents the full pattern.

`FencePool` provides acquire/release fence recycling (reset on release) for code that manages submission lifetimes manually.

## Memory subsystem

### The `Allocator` trait

```rust
pub trait Allocator: Send + Sync {
    fn allocate(&self, req: &vk::MemoryRequirements, loc: MemoryLocation) -> Result<Allocation>;
    fn free(&self, allocation: &Allocation);
    fn name(&self) -> &str;
}
```

`Allocation` is plain data (`memory`, `offset`, `size`, optional persistently mapped pointer, `memory_type_index`); it does not free on drop. `MemoryLocation` is `GpuOnly`, `CpuToGpu` (host-visible coherent, prefers device-local for ReBAR), or `GpuToCpu`. Memory type selection is two-pass: required+preferred flags first, required-only fallback.

### Implementations

| Allocator | Strategy | Intended use |
|---|---|---|
| `BlockAllocator` (default) | 256 MiB blocks, suballocation with next-fit hint and free-list coalescing. One mutex per Vulkan memory type (max 32), so threads on different types never contend. `free` is O(1) pool lookup via `memory_type_index`. Oversized requests get a dedicated block | General purpose |
| `DedicatedAllocator` | One `VkDeviceMemory` per allocation | Few large resources; beware `maxMemoryAllocationCount` (~4096 on most drivers) |
| `SlabAllocator` (`slab-allocator`) | Size classes 256 B .. 1 MiB (powers of two), 2 MiB slabs, CPU-side bitmaps. Structural hardening at near-zero cost: O(1) double-free detection (bitmap), buffer-overflow detection via right-alignment and zero-prefix verification, SplitMix64-randomized slot placement, FIFO quarantine delaying reuse, zero-on-free. `SlabConfig::production()` / `::debug()` presets. Faster than `BlockAllocator` in mixed small-allocation benchmarks (bitmap scan beats free-list traversal; no coalescing on free) | Shipping builds |
| `HardenedAllocator` (`debug-tools`) | Decorator over any allocator: canary guard bands both sides of every allocation (per-allocation canary derived from a secret and the allocation identity), verification on free, byte-budgeted quarantine with re-verification on eviction, zero/junk fill patterns, optional paranoid full-sweep on every free, leak report on drop. Corruption events carry a hex diff window, pattern analysis (memset, MSVC heap patterns, ASCII, float-like, pointer-like, Shannon entropy), and a configurable action (log / panic / callback) | Development and testing |
| `AllocationProfiler` (`debug-tools`) | Decorator: attributes every allocation to its call site via runtime backtrace parsing (Rust hash suffixes stripped, internal frames filtered), aggregates `SiteStats` (total/active/peak counts and bytes), retains live-allocation records for the SVG `MemoryVisualizer`, and optionally mirrors events into `ResourceTrace` and the live link | Memory forensics |

Decorators stack: `AllocationProfiler -> HardenedAllocator -> BlockAllocator` is a valid chain.

### Resources and helpers

`Buffer` and `Image` bind a Vulkan handle to an `Allocation` and free both on drop. Host-visible buffers expose the persistent mapping (`mapped_ptr`, `mapped_slice`, `write`, `write_struct`). `Buffer::device_address()` requires `SHADER_DEVICE_ADDRESS` usage. With `tracking`, both types offer `retire(dq, guard)` (deferred destruction) and `into_raw()`.

Convenience layers, all allocator-parameterized with a lazily created shared default:

| Type | Purpose |
|---|---|
| `StagingRing` | Per-frame ring of `TRANSFER_SRC` buffers; `push(&[u8]) -> StagingRegion` bump-allocates (16-byte aligned) for copy commands |
| `FrameAllocator` | Per-frame bump allocator for transient uniforms/vertex data; `push_bytes` / unsafe typed `push` |
| `TypedBuffer<T>` | Element-indexed access with bounds assertions |
| `ReadbackRequest` | Bundles staging allocation, `record(&rec)` copy, and post-wait `data()` access for GPU-to-CPU readback |

## Resource state tracking

*(feature `tracking`)*

`ResourceTracker` maintains per-subresource image state (every mip and layer independently, which mipmap generation requires) and per-buffer access state. Instead of guessing pipeline stages from layouts, the caller declares intent through enums that resolve unambiguously:

```rust
ImageUsageContext::ComputeShaderRead.resolve()
    // (SHADER_READ_ONLY_OPTIMAL, SHADER_READ, COMPUTE_SHADER)  <- not FRAGMENT_SHADER
```

`transition_image` / `transition_mip` / `transition_buffer` return `Option<Transition>` (None when already in state); `CommandRecorder::apply_image_transitions` / `apply_buffer_transitions` batch them into single `vkCmdPipelineBarrier` calls with union stage masks.

Each context also reports its `required_usage()` flags, consumed by `Ignis::create_image_with_intent(&info, &[contexts])` and `create_buffer_with_intent`, which reject a creation whose usage flags cannot legally serve the declared access patterns, before the driver ever sees the object. This converts an entire class of runtime layout-transition VUIDs into creation-time `Error::InvalidConfig` with the missing flags named.

`tracking::mipmap::generate_mipmaps` records the standard blit chain using per-mip transitions from the tracker.

## Deferred destruction

*(feature `tracking`)*

`DeletionQueue` retires resources under a `DeletionGuard`:

```rust
pub enum DeletionGuard {
    Timeline { timeline: Arc<QueueTimeline>, value: u64 },  // safe when counter >= value
    Fence(vk::Fence),                                       // safe when signaled (not owned)
    Immediate,
}
```

`poll()` destroys every entry whose guard is satisfied (timeline counter reads are cached per semaphore per poll); `flush()` blocks on every guard. Drop flushes with an `IGN-Q001` warning listing the count. `retire_*` methods cover buffers, images (with optional allocator+allocation to free), views, pipelines, layouts, shader modules, render passes, framebuffers, samplers, descriptor pools/set layouts, fences, semaphores, and `retire_custom` for arbitrary destructors (acceleration structures, extension objects).

There is deliberately no frame concept: the same queue serves multiple windows, async compute, and transfer queues simultaneously.

## Pipelines, render passes, descriptors

Builders follow one pattern: accumulate, `build()`, all intermediate Vulkan structs kept alive on the stack across the create call.

- `GraphicsPipelineBuilder`: shader stages with per-stage specialization constants, vertex bindings/attributes, fixed-function state, dynamic states (viewport+scissor by default), layout, render pass, cache. Returns a raw `vk::Pipeline` (caller destroys or retires).
- `ComputePipelineBuilder`, `RayTracingPipelineBuilder` (see [Ray tracing](#ray-tracing)).
- `PipelineLayoutBuilder` -> RAII `PipelineLayoutHandle`.
- `RenderPassBuilder`: attachments, subpasses (color/depth/input/preserve), dependencies -> RAII `RenderPassHandle`.
- `PipelineCache`: disk load (`from_file`, tolerant of missing/invalid data), `save`, `merge`.
- `DynamicRenderPassBuilder` (Vulkan 1.3): `vkCmdBeginRendering` with color/depth/stencil attachments and MSAA resolve, paired with `CommandRecorder::end_rendering`.

With `descriptors`: layout and pool builders, `DescriptorWriter` (owns the `DescriptorBufferInfo`/`DescriptorImageInfo` lifetimes that make raw `vkUpdateDescriptorSets` miserable), `DescriptorArena` (auto-grows on `ERROR_OUT_OF_POOL_MEMORY`/`ERROR_FRAGMENTED_POOL`), `DescriptorRing` (one arena per frame in flight, oldest reset on `advance`).

`BindlessHeap` (core, requires descriptor-indexing features at device creation): one update-after-bind, partially-bound set with four fixed bindings (sampled images, storage images, samplers, storage buffers). Handles are generation-counted; `update_*` and `free_*` on a stale handle return `BindlessError::StaleHandle { slot, expected_generation, current_generation }` instead of corrupting a live slot. Shaders index with `nonuniformEXT(handle.raw())`.

## Frame graph

Declarative pass ordering with automatic barriers (barriers require `tracking`; without it the graph still orders passes).

```rust
let mut fg = FrameGraph::new();
let gbuf  = fg.declare_image("gbuffer", ImageDesc::color(1920, 1080, format));
let depth = fg.declare_image("depth",   ImageDesc::depth(1920, 1080, vk::Format::D32_SFLOAT));

fg.add_pass("geometry", |p| {
    p.writes_image(gbuf,  ImageUsageContext::ColorAttachment);
    p.writes_image(depth, ImageUsageContext::DepthStencilAttachment);
    p.execute(Box::new(|rec, res| { /* draws; res.image(gbuf) resolves the handle */ }));
});
fg.add_pass("lighting", |p| {
    p.reads_image(gbuf,  ImageUsageContext::FragmentShaderRead);
    p.reads_image(depth, ImageUsageContext::FragmentShaderRead);
    p.execute(Box::new(|rec, res| { /* ... */ }));
});

let compiled = fg.compile(&ctx)?;      // allocates transients, toposorts
compiled.execute(&ctx, &pool, queue)?; // or compiled.record(&rec) for frame integration
```

Semantics: readers depend on every writer of a resource; multiple writers of the same resource serialize in registration order; read-write self-reference does not self-loop. Ordering is Kahn's algorithm; a cycle yields `Error::InvalidConfig`. Resources are transient (graph-allocated) or imported (caller-owned, with declared initial layout). `dump_plan()` prints the schedule with access annotations.

Not implemented: transient memory aliasing, multi-queue scheduling.

## SPIR-V reflection

`shader_reflection::reflect(&[u32]) -> Result<ShaderReflection>` is a dependency-free linear parser over the SPIR-V word stream.

Extracted: entry points (all graphics stages, compute, six RT stages), descriptor bindings with set/binding/type/array count (fixed arrays resolved through `OpConstant`; runtime arrays report count 0), push constant ranges (offset = min member offset, size to the end of the highest member), vertex input locations with formats inferred up to 32-bit vec4, specialization constants with byte sizes, compute `LocalSize`.

Derived helpers: `descriptor_set_layouts_from` (per-set sorted `VkDescriptorSetLayoutBinding` arrays) and `push_constant_ranges_from`. `Ignis::create_shader_module_with_reflection` fuses module creation and reflection.

Limits: packed vertex formats (`R10G10B10A2` and similar) are not recoverable from SPIR-V; malformed-but-not-truncated modules produce best-effort partial output rather than errors; truncation, bad magic, and zero-length instructions yield `Error::InvalidSpirv`.

## Ray tracing

Enabled by `ManagedConfig::enable_raytracing` (or an appropriately built external device).

- `RayTracingPipelineBuilder`: stages, `ShaderGroup::{General, TrianglesHit, ProceduralHit}`, recursion depth. Returns RAII `RayTracingPipeline`.
- `RayTracingPipeline::sbt_layout(raygen, miss, hit, callable)`: fetches group handles and computes a complete `ShaderBindingTableLayout` (per-region offsets/sizes/strides honoring `shader_group_handle_alignment` and `shader_group_base_alignment`), with `raygen_region(base_addr)` etc. producing ready `VkStridedDeviceAddressRegionKHR` values.
- `BlasBuilder` / `TlasBuilder`: triangle and AABB geometry, instance encoding in the spec-mandated little-endian 64-byte wire format, scratch buffers aligned to `min_acceleration_structure_scratch_offset_alignment` (a silent-corruption source when ignored). Synchronous `build` and asynchronous `build_async` (returns `(AccelerationStructure, GpuFuture)`; scratch and instance buffers are parked inside the structure until the future resolves). BLAS compaction (`compact(true)`, synchronous only) typically shrinks structures 30-50%. `with_allocator` shares one allocator across many builds; without it each builder creates a fresh `BlockAllocator`, which exhausts `maxMemoryAllocationCount` on large scenes.
- `CommandRecorder::trace_rays` dispatches against the loaded extension table.

## Swapchain

*(feature `swapchain`)*

The caller creates the `VkSurfaceKHR` (winit, SDL, raw platform calls) and retains ownership; `Swapchain` never destroys it.

`Swapchain::new(shared, surface, &SwapchainConfig, w, h)` negotiates format (preferred with fallback to first supported), present mode (fallback FIFO), image count (clamped to capabilities), and builds image views plus **one render-complete semaphore per swapchain image** (see the present-wait caveat above). `acquire_next_image` maps `ERROR_OUT_OF_DATE_KHR` to `Error::SwapchainOutOfDate` and `ERROR_SURFACE_LOST_KHR` to `Error::SurfaceLost`; `recreate(w, h)` waits idle, rebuilds the chain via `old_swapchain`, and reallocates views and semaphores (image count may change).

## Diagnostic system

All diagnostic output flows through one formatter (`src/diagnostic.rs`): severity-colored framed blocks with timestamp, thread, PID, uptime, optional GPU environment block, filtered backtrace, Vulkan spec section reference, hex dumps with diff markers, corruption pattern analysis, progress bars. `NO_COLOR` is honored. Global atomic counters feed a session summary emitted on `Ignis` drop (severity totals, ratio bar, per-code frequency table).

Stable code namespaces:

| Prefix | Subsystem |
|---|---|
| `IGN-H001..H006` | Hardened allocator: front/back guard corruption, invalid free, quarantine re-verification, leaks, report |
| `IGN-S001/S002` | Command state machine: invalid command for state, missing pipeline binding |
| `IGN-S010..S012` | Slab allocator: double free, zero-prefix overflow, statistics |
| `IGN-T001` | Command pool cross-thread access |
| `IGN-A001` | Read-after-write without barrier |
| `IGN-O001` | Barrier over-broadness / redundancy suggestions |
| `IGN-D001` | Stale descriptor references |
| `IGN-P001` | Pipeline layout incompatibility |
| `IGN-W001` | GPU hang (watchdog) |
| `IGN-M001` | Memory budget threshold |
| `IGN-L001` | Leaked Vulkan objects |
| `IGN-J001/J002` | Submission journal dump (with/without error) |
| `IGN-Q001` | Deletion queue flushed at shutdown |
| `IGN-V001/V002` | Validation layer messages (generic / forensic) |
| `IGN-XQ001` | Cross-queue analysis |
| `IGN-DET` | Determinism divergence |
| `IGN-VLB` | VL baseline diff |
| `IGN-PROF` | Allocation profiler report |
| `IGN-SUM` | Session summary |

## Debug toolkit reference

*(feature `debug-tools`; every facility is inert until constructed)*

| Module | Detects / provides | Notes |
|---|---|---|
| `hardened` | Overflow, underflow, UAF, double free, info leaks | See [Memory](#memory-subsystem) |
| `alloc_profiler` + `memory_viz` | Per-call-site allocation attribution; standalone SVG memory layout with hover tooltips | |
| `lifetime` | Leaked objects with `#[track_caller]` creation sites, usage counts, never-used flagging | Leak action: log/panic/callback/ignore |
| `cmd_state` (`ValidatedRecorder`) | Draw outside render pass, dispatch/transfer inside one, unbound pipeline, begin/end pairing; renders the state machine and the last 32 commands on violation | CPU-side, zero GPU cost; invalid calls are skipped, not forwarded |
| `thread_audit` (`AuditedPool`) | Command pool concurrent-thread use, with spec citation and remediation ranking | `release_ownership()` for intentional transfer |
| `hang_detector` + `BreadcrumbBuffer` | Fences unsignaled past a timeout; breadcrumb trail (fill-buffer markers) pinpoints the hung operation with a progress bar and ranked probable causes | |
| `journal` (`SubmissionJournal`) | Flight recorder ring of submissions (queues, semaphores, fences, status); `dump_with_error` on device lost | |
| `aliasing` | Read/write conflicts on a resource without an intervening barrier, rendered as an execution timeline | |
| `barrier_opt` | `ALL_COMMANDS` stages, `MEMORY_READ|WRITE` masks, identical consecutive barriers; suggests minimal replacements in a before/after table | |
| `budget` | `VK_EXT_memory_budget` polling against configurable thresholds (0.80/0.90/0.95), heap-size fallback | |
| `descriptor_audit` | Descriptor sets referencing destroyed buffers/views/samplers | `audit_all()` for crash-time snapshots |
| `pipeline_audit` | Bound-set-count and push-constant-range mismatches against registered layouts | |
| `debug_utils` | Object naming and command labels (`VK_EXT_debug_utils`) for RenderDoc/validation output | |
| `profiler` (`GpuProfiler`) | Timestamp query scopes with millisecond readback | |
| `pipeline_stats` | `PIPELINE_STATISTICS` queries: named counters per scope (vertex/fragment/compute invocations, clipping, IA) | Requires `enable_pipeline_stats` |
| `shader_printf` | `debugPrintfEXT` message parsing (stable across SDK id renames) routed to a process-wide handler | Requires `enable_shader_printf` |
| `crash_report` (`CrashReporter`) | On `trigger(DEVICE_LOST)`: one Markdown file bundling environment, journal, breadcrumb trails, descriptor audit, device fault data, custom sections; atomic write with temp-directory fallback | |
| `device_fault` | `VK_EXT_device_fault` (vendor fault info, address faults, binary blob), NV checkpoints readback, AMD buffer markers with per-stage fired status | Each extension independent; absent ones no-op |
| `determinism` (`DeterminismChecker`) | Runs a recording closure N times, xxh64-hashes declared buffer/image captures, panics on divergence with a BMP diff bitmap (differing pixels red over dimmed baseline) | Closure must be `Fn + Send + Sync + 'static` capturing raw handles |
| `cross_queue` (`CrossQueueTracker`) | Semaphore graph analysis: cycles (guaranteed deadlock), orphan signals/waits, cross-queue edges, longest dependency chain | Import from `SubmissionJournal` or record directly |
| `sync_dag_viz` | Renders the above as DOT, Mermaid, standalone SVG, BMP (single or combined scrollable), or a self-contained HTML index with sticky navigation | |
| `vl_baseline` | Deterministic VUID snapshot (TSV: vuid/severity/category/function/count) with diff: new VUIDs and count increases are regressions, removals and decreases improvements; `Ignis::dump_vl_baseline` / `diff_vl_baseline` for CI gates | Fed before user filtering; suppression does not hide entries |
| `shader_watcher` | Polling file watcher for SPIR-V hot reload; `bytes_to_spirv` helper | |

## Validation layer pipeline

The debug messenger parses every VUID-tagged layer message into a structured `ValidationDiagnostic`: VUID, function, parameter path, involved objects (resolved to debug names and creation sites through a pluggable `ObjectResolver`), Vulkan enum values, category, and a thread-local submit backtrace captured by `SubmitBacktraceGuard` at `vkQueueSubmit` time (the layer callback fires with no stack linkage otherwise).

Diagnostics then traverse a five-stage pipeline, default-configured to be behaviorally identical to plain stderr output:

```text
scope severity overrides -> global overrides -> suppression -> dedup
  -> capture short-circuit -> sinks -> legacy handler -> breakpoint -> action
```

Configuration is declarative:

```rust
ignis_vl! {
    suppress "UNASSIGNED-BestPractices-*";
    escalate "VUID-*-00067" => error;
    action error => panic;
    dedup per_vuid 10;
    sink file "validation.log";
}

let captured = vl_capture! { /* code under test */ };
assert!(captured.errors().is_empty());

vl_expect! {
    rules: { never errors; exactly 1 vuid "VUID-*-01047"; }
    in: { bind_buffer_twice(); }
}

vuid! {   // register an application-specific entry in the knowledge base
    code: "MY-APP-A001", title: "bindless heap near exhaustion",
    severity: Warning, category: Other,
    what_happened: "...", why_rejected: "...", ignis_fix: "...",
    spec_section: "N/A",
}
```

The VUID knowledge base (`debug::vuid_kb`) ships ~60 curated entries, each with plain-language cause, spec rationale, an ignis-specific fix with code, and a spec section reference; forensic output embeds the matching entry automatically. Runtime entries extend it without a rebuild.

## Live link protocol

*(feature `live-link`; Windows)*

`LiveLink::create(name, capacity)` publishes a single-producer shared-memory ring under `Local\<name>` for out-of-process viewers (the [ignis-viz](../ignis-viz) GUI, or the PowerShell `live` terminal workspace in this repository).

Wire contract, little-endian, all offsets in bytes:

```text
Header (64 B):
  0  u64  magic = 0x49474E5356495A30
  8  u32  version = 1
 12  u32  writer_pid
 16  u32  capacity            power of two
 20  u32  record_size = 256
 24  u64  write_idx           monotonic; published after record bytes
 32  u64  read_idx            unused by ignis consumers
 40  u64  last_heartbeat_ns   UNIX epoch ns, refreshed via heartbeat()
 48  16B  reserved

Record (256 B), slot = HEADER + (idx & (capacity-1)) * 256:
  0  u64  timestamp_ns
  8  u64  thread_id
 16  u32  kind
 20  u32  seq                 write_idx & 0xFFFFFFFF at publish (torn-slot guard)
 24  232B payload             #[repr(C)] per kind
```

Event kinds: node/edge register, remove, toggle (1, 2, 8, 9); submission (3); pass (4); allocation/free (5, 6); resource name (10); validation (11); GPU timestamp (12); pipeline stats (13); budget (14); sync cycle/orphan mark with TTL (15); canary corruption with hex windows (16); hardened stats snapshot (17); determinism run/divergence (18, 19); text continuation for oversized strings (20); shader printf (21); hang + breadcrumbs (22, 23); device fault (24); object registered/destroyed (25, 26); descriptor issue (27); aliasing conflict (28); pipeline issue (29); allocation-site snapshot batch keyed by epoch (30).

Consumers must be lossy: when the backlog exceeds their decode budget they drop the oldest records and jump the cursor, and they must validate `seq` against the expected logical index to discard slots overwritten mid-copy. Producer death is observed as a stalled heartbeat, not a fault (a mapped view keeps pages resident).

Bridges wire crate subsystems into the ring: `bridge_validation_to_live_link`, `bridge_cross_queue_to_live_link` (spawned analysis thread with TTL-refreshed lane marks), `bridge_shader_printf_to_live_link`, `bridge_alloc_profiler_to_live_link` (epoch-batched top-N site snapshots); others document canonical inline wiring at the subsystem's construction site.

## Debug window

*(feature `debug-window`; Windows)*

A dependency-free diagnostic window on its own thread: raw `user32` window, `VkSurfaceKHR`, swapchain, and a pipeline-free render path (CPU BGRA framebuffer -> staging buffer -> `vkCmdCopyBufferToImage` -> present), with an embedded 8x8 font. Panels: live memory layout (per-`VkDeviceMemory` bars from an `AllocationProfiler`) and a resource timeline (lanes per event kind from a `ResourceTrace`, allocation-free iteration). Requires surface/swapchain extensions on the context; their absence yields a descriptive `Error::FeatureNotEnabled` rather than a dispatch-table panic. Drop the handle to close; the system close button flips `is_closed()`.

## Interoperability

*(feature `interop`)*

- `QueueBroker`: mutex-mediated exclusive access to one `VkQueue` shared between ignis and a foreign engine (wgpu, vulkano). `acquire()` returns a lock-holding `QueueGuard`; `try_acquire()` is non-blocking. Prefer distinct queue indices when the family provides them; the broker is for the single-queue case.
- `InteropSync`: an owned semaphore pair (`a_done`, `b_done`) implementing the standard cross-engine handoff.

Combined with external device mode and `DeviceHandle`, ignis embeds into an existing engine without owning anything.

## Development shell

The repository ships a PowerShell 7 development environment (`shell.ps1` + `wincommands/` + `wintests/` + `ci.ps1`). It is Windows tooling for working on the crate, not part of the crate.

```text
.\shell.ps1            interactive REPL (Tab completion, persistent history,
                       Ctrl+P fuzzy palette, per-command trace capture)
.\shell.ps1 test all   single-command mode
.\ci.ps1               full CI: feature matrix, lint, unit, import audit,
                       doc coverage, GPU smoke, binary size, miri
```

Selected commands: `build` / `check` / `test` / `lint` / `run` with live progress bars parsed from cargo output; `trace` (structured failure analysis with rustc error explanations and root-cause heuristics); `watch` (debounced rerun on source change with error deltas); `vuid` (offline browser over the crate's VUID knowledge base, parsed from source); `stub` (LLM-oriented API digest: bodies stripped, signatures and docs kept); `mux` (mouse-driven terminal multiplexer: split/zoom/drag/click, persisted layouts, themes); `live <name>` (a terminal consumer of the live link ring: event feed with category filters, memory bars, GPU scope bars, deduplicated validation list with clickable rows opening a knowledge-base-cross-referenced detail overlay, sync marks, hardened stats); `theme`, `gpu`, `crash`, `chrome`.

## Examples

| Example | Requires | Demonstrates |
|---|---|---|
| `smoke_test` | GPU, `full` | End-to-end pass over the core surface; CI gate |
| `smoke_test_advanced` | GPU, `full` | Debug toolkit surface |
| `animated_window`, `window_frame_graph` | GPU, `swapchain` | Present loop; frame graph driving a window |
| `trigger_validation` | validation layer | Forensic VL output on deliberate violations |
| `vl_pipeline_showcase` | `debug-tools` | `ignis_vl!` / capture / expectations |
| `alloc_profiler` | `debug-tools` | Site attribution + SVG visualization |
| `shader_reflection_demo` | none | Reflection over embedded SPIR-V |
| `device_fault_demo`, `vl_baseline_demo`, `determinism_demo`, `cross_queue_demo`, `sync_dag_viz_demo` | `debug-tools` | Respective modules |
| `debug_window_demo` | `debug-window` | Live diagnostic window |
| `live_link_demo` | `live-link` | Full-surface synthetic producer for ignis-viz / `live` (ring name `ignis_demo`) |

```text
cargo run --example smoke_test --features full
cargo run --release --example live_link_demo --features live-link
```

## Error model

One enum, `ignis::Error`: `Vulkan(vk::Result)`, `LoadFailed`, `NoSuitableDevice`, `NoSuitableQueueFamily`, `FeatureNotEnabled(&str)`, `InvalidConfig(&str)`, `ThreadPanic`, `InvalidSpirv`, `Timeout`, `NoSuitableMemoryType`, `SwapchainOutOfDate`, `SurfaceLost`, and `Context(inner, String)` produced by the `WithContext` extension trait (`.context("uploading atlas")`). `Result<T>` aliases `std::result::Result<T, Error>`.

Panics are reserved for caller contract violations (out-of-bounds `TypedBuffer` writes, writes to non-host-visible memory) and for debug facilities explicitly configured with a panic action.

## Thread safety

| Type | Guarantee |
|---|---|
| `Ignis`, `SharedState` | `Send + Sync` (statically asserted) |
| `AsyncQueue` | `Send + Sync`; submissions serialize on an internal mutex |
| `CommandPool` | Single-thread use per Vulkan spec; wrap in `AuditedPool` to detect violations, or use `ParallelRecorder` for one pool per thread |
| `BlockAllocator`, `SlabAllocator`, `HardenedAllocator`, `AllocationProfiler` | `Send + Sync` |
| `GpuFuture` | `Send + Sync`; safe to await on any executor (timeline mode) |
| `FrameSync`, `FencePool`, `DeletionQueue`, `BindlessHeap`, watchers, journal, trackers | Internally synchronized |
| `ResourceTracker`, `FrameGraph`, builders | Not synchronized; single-owner by design |

Background threads (`FenceWatcher`, `TimelineWatcher`, `HangDetector`, `ShaderWatcher`, live-link bridges) are named `ignis-*` and join on drop.