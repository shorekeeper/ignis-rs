# Ignis

Vulkan queue orchestration crate built on top of `ash`. No other dependencies.

Ignis sits between your application and the Vulkan driver, handling the stuff that every Vulkan project ends up reimplementing: async queue submission, per-frame synchronization, multi-threaded command recording, pipeline builders, memory allocation, and a pile of debugging tools that save hours of staring at validation layer output.

It does not try to be a rendering engine or a full abstraction like wgpu/vulkano. You still write Vulkan code. Ignis just makes the plumbing less painful and the bugs easier to find.

## What it does

**Queue orchestration.** Wrap any number of Vulkan queues in thread-safe `AsyncQueue` objects. Submit work through a builder, get back a `GpuFuture` that implements `std::future::Future`. Poll it, await it, or just block. On Vulkan 1.2+ devices, futures use timeline semaphores for O(1) kernel-side completion tracking instead of per-fence polling. On 1.1, a `FenceWatcher` background thread handles it with sleep-based monitoring.

**Per-frame sync.** `FrameSync` manages N fences and semaphores for the classic double/triple buffering loop. `begin_frame()` waits on the right fence, `advance()` moves to the next slot. Nothing fancy, just correct.

**Multi-threaded recording.** `ParallelRecorder` creates one command pool per thread and records secondary command buffers in parallel via `std::thread::scope`. No external thread pool needed.

**Pipeline builders.** Graphics, compute, and ray tracing pipelines through builder patterns with specialization constant support. `PipelineLayoutBuilder` provides RAII layout construction. `PipelineCache` handles disk save/load/merge for faster startup on subsequent runs. The ray tracing builder handles shader groups, SBT layout computation, and `VkStridedDeviceAddressRegionKHR` generation.

**Memory allocation.** Three built-in allocators behind an `Allocator` trait. `BlockAllocator` suballocates from 256 MiB blocks with free-list coalescing and per-memory-type lock sharding, staying well under the ~4096 `VkDeviceMemory` driver limit. `DedicatedAllocator` does one VkDeviceMemory per resource for when you have a handful of large buffers and don't need suballocation. `SlabAllocator` is a production-grade hardened allocator with size-class slabs, bitmap-based double-free detection, randomized slot placement, right-alignment for overflow detection, quarantine for use-after-free mitigation, and zero-on-free. Near-zero overhead compared to BlockAllocator (actually faster in benchmarks due to cache-friendly bitmap scans).

`Buffer` and `Image` are RAII wrappers that take an `Arc<dyn Allocator>`, so you can swap allocators without touching resource code. `TypedBuffer<T>` adds element-level access with bounds checking on top. For a production project you'd probably implement the trait as a thin wrapper around gpu-allocator or vk-mem and get the best of both worlds.

**Staging and transient data.** `StagingRing` provides a per-frame ring buffer for CPU->GPU uploads. `FrameAllocator` is a bump allocator for transient per-frame data (uniforms, dynamic vertices). `ReadbackRequest` bundles GPU->CPU copy, staging buffer allocation, and submission into one call. `FencePool` recycles fences to avoid per-submission create/destroy overhead.

**Swapchain.** Wraps `VK_KHR_swapchain` with acquire/present/recreate. You create the surface externally (winit, SDL, whatever), Ignis manages the rest. Surface ownership stays with you.

**Resource tracker.** Tracks per-subresource image layouts (individual mip levels and array layers can be in different states) and buffer barriers. Uses explicit `ImageUsageContext` / `BufferUsageContext` enums to determine pipeline stages unambiguously instead of guessing from layouts. `TransferDst` vs `ComputeShaderRead` vs `FragmentShaderRead` all produce different, correct barriers. Comes with a mipmap generation utility that uses the tracker for automatic barrier computation across the blit chain.

**Deferred deletion.** `DeletionQueue` tags resources with a timeline semaphore value from the queue they were last used on. Resources are destroyed only after `vkGetSemaphoreCounterValue` confirms the GPU moved past that point. No frame concept, works with multiple windows, async compute, and independent transfer queues. Covers buffers, images, views, pipelines, layouts, samplers, descriptor pools, and custom destructors.

**Descriptors.** `DescriptorArena` auto-grows when a pool runs out of space. `DescriptorRing` maintains one arena per frame-in-flight and resets the oldest each frame, which is the standard pattern for transient per-frame descriptors. `DescriptorWriter` handles the `VkWriteDescriptorSet` lifetime juggling.

**Dynamic rendering.** `DynamicRenderPassBuilder` for Vulkan 1.3+. Traditional `RenderPassBuilder` for 1.2 and earlier. Both coexist.

## Debugging tools

This is where Ignis gets opinionated. Every tool below is opt-in, zero-cost when not used, and produces structured diagnostics with ANSI-colored framed output. Each diagnostic includes a unique code (like `IGN-H001`), a Vulkan spec section reference, GPU/driver/API version context, and a filtered backtrace pointing to the offending call site. Repeated diagnostics are annotated with their occurrence count, and at shutdown a session summary shows total errors/warnings/infos with a per-code breakdown.

**Hardened allocator.** `HardenedAllocator` wraps any allocator with guard bands (canary patterns before and after every allocation), canary verification on free, quarantine queue (delays address reuse), zero-on-free or junk-fill, and per-allocation statistics. When corruption is detected, the diagnostic includes a hex diff, corruption pattern analysis (detects memset fills, ASCII text, float data, pointer values), a layout diagram showing which guard was hit, and a targeted suggestion based on where in the guard the corruption occurred.

```
 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
 ▓▓                          IGNIS DIAGNOSTIC ERROR                          ▓▓
 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 error[IGN-H001]: front guard band corruption
 at 14:23:45.123 | thread="main" | pid=12345 | uptime=3.21s
 spec: Vulkan §11.6 Resource Memory Association
   |  -- Backtrace --
   |    0: ignis::debug::hardened::HardenedAllocator::free
   |    1: ignis::memory::resources::Buffer::drop
   |    2: my_app::renderer::cleanup
   --> VkDeviceMemory(0xec0000ec) offset=448 size=128B
   |  [== front 64B ==][-------- user 128B --------][== back 64B ==]
   |                  ^-- byte 63
   |  -- Corruption Analysis --
   |  pattern: possible float data (first 4 bytes = 0.120682)
   |  extent: 1/64 front guard bytes corrupted (1.6%)
   = help: byte 63/64 of front guard (boundary with user data)
           typically indicates buffer underflow: write before offset 0
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Slab allocator.** `SlabAllocator` builds security into the allocation strategy itself instead of layering it on top. Size-class slabs with bitmap tracking, right-alignment so overflows land in detectable zero-prefixes, randomized slot placement, and quarantine bitmaps. Suitable for shipping builds with near-zero overhead. Debug config adds per-slot event history with caller hashes.

**Object lifetime tracker.** Registers Vulkan objects with `#[track_caller]` location capture. On shutdown (or on demand), reports leaked objects with creation site, age, usage count, and flags never-used objects as likely orphans.

**Command buffer state validator.** `ValidatedRecorder` wraps `CommandRecorder` and validates the recording state machine. Draw outside a render pass? Dispatch inside one? Caught before the call reaches the driver, with a state machine diagram and the full recording history showing how you got there. Invalid calls are skipped entirely so the driver doesn't crash.

**GPU hang detector.** Background watchdog thread monitors fences. If one doesn't signal within a timeout, dumps a diagnostic with timing breakdown and ranked probable causes. Pair it with `BreadcrumbBuffer`, a small CPU-visible GPU buffer that records sequential markers via `vkCmdFillBuffer`. After a hang, you see a progress bar of completed vs pending operations and know exactly which draw or dispatch was the last to finish.

**Submission journal.** Lock-free ring buffer logging every queue submission with timestamps, semaphores, and fence handles. When you get `VK_ERROR_DEVICE_LOST`, dump the journal and see the full chronological sequence of everything in flight, with status markers for completed/pending/error entries.

**Thread safety auditor.** `AuditedPool` wraps `CommandPool` and remembers which thread owns it. Access from a different thread produces an immediate diagnostic with a Vulkan spec quote and remediation options, instead of a sporadic crash 10000 frames later.

**Resource aliasing detector.** Tracks read/write accesses within a recording session. Detects write-then-read without a barrier and write-write conflicts. The diagnostic includes a visual execution timeline showing exactly which operations conflict and where the barrier should go.

**Memory budget monitor.** Polls `VK_EXT_memory_budget` and warns when heap usage approaches configurable thresholds (80%, 90%, 95%). Shows per-heap utilization bars.

**Descriptor set validator.** Tracks resource liveness and descriptor writes. Catches descriptors that still reference destroyed buffers or images before they cause a GPU crash.

**Pipeline compatibility checker.** Validates that bound descriptor set counts match the pipeline layout, and that push constant ranges cover what you're pushing.

**Barrier optimizer.** Records pipeline barriers and flags suboptimal patterns: `ALL_COMMANDS` stages (serializes the GPU), `MEMORY_READ|MEMORY_WRITE` access masks (too broad), redundant consecutive barriers. Shows a before/after comparison table with the suggested tighter masks.

**GPU profiler.** `GpuProfiler` manages a `VkQueryPool` of timestamp queries. Insert named scopes into command buffers, read back elapsed times in milliseconds after execution. `DebugUtils` wraps `VK_EXT_debug_utils` for naming objects (visible in RenderDoc) and inserting command buffer labels.

**Session summary.** At shutdown, Ignis prints a summary of every diagnostic emitted during the session with a severity breakdown and per-code frequency table so noisy codes are easy to spot:

```
 error[IGN-SUM]: diagnostic session summary: 16 total emission(s)
   |    8 error(s)  |  4 warning(s)  |  4 info(s)
   |    [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
   |  -- Breakdown by Diagnostic Code --
   |     ERR   IGN-S001  x2
   |    WARN   IGN-L001  x2
   |    INFO   IGN-J002  x2
   |     ERR   IGN-T001  x1
   |     ...
```

## Device modes

**Managed** -> Ignis creates the Vulkan instance, picks a physical device, creates the logical device and queues. You own nothing, Ignis cleans up on drop.

```rust
let ignis = Ignis::managed(
    ManagedConfig::new("my_app", vk::API_VERSION_1_3)
        .enable_validation(true)
        .enable_raytracing(true),
)?;
```

**External** -> you already have a device (from wgpu, vulkano, egui, whatever). Hand Ignis the handles and it wraps them without taking ownership. Nothing gets destroyed when Ignis drops.

```rust
let ignis = Ignis::external(ExternalDeviceInfo {
    instance: my_instance.clone(),
    device: my_device.clone(),
    physical_device: my_physical,
    queue_allocations: vec![...],
    enable_raytracing: false,
})?;
```

Both modes expose the same API. The `DeviceHandle` trait lets you pass Ignis (or your own engine struct) to utility functions generically.

## Quick example

```rust
use ignis::*;
use ash::vk;

let ignis = Ignis::managed(ManagedConfig::new("demo", vk::API_VERSION_1_2))?;
let alloc = ignis.create_block_allocator();
let queue = ignis.queue(QueueType::Graphics)?;
let pool = ignis.create_command_pool(QueueType::Graphics)?;
let sync = ignis.create_frame_sync(2)?;

// Per-frame loop
let frame = sync.begin_frame()?;
let staging = ignis.create_buffer_with(&alloc, &BufferInfo::staging(1024))?;
staging.write(0, &my_data);

let cmd = pool.allocate_primary()?;
let rec = pool.begin_primary(cmd)?;
rec.copy_buffer(staging.handle(), gpu_buffer.handle(), &[...]);
let cmd = rec.end()?;

queue.submit()
    .command_buffer(cmd)
    .wait_semaphore(frame.image_available_semaphore(), vk::PipelineStageFlags::TRANSFER)
    .signal_semaphore(frame.render_finished_semaphore())
    .build()?
    .wait()?;

sync.advance();
```

## Features

Everything is behind feature flags so you only compile what you use:

- `tracking` -> `ResourceTracker` (per-subresource barriers), `DeletionQueue`
- `descriptors` -> descriptor set/pool builders, `DescriptorArena`, `DescriptorRing`
- `slab-allocator` -> production hardened `SlabAllocator`
- `swapchain` -> swapchain and surface management
- `interop` -> `QueueBroker`, `InteropSync` for cross-engine sharing
- `debug-tools` -> all 14 diagnostic modules listed above
- `full` -> all of the above

Default is no features. Core functionality (queues, sync, command pools, pipeline builders, block/dedicated allocators, buffers, images) is always available.

## Development vs production

The debugging tools are designed for development builds. In production, swap one line:

```rust
// Development: guard bands, canary checks, quarantine, corruption callbacks
let alloc = ignis.create_hardened_allocator(HardenedConfig::default());

// Production: structural hardening, near-zero overhead
let alloc = ignis.create_slab_allocator();

// Minimum viable: plain suballocation, no hardening
let alloc = ignis.create_block_allocator();
```

Same `Arc<dyn Allocator>`, same API. The other tools (lifetime tracker, state validator, hang detector, etc.) simply aren't constructed in production code, so they don't exist and cost nothing.

## Ray tracing

First-class support for `VK_KHR_ray_tracing_pipeline`:

```rust
let ignis = Ignis::managed(
    ManagedConfig::new("rt_app", vk::API_VERSION_1_2)
        .enable_raytracing(true),
)?;

let pipeline = ignis.raytracing_pipeline_builder()?
    .stage(vk::ShaderStageFlags::RAYGEN_KHR, raygen_module, "main")
    .stage(vk::ShaderStageFlags::MISS_KHR, miss_module, "main")
    .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, hit_module, "main")
    .group(ShaderGroup::General { shader_index: 0 })
    .group(ShaderGroup::General { shader_index: 1 })
    .group(ShaderGroup::TrianglesHit { closest_hit: 2, any_hit: None })
    .max_recursion_depth(2)
    .layout(layout)
    .build()?;

let sbt = pipeline.sbt_layout(1, 1, 1, 0)?;
// sbt.raygen_region(buffer_address), sbt.miss_region(...), etc.
```

The `RayTracingPipeline` computes SBT region offsets, alignments, and strides. You allocate the GPU buffer and copy the handle data. Ignis doesn't assume which allocator you're using for the SBT.

## Interop

Ignis plays nice with other crates. It only depends on `ash` and doesn't touch global state. If you're using wgpu or vulkano for rendering and just want Ignis for queue orchestration or debugging, get the raw Vulkan handles from your engine, create Ignis in external mode, and use whichever modules you need. Both libraries operate on separate Vulkan objects and don't interfere.

The `QueueBroker` provides mutex-guarded access when two engines must share the same `VkQueue`. `InteropSync` creates semaphore pairs for cross-engine work handoff. The `Allocator` trait can wrap gpu-allocator or vk-mem if you prefer those for memory management but still want the hardened allocator layer on top.

## Platform support

Tested on Windows and Linux. macOS support is included (portability subset extensions are auto-enabled) but not regularly tested.

## Building

```sh
cargo build --features full
cargo run --example smoke_test --features full
```

The smoke test exercises every subsystem headlessly (no window needed) across 41 test steps. It requires a Vulkan driver. Integrated GPUs work fine.