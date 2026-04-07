# Ignis

Vulkan queue orchestration crate built on top of `ash`. No other dependencies.

Ignis sits between your application and the Vulkan driver, handling the stuff
that every Vulkan project ends up reimplementing: async queue submission,
per-frame synchronization, multi-threaded command recording, pipeline builders,
memory allocation, and a bunch of debugging tools that save hours of staring at
validation layer output.

It does not try to be a rendering engine or a full abstraction like wgpu/vulkano.
You still write Vulkan code. Ignis just makes the plumbing less painful and the
bugs easier to find.

## What it does

**Queue orchestration.** Wrap any number of Vulkan queues in thread-safe
`AsyncQueue` objects. Submit work through a builder, get back a `GpuFuture`
that implements `std::future::Future`. Poll it, await it, or just block - your
choice. Optional `FenceWatcher` background thread replaces busy-wait polling
with sleep-based monitoring.

**Per-frame sync.** `FrameSync` manages N fences and semaphores for the
classic double/triple buffering loop. `begin_frame()` waits on the right fence,
`advance()` moves to the next slot. Nothing fancy, just correct.

**Multi-threaded recording.** `ParallelRecorder` creates one command pool per
thread and records secondary command buffers in parallel via `std::thread::scope`.
No external thread pool needed.

**Pipeline builders.** Graphics, compute, and ray tracing pipelines through
builder patterns. The ray tracing builder handles shader groups, SBT layout
computation, and `VkStridedDeviceAddressRegionKHR` generation.

**Memory allocation.** Two built-in allocators behind an `Allocator` trait:
- `BlockAllocator` - suballocates from 256 MiB blocks, free-list with
  coalescing, stays well under the ~4096 `VkDeviceMemory` driver limit.
- `DedicatedAllocator` - one allocation per resource, for when you have a
  handful of large buffers and don't want the complexity.

`Buffer` and `Image` are RAII wrappers that take an `Arc<dyn Allocator>`, so
you can swap allocators without changing resource code. For a production project
you'd probably implement the trait as a thin wrapper around gpu-allocator or
vk-mem and get the best of both worlds.

**Swapchain.** Wraps `VK_KHR_swapchain` with acquire/present/recreate. You
create the surface externally (winit, SDL, whatever), Ignis manages the rest.
Surface ownership stays with you.

**Resource tracker.** Tracks image layouts and computes pipeline barriers
automatically. Infers access masks and pipeline stages from layouts so you
don't have to look up the table every time. Completely opt-in - if you prefer
manual barriers, the tracker doesn't inject anything.

**Dynamic rendering.** `DynamicRenderPassBuilder` for Vulkan 1.3+. Traditional
`RenderPassBuilder` for 1.2 and earlier. Both coexist.

## Debugging tools

This is where Ignis gets opinionated. Every tool below is opt-in, zero-cost
when not used, and produces structured diagnostics that look like a mix of
rustc error output and Vulkan validation layer messages.

**Hardened allocator.** `HardenedAllocator` wraps any allocator with:
- Guard bands (canary patterns before and after every allocation)
- Canary verification on free (detects buffer overflow/underflow)
- Quarantine queue (delays address reuse, catches use-after-free)
- Zero-on-free or junk-fill (prevents information leaks)
- Per-allocation statistics and corruption callbacks

When it detects corruption, you get output like this:
```
error[IGN-H001]: front guard band corruption
  --> VkDeviceMemory(0xec0000ec) offset=448 size=128B
   |
   |  [== front 64B ==][-------- user 128B --------][== back 64B ==]
   |                  ^-- byte 63
   |
   |  guard hex at +0x0038:
   |   expect: d8 08 96 f8 12 12 4d 0a
   |   actual: d8 08 96 f8 12 12 4d ff
   |                               ^^
   |
   = note: 1/64 front guard bytes corrupted (1.6%)
   = note: detected during Allocator::free()
   = help: byte 63/64 of front guard (boundary with user data)
           typically indicates buffer underflow: write before offset 0
```

**Object lifetime tracker.** Registers every Vulkan object with
`#[track_caller]` location capture. On shutdown (or on demand), reports all
leaked objects with their creation site, age, and usage count.

**Command buffer state validator.** `ValidatedRecorder` wraps `CommandRecorder`
and validates the recording state machine. Draw outside a render pass?
Dispatch inside one? Caught before the call reaches the driver, with a state
trace showing how you got there. Invalid calls are skipped entirely so the
driver doesn't crash.

**GPU hang detector.** Background watchdog thread monitors fences. If one
doesn't signal within a timeout, dumps everything in flight. Pair it with
`BreadcrumbBuffer` - a small CPU-visible GPU buffer that records sequential
markers via `vkCmdFillBuffer`. After a hang, you know exactly which draw or
dispatch was the last to complete.

**Submission journal.** Lock-free ring buffer logging every queue submission
with timestamps, semaphores, and fence handles. When you get
`VK_ERROR_DEVICE_LOST`, dump the journal and see the chronological sequence
of everything that was in flight.

**Thread safety auditor.** `AuditedPool` wraps `CommandPool` and remembers
which thread owns it. Access from a different thread produces an immediate
diagnostic instead of a sporadic crash 10000 frames later.

**Resource aliasing detector.** Tracks read/write accesses within a recording
session. Detects write-then-read without a barrier and write-write conflicts.

**Memory budget monitor.** Polls `VK_EXT_memory_budget` and warns when heap
usage approaches configurable thresholds (80%, 90%, 95%). Shows per-heap
utilization bars so you know where your memory is going.

**Descriptor set validator.** Tracks resource liveness and descriptor writes.
Before draw time, catches descriptors that still reference destroyed buffers or
images.

**Pipeline compatibility checker.** Validates that bound descriptor set counts
match the pipeline layout, and that push constant ranges cover what you're
actually pushing.

**Barrier optimizer.** Records pipeline barriers and flags suboptimal patterns:
`ALL_COMMANDS` stages (serializes the GPU), `MEMORY_READ|MEMORY_WRITE` access
masks (too broad), redundant consecutive barriers. Suggests tighter masks based
on actual access patterns.

## Device modes

Ignis supports two modes:

**Managed** - Ignis creates the Vulkan instance, picks a physical device,
creates the logical device and queues. You own nothing, Ignis cleans up on drop.
Good for standalone apps.

```rust
let Ignis = Ignis::managed(
    ManagedConfig::new("my_app", vk::API_VERSION_1_3)
        .enable_validation(true)
        .enable_raytracing(true)
)?;
```

**External** - you already have a device (from wgpu, vulkano, egui, whatever).
Give Ignis the handles and it wraps them without taking ownership. Nothing gets
destroyed when Ignis drops.

```rust
let Ignis = Ignis::external(ExternalDeviceInfo {
    instance: my_instance.clone(),
    device: my_device.clone(),
    physical_device: my_physical,
    queue_allocations: vec![...],
    enable_raytracing: false,
})?;
```

Both modes expose the same API. The `DeviceHandle` trait lets you pass Ignis
(or your own engine) to utility functions generically.

## Quick example

```rust
use Ignis::*;
use ash::vk;

let Ignis = Ignis::managed(
    ManagedConfig::new("demo", vk::API_VERSION_1_2)
)?;

let alloc = Ignis.create_block_allocator();
let queue = Ignis.queue(QueueType::Graphics)?;
let pool = Ignis.create_command_pool(QueueType::Graphics)?;
let sync = Ignis.create_frame_sync(2)?;

// Per-frame loop
let frame = sync.begin_frame()?;

let staging = Ignis.create_buffer_with(&alloc, &BufferInfo::staging(1024))?;
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

## Development vs production

The debugging tools are designed for development builds. In production, swap
one line:

```rust
// Development:
let alloc = Ignis.create_hardened_allocator(HardenedConfig::default());

// Production:
let alloc = Ignis.create_block_allocator();
```

Same `Arc<dyn Allocator>`, same API, zero overhead from the debugging layer.

The other tools (lifetime tracker, state validator, hang detector, etc.) simply
aren't constructed in production code - they don't exist, they cost nothing.

## Ray tracing

First-class support for `VK_KHR_ray_tracing_pipeline`:

```rust
let Ignis = Ignis::managed(
    ManagedConfig::new("rt_app", vk::API_VERSION_1_2)
        .enable_raytracing(true)
)?;

let pipeline = Ignis.raytracing_pipeline_builder()?
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

The `RayTracingPipeline` computes SBT region offsets, alignments, and strides.
You allocate the GPU buffer and copy the handle data - Ignis doesn't assume
which allocator you're using.

## Interop

Ignis plays nice with other crates. It only depends on `ash` and doesn't touch
global state. If you're using wgpu or vulkano for rendering and just want Ignis
for queue orchestration or debugging:

1. Get the raw Vulkan handles from your engine
2. Create Ignis in external mode
3. Use whichever Ignis modules you need
4. Both libraries operate on separate Vulkan objects and don't interfere

The `Allocator` trait can wrap gpu-allocator or vk-mem if you prefer those
for memory management but still want Ignis's hardened allocator layer on top.

## Platform support

Tested on Windows and Linux. macOS support is included (portability subset
extensions) but not regularly tested - MoltenVK may have quirks.

## Building

```sh
cargo build
cargo run --example smoke_test
```

The smoke test exercises every subsystem headlessly (no window needed) and
produces a pass/skip/fail report across 32 test steps. It requires a Vulkan
driver - integrated GPUs work fine.