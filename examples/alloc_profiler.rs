//! Allocation site profiler + memory layout visualizer demonstration.
//!
//! Creates a managed Ignis context, wraps the default block allocator with
//! [`AllocationProfiler`], simulates a typical engine workload (textures,
//! meshes, uniform buffers, staging) where each "subsystem" allocates from
//! a distinct call site, then produces:
//!
//! 1. A ranked report on stderr showing the top N call sites by active bytes.
//! 2. An SVG visualization of the memory layout written to disk.
//!
//! The example also demonstrates peak tracking: half of the allocations are
//! freed mid-run and a second report is produced. Active counters drop
//! while peak counters remain at their high-water mark.
//!
//! Run with:
//! ```sh
//! cargo run --example alloc_profiler --features full
//! ```
//!
//! Output files:
//! - `alloc_profiler_full.svg`  layout snapshot with all allocations live
//! - `alloc_profiler_after.svg` layout snapshot after freeing half

#[cfg(not(feature = "debug-tools"))]
compile_error!("alloc_profiler requires --features debug-tools");

use std::sync::Arc;

use ash::vk;
use ignis::{
    Allocator, Buffer, BufferInfo, Ignis, Image, ImageInfo, ManagedConfig, MemoryLocation,
};

const TOTAL_STEPS: u32 = 7;

fn main() {
    println!();
    println!("    IGNIS ALLOCATION PROFILER + MEMORY VISUALIZER DEMO");
    println!("    Wraps the block allocator, attributes each allocation to its");
    println!("    Rust call site, and renders an SVG layout of live memory.");
    println!();

    if let Err(e) = run() {
        eprintln!();
        eprintln!("    FATAL: {e}");
        eprintln!();
        std::process::exit(1);
    }

    println!();
    println!("    DONE");
    println!();
}

fn run() -> ignis::Result<()> {
    // Step 1: managed context. No validation noise for this demo.
    step(1, "Create managed Ignis context");
    let ctx = Ignis::managed(
        ManagedConfig::new("alloc-profiler-demo", vk::API_VERSION_1_2)
            .enable_validation(false),
    )?;
    let dev_name = unsafe {
        std::ffi::CStr::from_ptr(ctx.device_properties().device_name.as_ptr())
    }
    .to_str()
    .unwrap_or("<unknown>");
    info(&format!("device: {dev_name}"));
    ok();

    // Step 2: build the profiled allocator. The profiler returns an
    // Arc<AllocationProfiler>; we clone it once for each role:
    //   - `profiler` keeps the AllocationProfiler API (snapshot, reports)
    //   - `alloc` is the Arc<dyn Allocator> handle used by create_*_with()
    step(2, "Wrap block allocator with AllocationProfiler");
    let profiler = ctx.create_profiled_block_allocator();
    let alloc: Arc<dyn Allocator> = profiler.clone();
    info("profiler installed; backtrace capture is enabled by default");
    ok();

    // Step 3: simulate engine subsystems. Each helper function allocates
    // from a distinct call site so the profiler has interesting data to
    // attribute. Container vectors keep the resources alive until we
    // explicitly drop them later.
    step(3, "Simulate engine subsystems (textures, meshes, uniforms, staging)");
    let textures = load_textures(&ctx, &alloc, 12)?;
    info(&format!(
        "loaded {} textures (total {})",
        textures.len(),
        format_bytes(approx_image_bytes(&textures))
    ));

    let meshes = load_meshes(&ctx, &alloc, 6)?;
    info(&format!(
        "loaded {} meshes (vertex+index buffers)",
        meshes.len() * 2
    ));

    let uniforms = create_uniforms(&ctx, &alloc, 64)?;
    info(&format!("created {} uniform buffers", uniforms.len()));

    let staging = create_staging(&ctx, &alloc, 8)?;
    info(&format!("created {} staging buffers", staging.len()));
    ok();

    // Step 4: first report - everything is live, peaks equal active.
    step(4, "Report: full population");
    info(&format!(
        "totals: {} allocs, {} active, {} bytes live",
        profiler.total_allocations(),
        profiler.active_allocations(),
        format_bytes(profiler.active_bytes())
    ));
    println!();
    eprint!("{}", profiler.report_top_sites(8));
    println!();
    ok();

    // Step 5: dump SVG of the full population.
    step(5, "Render full-population layout SVG");
    let viz = ctx.create_memory_visualizer();
    let path1 = "alloc_profiler_full.svg";
    viz.save_svg(&profiler, path1).expect("failed to write SVG");
    info(&format!(
        "wrote {} ({} bytes)",
        path1,
        std::fs::metadata(path1).map(|m| m.len()).unwrap_or(0)
    ));
    ok();

    // Step 6: free half the textures and all staging. Active counters
    // drop, peaks stay. The second SVG will show fewer rectangles per row
    // (or fewer rows if a whole VkDeviceMemory got drained).
    step(6, "Free half of textures and all staging");
    let mut textures = textures;
    let dropped = textures.split_off(textures.len() / 2);
    drop(dropped);
    drop(staging);
    info(&format!(
        "after free: {} active allocs, {} bytes live",
        profiler.active_allocations(),
        format_bytes(profiler.active_bytes())
    ));
    info(&format!(
        "peaks retained: {} bytes peak active observed during the run",
        format_bytes(peak_total_active_bytes(&profiler))
    ));
    println!();
    eprint!("{}", profiler.report_top_sites(8));
    println!();
    ok();

    // Step 7: dump second SVG and tear down.
    step(7, "Render reduced-population layout SVG");
    let path2 = "alloc_profiler_after.svg";
    viz.save_svg(&profiler, path2).expect("failed to write SVG");
    info(&format!(
        "wrote {} ({} bytes)",
        path2,
        std::fs::metadata(path2).map(|m| m.len()).unwrap_or(0)
    ));
    info("compare both SVGs side by side in any browser to see the diff");
    ok();

    // Drop remaining resources before the context.
    drop(textures);
    drop(meshes);
    drop(uniforms);
    drop(profiler);
    drop(ctx);

    Ok(())
}

// Subsystem simulators. Each function is a distinct call site so the
// profiler can attribute its allocations independently.

#[inline(never)]
fn load_textures(
    ctx: &Ignis,
    alloc: &Arc<dyn Allocator>,
    count: usize,
) -> ignis::Result<Vec<Image>> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // Vary sizes so the visualizer shows distinguishable rectangles.
        let dim = match i % 4 {
            0 => 256,
            1 => 512,
            2 => 1024,
            _ => 2048,
        };
        let img = ctx.create_image_with(
            alloc,
            &ImageInfo::texture_2d(
                dim,
                dim,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ),
        )?;
        out.push(img);
    }
    Ok(out)
}

#[inline(never)]
fn load_meshes(
    ctx: &Ignis,
    alloc: &Arc<dyn Allocator>,
    count: usize,
) -> ignis::Result<Vec<(Buffer, Buffer)>> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let vbo_size = (256 * 1024 * (1 + i as u64 % 4)) as vk::DeviceSize;
        let ibo_size = (64 * 1024 * (1 + i as u64 % 4)) as vk::DeviceSize;
        let vbo = ctx.create_buffer_with(
            alloc,
            &BufferInfo::vertex(vbo_size, MemoryLocation::GpuOnly),
        )?;
        let ibo = ctx.create_buffer_with(
            alloc,
            &BufferInfo::index(ibo_size, MemoryLocation::GpuOnly),
        )?;
        out.push((vbo, ibo));
    }
    Ok(out)
}

#[inline(never)]
fn create_uniforms(
    ctx: &Ignis,
    alloc: &Arc<dyn Allocator>,
    count: usize,
) -> ignis::Result<Vec<Buffer>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // 256-byte UBOs, the common camera/material constant size.
        let buf = ctx.create_buffer_with(alloc, &BufferInfo::uniform(256))?;
        out.push(buf);
    }
    Ok(out)
}

#[inline(never)]
fn create_staging(
    ctx: &Ignis,
    alloc: &Arc<dyn Allocator>,
    count: usize,
) -> ignis::Result<Vec<Buffer>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // 1 MiB upload staging.
        let buf = ctx.create_buffer_with(alloc, &BufferInfo::staging(1024 * 1024))?;
        out.push(buf);
    }
    Ok(out)
}

// Helpers.

/// Estimate texture bytes from extent and format. Real driver allocation
/// will round up to alignment, but this is good enough for a banner.
fn approx_image_bytes(images: &[Image]) -> u64 {
    images
        .iter()
        .map(|i| {
            let e = i.extent();
            (e.width as u64) * (e.height as u64) * 4
        })
        .sum()
}

/// Sum of `peak_active_bytes` across all sites. Approximates the
/// high-water mark of the entire profiler in bytes.
fn peak_total_active_bytes(profiler: &Arc<ignis::AllocationProfiler>) -> u64 {
    profiler
        .snapshot()
        .iter()
        .map(|(_, st)| st.peak_active_bytes)
        .sum()
}

fn format_bytes(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", b as f64 / 1_073_741_824.0)
    } else if b >= 1024 * 1024 {
        format!("{:.1} MiB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

fn step(n: u32, title: &str) {
    println!("[{n:>2}/{TOTAL_STEPS}] {title}");
}

fn info(msg: &str) {
    println!("       {msg}");
}

fn ok() {
    println!("       PASSED");
    println!();
}