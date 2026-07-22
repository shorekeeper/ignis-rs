//! Real-time debug window demonstration.
//!
//! Opens a native debug window on Windows, configured with both a memory
//! panel (driven by AllocationProfiler) and a timeline panel (driven by
//! ResourceTrace). On the main thread, runs a synthetic workload that
//! periodically allocates and frees buffers/images, dispatches fake
//! "submissions" and "passes" via `record_submission` / `record_pass`,
//! and emits custom events. The window updates live.
//!
//! Close the window with the system close button, or press Ctrl+C to
//! exit the application (the window closes automatically when its
//! handle is dropped).
//!
//! Run with:
//! ```sh
//! cargo run --example debug_window_demo --features full
//! ```

#[cfg(not(feature = "debug-window"))]
compile_error!("debug_window_demo requires --features debug-window (implied by --features full)");

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("debug_window_demo currently runs on Windows only.");
    eprintln!("On Linux/macOS the builder returns an error at open() time.");
    std::process::exit(0);
}

#[cfg(target_os = "windows")]
fn main() {
    if let Err(e) = run() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "windows")]
fn run() -> ignis::Result<()> {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ash::vk;
    use ignis::{
        debug_window::DebugWindow, AllocationProfiler, Allocator, Buffer, BufferInfo, Ignis,
        Image, ImageInfo, ManagedConfig, MemoryLocation, ResourceTrace,
    };

    println!("==================================================");
    println!(" IGNIS REAL-TIME DEBUG WINDOW DEMO");
    println!("==================================================");
    println!();

    let ctx = Ignis::managed(
        ManagedConfig::new("debug-window-demo", vk::API_VERSION_1_2)
            .enable_validation(false)
            .instance_extension(ash::khr::surface::NAME)
            .instance_extension(ash::khr::win32_surface::NAME)
            .device_extension(ash::khr::swapchain::NAME),
    )?;
    println!("[1/4] Ignis context ready on {}",
        unsafe { std::ffi::CStr::from_ptr(ctx.device_properties().device_name.as_ptr()) }
            .to_str().unwrap_or("?"));

    // Resource trace + profiler with trace mirroring.
    let trace = ResourceTrace::new(8000);
    let inner = ctx.create_block_allocator();
    let profiler = AllocationProfiler::new(inner);
    profiler.with_trace(Some(Arc::clone(&trace)));
    println!("[2/4] Profiler + trace wired up");

    let alloc: Arc<dyn Allocator> = profiler.clone();

    // Open the debug window.
    let win = DebugWindow::builder()
        .title("Ignis Debug")
        .size(1400, 800)
        .memory_source(Arc::clone(&profiler))
        .trace_source(Arc::clone(&trace))
        .refresh_hz(60)
        .timeline_window_ms(8_000)
        .open(&ctx)?;
    println!("[3/4] Debug window open. Close it or wait for the demo to finish.");

    // Synthetic workload: simulate allocations, frees, fake submissions,
    // fake passes, and custom events. We keep buffers/images in vectors
    // so we can drop them gradually to drive the "free" lane.
    let mut buffers: Vec<Buffer> = Vec::new();
    let mut images: Vec<Image> = Vec::new();
    let start = Instant::now();
    let total_duration = Duration::from_secs(45);
    let mut frame: u64 = 0;

    println!("[4/4] Running synthetic workload for {} seconds...", total_duration.as_secs());
    println!();

    while start.elapsed() < total_duration && !win.is_closed() {
        frame += 1;

        // Every frame: a "render submit".
        let submit_label = format!("frame_{}", frame);
        trace.record_submission(0, 0, &submit_label, 1_500_000 + (frame as u64 % 5) * 200_000);

        // Every frame: 3 simulated passes.
        trace.record_pass("shadow_pass", 250_000);
        trace.record_pass("geometry_pass", 800_000);
        trace.record_pass("post_processing", 400_000);

        // Periodic transitions.
        if frame % 4 == 0 {
            trace.record_transition(
                "Image",
                0xABCD_0000 + frame,
                "COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL",
            );
        }

        // Custom event per second.
        if frame % 30 == 0 {
            trace.record_custom("user", "checkpoint", &format!("frame={frame}"));
        }

        // Allocations: every few frames, add some resources.
        if frame % 3 == 0 && buffers.len() < 60 {
            let size = 64 * 1024 * (1 + (frame % 8));
            buffers.push(ctx.create_buffer_with(
                &alloc,
                &BufferInfo::storage(size as u64, MemoryLocation::GpuOnly),
            )?);
        }
        if frame % 8 == 0 && images.len() < 30 {
            let dim = 256 + 128 * ((frame % 4) as u32);
            images.push(ctx.create_image_with(
                &alloc,
                &ImageInfo::texture_2d(
                    dim,
                    dim,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            )?);
        }

        // Frees: drop the oldest resource periodically to make the
        // memory layout panel visibly evolve.
        if frame % 11 == 0 && !buffers.is_empty() {
            let _ = buffers.remove(0);
        }
        if frame % 23 == 0 && !images.is_empty() {
            let _ = images.remove(0);
        }

        std::thread::sleep(Duration::from_millis(33));
    }

    println!("[done] Workload complete. Closing window.");

    // Export the final trace as Chrome JSON for offline inspection.
    let trace_path = "debug_window_demo_trace.json";
    trace.export_chrome_json(trace_path).ok();
    println!("[done] Wrote Chrome trace JSON to {}", trace_path);
    println!("       Open in chrome://tracing or https://ui.perfetto.dev/");

    // Window closes when win drops; resources release on context drop.
    drop(win);
    drop(buffers);
    drop(images);
    drop(profiler);
    drop(ctx);

    Ok(())
}