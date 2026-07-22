//! Validation Layer baseline capture and diff demonstration.
//!
//! Steps:
//!  1. Create a context with validation enabled.
//!  2. Run a "smoke test" workload that deliberately commits a couple
//!     of validation violations (oversized copy, bad layout transition).
//!  3. Dump the resulting baseline to disk.
//!  4. Reset the collector.
//!  5. Run a different workload that triggers a NEW violation
//!     (extra oversized copy of a different size, plus the original two).
//!  6. Diff against the on-disk baseline. Show that
//!     `has_regressions()` is true.
//!  7. Run yet another workload that DOES NOT trigger any violations,
//!     diff again. Show that `has_regressions()` is false but
//!     `has_improvements()` is true (the previously-emitted VUIDs are
//!     gone).
//!
//! Run with:
//! ```sh
//! cargo run --example vl_baseline_demo --features debug-tools
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("vl_baseline_demo requires --features debug-tools");

use ash::vk;
use ignis::{BufferInfo, Ignis, ManagedConfig, MemoryLocation, QueueType};

const TOTAL_STEPS: u32 = 7;

fn main() {
    println!();
    println!("    IGNIS VL BASELINE DEMO");
    println!("    Capture and diff validation diagnostics across runs.");
    println!();

    if let Err(e) = run() {
        eprintln!();
        eprintln!("    FATAL: {e}");
        std::process::exit(1);
    }

    println!();
    println!("    DONE");
    println!();
}

fn run() -> ignis::Result<()> {
    // Step 1: validation-enabled context.
    step(1, "Create context with validation enabled");
    let ctx = Ignis::managed(
        ManagedConfig::new("vl-baseline-demo", vk::API_VERSION_1_2)
            .enable_validation(true),
    )?;
    info("validation layer is on; every VUID will be captured");
    ok();

    let baseline_path =
        std::env::temp_dir().join(format!("ignis_baseline_{}.vl", std::process::id()));

    // Step 2: trigger two distinct violations.
    step(2, "Run baseline workload (2 violations)");
    trigger_oversized_copy(&ctx, 64, 128)?; // VUID for source
    trigger_oversized_copy(&ctx, 64, 128)?; // again -> count 2
    trigger_bad_layout(&ctx)?;
    info("baseline workload completed");
    ok();

    // Step 3: dump baseline.
    step(3, "Dump baseline to disk");
    ctx.dump_vl_baseline(&baseline_path)?;
    let metadata = std::fs::metadata(&baseline_path)
        .map_err(|_| ignis::Error::InvalidConfig("could not stat baseline"))?;
    info(&format!(
        "baseline written to {} ({} bytes)",
        baseline_path.display(),
        metadata.len()
    ));
    let snapshot = ignis::vl_baseline_snapshot();
    info(&format!(
        "snapshot has {} unique VUID(s), {} total emission(s)",
        snapshot.unique_count(),
        snapshot.total_count()
    ));
    ok();

    // Step 4: reset and run a regressed workload.
    step(4, "Reset collector, run REGRESSED workload");
    ctx.reset_vl_baseline();
    trigger_oversized_copy(&ctx, 64, 128)?; // same as before
    trigger_oversized_copy(&ctx, 64, 128)?; // same as before
    trigger_bad_layout(&ctx)?;
    trigger_oversized_copy(&ctx, 32, 96)?; // NEW: different sizes
    info("regressed workload introduces new copy violation");
    ok();

    // Step 5: diff and inspect regressions.
    step(5, "Diff against on-disk baseline");
    let diff = ctx.diff_vl_baseline(&baseline_path)?;
    println!();
    eprint!("{diff}");
    println!();
    info(&format!(
        "has_regressions = {}  has_improvements = {}",
        diff.has_regressions(),
        diff.has_improvements()
    ));
    if diff.has_regressions() {
        info("CI would fail the build at this point");
    }
    ok();

    // Step 6: reset and run a clean workload.
    step(6, "Reset, run CLEAN workload (no violations)");
    ctx.reset_vl_baseline();
    info("performing only valid Vulkan operations");
    let _buf = ctx.create_buffer(&BufferInfo::staging(1024))?;
    info("clean workload completed");
    ok();

    // Step 7: diff again, expect improvements only.
    step(7, "Diff clean workload against baseline");
    let diff2 = ctx.diff_vl_baseline(&baseline_path)?;
    println!();
    eprint!("{diff2}");
    println!();
    info(&format!(
        "has_regressions = {}  has_improvements = {}",
        diff2.has_regressions(),
        diff2.has_improvements()
    ));
    if !diff2.has_regressions() && diff2.has_improvements() {
        info("baseline could be updated to lock in the improvement");
    }
    ok();

    let _ = std::fs::remove_file(&baseline_path);
    Ok(())
}

/// Force a VUID by copying more bytes than the source/destination buffers hold.
fn trigger_oversized_copy(ctx: &Ignis, buf_size: u64, copy_size: u64) -> ignis::Result<()> {
    let queue = ctx.queue(QueueType::Graphics)?;
    let pool = ctx.create_command_pool(QueueType::Graphics)?;
    let src = ctx.create_buffer(&BufferInfo::staging(buf_size))?;
    let dst = ctx.create_buffer(&BufferInfo {
        size: buf_size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        location: MemoryLocation::GpuOnly,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
    })?;
    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;
    rec.copy_buffer(
        src.handle(),
        dst.handle(),
        &[vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: copy_size, // intentionally larger than buf_size
        }],
    );
    let cmd = rec.end()?;
    let _ = queue.submit_simple(cmd).and_then(|f| f.wait());
    Ok(())
}

/// Force a layout-vs-usage VUID by transitioning a SAMPLED-only image
/// to COLOR_ATTACHMENT_OPTIMAL.
fn trigger_bad_layout(ctx: &Ignis) -> ignis::Result<()> {
    let queue = ctx.queue(QueueType::Graphics)?;
    let pool = ctx.create_command_pool(QueueType::Graphics)?;
    let img = ctx.create_image(&ignis::ImageInfo::texture_2d(
        32,
        32,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED, // missing COLOR_ATTACHMENT
    ))?;
    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;
    rec.pipeline_barrier(
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .image(img.handle())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })],
    );
    let cmd = rec.end()?;
    let _ = queue.submit_simple(cmd).and_then(|f| f.wait());
    Ok(())
}

fn step(n: u32, t: &str) {
    println!("[{n:>2}/{TOTAL_STEPS}] {t}");
}
fn info(m: &str) {
    println!("       {m}");
}
fn ok() {
    println!("       PASSED");
    println!();
}