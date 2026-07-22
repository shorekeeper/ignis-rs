//! Device fault diagnostics + intent-based resource creation demo.
//!
//! Demonstrates two complementary debug features:
//!
//! 1. **Static intent validation** at resource creation time: declare
//!    how an image or buffer will be used, and ignis fails fast if the
//!    usage flags do not cover the declared accesses. The mistake is
//!    rejected before any VkImage/VkBuffer is created, far cleaner than
//!    a downstream layout transition error.
//!
//! 2. **Device fault diagnostic recording**: insert NV checkpoints and
//!    AMD buffer markers during normal command recording. After a
//!    DEVICE_LOST (simulated here, since triggering a real one is
//!    dangerous), the CrashReporter aggregates all available data into
//!    a single markdown report including per-stage checkpoint values,
//!    per-slot marker fired status, and EXT_device_fault description.
//!
//! On hardware where some extensions are missing (most common: a card
//! without VK_NV_device_diagnostic_checkpoints), the corresponding
//! sections are simply absent from the report. The demo runs to
//! completion regardless.
//!
//! Run with:
//! ```sh
//! cargo run --example device_fault_demo --features full
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("device_fault_demo requires --features debug-tools");

use std::sync::Arc;

use ash::vk;
use ignis::{
    AmdMarkerBuffer, BufferInfo, BufferUsageContext, DeviceFaultRecorder, Ignis, ImageInfo,
    ImageUsageContext, ManagedConfig, MemoryLocation, QueueType,
};

fn main() {
    println!();
    println!("    IGNIS DEVICE FAULT + INTENT VALIDATION DEMO");
    println!();
    if let Err(e) = run() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}

fn run() -> ignis::Result<()> {
    // Step 1: build a context with device fault extensions enabled.
    let ctx = Ignis::managed(
        ManagedConfig::new("device-fault-demo", vk::API_VERSION_1_3)
            .enable_validation(false)
            .enable_device_fault(true),
    )?;
    let dev_name = unsafe {
        std::ffi::CStr::from_ptr(ctx.device_properties().device_name.as_ptr())
    }
    .to_str()
    .unwrap_or("?");
    println!("[1/6] device: {dev_name}");

    // Step 2: probe extension availability.
    let recorder = ctx.create_device_fault_recorder();
    println!("[2/6] extension probe:");
    println!(
        "        VK_NV_device_diagnostic_checkpoints = {}",
        recorder.supports_checkpoints()
    );
    println!(
        "        VK_EXT_device_fault                 = {}",
        recorder.supports_fault_info()
    );
    println!(
        "        VK_AMD_buffer_marker                = {}",
        recorder.supports_buffer_markers()
    );

    // Step 3: intent validation. First, a CORRECT call.
    println!("[3/6] intent validation:");
    let _good_image = ctx.create_image_with_intent(
        &ImageInfo::texture_2d(
            128,
            128,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        ),
        &[ImageUsageContext::TransferDst, ImageUsageContext::FragmentShaderRead],
    )?;
    println!("        good image accepted (SAMPLED|TRANSFER_DST satisfies intents)");

    // Step 3b: an INCORRECT call. We declare ColorAttachment intent but
    // forget to include COLOR_ATTACHMENT in info.usage. ignis rejects.
    let bad_image_result = ctx.create_image_with_intent(
        &ImageInfo::texture_2d(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED, // missing COLOR_ATTACHMENT
        ),
        &[ImageUsageContext::ColorAttachment],
    );
    match bad_image_result {
        Err(e) => println!("        bad image correctly rejected: {e}"),
        Ok(_) => panic!("bad image should have been rejected"),
    }

    // Same idea with buffers.
    let bad_buffer_result = ctx.create_buffer_with_intent(
        &BufferInfo::staging(1024),
        &[BufferUsageContext::IndirectDraw],
    );
    match bad_buffer_result {
        Err(e) => println!("        bad buffer correctly rejected: {e}"),
        Ok(_) => panic!("bad buffer should have been rejected"),
    }

    // Step 4: insert NV checkpoints + AMD markers during a normal submit.
    println!("[4/6] recording diagnostic markers in a synthetic submit");
    let queue = ctx.queue(QueueType::Graphics)?;
    let raw_queue = unsafe {
        ctx.device()
            .get_device_queue(queue.family_index(), queue.queue_index())
    };
    let pool = ctx.create_command_pool(QueueType::Graphics)?;

    let markers: Option<Arc<AmdMarkerBuffer>> = if recorder.supports_buffer_markers() {
        Some(recorder.create_marker_buffer(64)?)
    } else {
        None
    };

    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;

    // 4 checkpoints encoding pass + frame number:
    let pack = |pass_id: u32, frame: u32| ((pass_id as u64) << 32) | (frame as u64);
    recorder.cmd_checkpoint(&rec, pack(1, 0));
    recorder.cmd_checkpoint(&rec, pack(2, 0));
    recorder.cmd_checkpoint(&rec, pack(3, 0));
    recorder.cmd_checkpoint(&rec, pack(4, 0));

    // 5 buffer markers at different stages:
    if let Some(m) = &markers {
        recorder.cmd_buffer_marker(&rec, m, "begin_frame", vk::PipelineStageFlags::TOP_OF_PIPE);
        recorder.cmd_buffer_marker(&rec, m, "vertex_done", vk::PipelineStageFlags::VERTEX_SHADER);
        recorder.cmd_buffer_marker(
            &rec,
            m,
            "fragment_done",
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        recorder.cmd_buffer_marker(
            &rec,
            m,
            "color_attachment_done",
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        recorder.cmd_buffer_marker(
            &rec,
            m,
            "end_frame",
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );
    }

    let cmd = rec.end()?;
    queue.submit_simple(cmd)?.wait()?;
    unsafe { ctx.device().queue_wait_idle(raw_queue)? };
    println!("        submit complete; markers and checkpoints have fired");

    // Step 5: read back diagnostic data from a healthy device. NV
    // checkpoints return last-completed-per-stage; AMD markers report
    // the fired/pending status of each slot.
    println!("[5/6] readback (no fault, healthy device):");
    let data = recorder.collect_all(Some(raw_queue), markers.as_ref());
    println!(
        "        checkpoints: {}, markers: {}, fault_info: {}",
        data.checkpoints.len(),
        data.buffer_markers.len(),
        data.fault_info
            .as_ref()
            .map(|fi| fi.description.as_str())
            .unwrap_or("(extension absent)"),
    );
    let fired = data.buffer_markers.iter().filter(|m| m.fired).count();
    println!(
        "        markers fired: {}/{}",
        fired,
        data.buffer_markers.len()
    );

    // Step 6: integrate with CrashReporter and produce the unified
    // markdown report. We pretend a DEVICE_LOST happened so the report
    // includes the device fault section.
    println!("[6/6] generating crash report (simulated DEVICE_LOST)");
    let reporter = ctx.create_crash_reporter();
    reporter.attach_device_fault(Arc::clone(&recorder));
    reporter.set_fault_queue(raw_queue);
    if let Some(m) = markers {
        reporter.set_fault_markers(m);
    }
    reporter.add_section(
        "Demo Context",
        "This is a simulated DEVICE_LOST report from device_fault_demo.\n\
         No real fault occurred. The Device Fault Diagnostics section\n\
         below shows live data from healthy execution.",
    );

    let report = reporter.generate(vk::Result::ERROR_DEVICE_LOST);
    let path = "device_fault_demo_report.md";
    std::fs::write(path, &report.body).ok();
    println!(
        "        wrote {} ({} bytes)",
        path,
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    );
    println!(
        "        report includes: env block, device fault section, custom section"
    );

    drop(ctx);

    println!();
    println!("    DONE");
    println!();
    Ok(())
}