//! Deliberate validation layer trigger tests.
//!
//! Exercises the forensic analyzer in src/debug/validation_forensic.rs by
//! committing specific Vulkan spec violations that the layer reliably
//! catches. Each subtest runs in an isolated Ignis context so driver state
//! from one violation cannot affect the next.
//!
//! The forensic pipeline is verified by:
//!   1. Installing a structured handler that records every diagnostic.
//!   2. Triggering a known violation via a deliberately-broken code path.
//!   3. Checking that the handler fired with a parsed VUID.
//!   4. Reporting whether the knowledge base matched the VUID.
//!
//! Expected output when the validation layer is loaded:
//!   - Each trigger should log the pretty-formatted forensic diagnostic
//!     to stderr (rich framed box with VUID, objects, fix suggestion).
//!   - The structured handler should report at least one captured VUID
//!     per trigger.
//!   - Knowledge base matches indicate which VUIDs our curated database
//!     understands; unmatched ones still produce structured parse output
//!     but without the "what you did / how to fix" explanation.
//!
//! Run with:
//! ```sh
//! cargo run --example trigger_validation --features full
//! ```

#[cfg(not(feature = "full"))]
compile_error!("trigger_validation requires --features full");

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

/// Records everything the structured validation handler receives.
///
/// Using Arc + interior mutability lets us clone it into the handler
/// closure while keeping the original accessible from the test body.
struct HandlerCapture {
    vuids: Mutex<Vec<String>>,
    knowledge_titles: Mutex<Vec<String>>,
    categories: Mutex<Vec<ignis::DiagnosticCategory>>,
    fired: AtomicBool,
}

impl HandlerCapture {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            vuids: Mutex::new(Vec::new()),
            knowledge_titles: Mutex::new(Vec::new()),
            categories: Mutex::new(Vec::new()),
            fired: AtomicBool::new(false),
        })
    }

    fn install(self: &Arc<Self>, ctx: &ignis::Ignis) {
        let me = Arc::clone(self);
        ctx.set_validation_handler(move |diag| {
            me.fired.store(true, Ordering::SeqCst);
            me.vuids.lock().unwrap().push(diag.vuid.clone());
            me.categories.lock().unwrap().push(diag.category);
            if let Some(kb) = &diag.knowledge {
                me.knowledge_titles
                    .lock()
                    .unwrap()
                    .push(kb.title.to_string());
            }
        });
    }

    fn report(&self) {
        let vuids = self.vuids.lock().unwrap();
        let titles = self.knowledge_titles.lock().unwrap();
        let cats = self.categories.lock().unwrap();

        println!(
            "    handler fired:      {}",
            self.fired.load(Ordering::SeqCst)
        );
        println!("    VUIDs captured:     {}", vuids.len());
        for v in vuids.iter().take(5) {
            println!("      {v}");
        }
        if vuids.len() > 5 {
            println!("      ... {} more", vuids.len() - 5);
        }
        println!("    knowledge matches:  {}", titles.len());
        for t in titles.iter().take(5) {
            println!("      -> {t}");
        }
        if !cats.is_empty() {
            let first = cats.first().unwrap();
            println!("    primary category:   {first:?}");
        }
    }

    fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

/// Build a fresh validation-enabled context.
fn make_ctx() -> ignis::Result<ignis::Ignis> {
    ignis::Ignis::managed(
        ignis::ManagedConfig::new("vl-trigger", vk::API_VERSION_1_2).enable_validation(true),
    )
}

fn main() {
    println!();
    println!("  VALIDATION LAYER FORENSIC TRIGGER TESTS");
    println!("  deliberate spec violations, verifying forensic pipeline");
    println!();

    let mut total = 0u32;
    let mut passed = 0u32;

    total += 1;
    println!("[1/3] trigger: image clear without TRANSFER_DST usage");
    println!("      expected VUIDs: VUID-*-oldLayout-01213, VUID-vkCmdClearColorImage-image-00002");
    match test_clear_without_transfer_dst() {
        Ok(true) => {
            println!("    [PASS] forensic handler captured a parsed VUID");
            passed += 1;
        }
        Ok(false) => {
            println!("    [WARN] handler did not fire (layer not loaded?)");
        }
        Err(e) => println!("    [FAIL] test setup error: {e}"),
    }
    println!();

    total += 1;
    println!("[2/3] trigger: barrier to layout incompatible with usage");
    println!("      expected VUIDs: VUID-VkImageMemoryBarrier-*-01213 family");
    match test_bad_layout_transition() {
        Ok(true) => {
            println!("    [PASS] forensic handler captured a parsed VUID");
            passed += 1;
        }
        Ok(false) => {
            println!("    [WARN] handler did not fire (layer not loaded?)");
        }
        Err(e) => println!("    [FAIL] test setup error: {e}"),
    }
    println!();

    total += 1;
    println!("[3/3] trigger: buffer copy region exceeds source size");
    println!("      expected VUIDs: VUID-vkCmdCopyBuffer-size-00115 or srcOffset family");
    match test_copy_out_of_bounds() {
        Ok(true) => {
            println!("    [PASS] forensic handler captured a parsed VUID");
            passed += 1;
        }
        Ok(false) => {
            println!("    [WARN] handler did not fire (layer not loaded?)");
        }
        Err(e) => println!("    [FAIL] test setup error: {e}"),
    }
    println!();

    println!("  RESULTS: {passed}/{total} tests successfully triggered the forensic pipeline");
    println!();
    if passed == 0 {
        println!("  no triggers fired. check that:");
        println!("    - VK_LAYER_KHRONOS_validation is installed");
        println!("    - Vulkan SDK is on PATH");
        println!("    - validation is enabled (it is by default for these tests)");
    } else if passed < total {
        println!("  some triggers did not fire. possible causes:");
        println!("    - layer version does not emit these VUIDs anymore");
        println!("    - driver filtered the command before layer saw it");
    } else {
        println!("  ALL FORENSIC TRIGGERS OK");
    }
    println!();
}

/// Test 1: create an image without TRANSFER_DST, try to clear it.
///
/// Two spec violations in one command buffer:
///   - The layout transition to TRANSFER_DST_OPTIMAL requires
///     TRANSFER_DST_BIT in usage (VUID-VkImageMemoryBarrier-oldLayout-01213).
///   - The subsequent cmd_clear_color_image also requires TRANSFER_DST_BIT
///     (VUID-vkCmdClearColorImage-image-00002).
///
/// Both VUIDs are in our knowledge base, so we expect two knowledge hits.
fn test_clear_without_transfer_dst() -> ignis::Result<bool> {
    let ctx = make_ctx()?;
    let capture = HandlerCapture::new();
    capture.install(&ctx);

    let gfx = ctx.queue(ignis::QueueType::Graphics)?;
    let pool = ctx.create_command_pool(ignis::QueueType::Graphics)?;

    // Deliberately omit TRANSFER_DST from usage.
    let img = ctx.create_image(&ignis::ImageInfo::texture_2d(
        32,
        32,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
    ))?;

    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;

    // Violation 1: transitioning into TRANSFER_DST_OPTIMAL without the
    // TRANSFER_DST_BIT usage flag.
    rec.pipeline_barrier(
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(img.handle())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })],
    );

    // Violation 2: the clear itself requires TRANSFER_DST_BIT usage.
    let clear = vk::ClearColorValue {
        float32: [1.0, 0.0, 0.0, 1.0],
    };
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    unsafe {
        ctx.device().cmd_clear_color_image(
            rec.raw_buffer(),
            img.handle(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &clear,
            std::slice::from_ref(&range),
        );
    }

    let cmd = rec.end()?;

    // Submit. The layer emits its callbacks either at record time or at
    // submit time depending on implementation; either way we want to reach
    // this point without panicking. We discard the submit result because
    // the command buffer is broken and any wait error is expected.
    let _ = gfx.submit_simple(cmd).and_then(|f| f.wait());

    capture.report();
    Ok(capture.fired())
}

/// Test 2: image with STORAGE usage, transition to COLOR_ATTACHMENT_OPTIMAL.
///
/// COLOR_ATTACHMENT_OPTIMAL layout requires COLOR_ATTACHMENT_BIT usage,
/// which this image does not have. This is a pure layout violation,
/// separate from the clear-related ones in test 1.
fn test_bad_layout_transition() -> ignis::Result<bool> {
    let ctx = make_ctx()?;
    let capture = HandlerCapture::new();
    capture.install(&ctx);

    let gfx = ctx.queue(ignis::QueueType::Graphics)?;
    let pool = ctx.create_command_pool(ignis::QueueType::Graphics)?;

    let img = ctx.create_image(&ignis::ImageInfo::texture_2d(
        32,
        32,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
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
    let _ = gfx.submit_simple(cmd).and_then(|f| f.wait());

    capture.report();
    Ok(capture.fired())
}

/// Test 3: copy 128 bytes out of a 64-byte buffer.
///
/// The layer catches copies that would read or write past the end of
/// either the source or destination buffer. This hits a different class
/// of VUIDs (offset/size family) than the image tests.
fn test_copy_out_of_bounds() -> ignis::Result<bool> {
    let ctx = make_ctx()?;
    let capture = HandlerCapture::new();
    capture.install(&ctx);

    let gfx = ctx.queue(ignis::QueueType::Graphics)?;
    let pool = ctx.create_command_pool(ignis::QueueType::Graphics)?;

    let src = ctx.create_buffer(&ignis::BufferInfo::staging(64))?;
    let dst = ctx.create_buffer(&ignis::BufferInfo {
        size: 64,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        location: ignis::MemoryLocation::GpuOnly,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
    })?;

    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;

    // Size 128 is twice the buffer size. The layer should flag the copy
    // as exceeding buffer bounds on both source and destination.
    rec.copy_buffer(
        src.handle(),
        dst.handle(),
        &[vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: 128,
        }],
    );

    let cmd = rec.end()?;
    let _ = gfx.submit_simple(cmd).and_then(|f| f.wait());

    capture.report();
    Ok(capture.fired())
}