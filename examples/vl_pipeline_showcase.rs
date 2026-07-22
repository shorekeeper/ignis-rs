//! Minimal VL pipeline showcase.
//!
//! Run with:
//! ```sh
//! cargo run --example vl_pipeline_showcase --features debug-tools
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("vl_pipeline_showcase requires --features debug-tools");

use ash::vk;
use ignis::{ignis_vl, vl_capture, vl_expect, vl_scope, vl_tag_scoped, vuid, Ignis, ManagedConfig};

fn main() -> ignis::Result<()> {
    let ctx = Ignis::managed(
        ManagedConfig::new("vl-showcase", vk::API_VERSION_1_2).enable_validation(true),
    )?;

    // 1. Global declarative config.
    ignis_vl! {
        suppress "VUID-*-99999";            // imaginary noise
        escalate category ObjectLifetime => error;
        action warning => log;
        action error => log;                // keep example non-fatal
        sink stderr;
        dedup per_vuid 3;
        backtrace errors_only;
    }

    // 2. Register an application VUID in the shared KB.
    vuid! {
        code: "APP-DEMO-001",
        title: "demo application rule",
        severity: Info,
        category: Other,
        what_happened: "this is a custom app-level diagnostic",
        why_rejected: "n/a",
        ignis_fix: "n/a; registered only to showcase vuid! macro",
        spec_section: "N/A",
    }

    // 3. Thread-local tag enriches every diagnostic in this scope.
    let _t = vl_tag_scoped!("demo_phase", "bad_copy");

    // 4. Capture diagnostics produced inside a block.
    let captured = vl_capture! {
        // errors inside are swallowed; validation diagnostics are what we want
        let _ = trigger_out_of_bounds_copy(&ctx);
    };
    println!("captured {} diagnostic(s)", captured.count());

    // 5. Local override: demote the same VUID inside this scope.
    {
        let _g = vl_scope! {
            demote "VUID-vkCmdCopyBuffer-size-00115" => info;
        };
        let _ = trigger_out_of_bounds_copy(&ctx);
    }

    // 6. Assert expectations.
    vl_expect! {
        rules: {
            at_least 1 vuid "VUID-vkCmdCopyBuffer-*";
            never vuid "VUID-*-99999";
        }
        in: {
            let _ = trigger_out_of_bounds_copy(&ctx);
        }
    }

    println!("done");
    Ok(())
}

/// Issues vkCmdCopyBuffer with size larger than source — reliably fires a VUID.
fn trigger_out_of_bounds_copy(ctx: &Ignis) -> ignis::Result<()> {
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
    rec.copy_buffer(
        src.handle(),
        dst.handle(),
        &[vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: 128, // intentional overflow
        }],
    );
    let cmd = rec.end()?;
    let _ = gfx.submit_simple(cmd).and_then(|f| f.wait());
    Ok(())
}