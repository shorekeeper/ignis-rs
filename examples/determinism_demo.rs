//! GPU determinism verifier demonstration.
//!
//! Two scenarios:
//!
//!  A) Deterministic compute: a fill_buffer followed by a copy. Output
//!     is bit-identical across all runs. The verifier reports success.
//!
//!  B) Non-deterministic CPU input: each run writes a fresh CPU-side
//!     pattern (depending on Instant::now() nanoseconds) into a staging
//!     buffer and copies it to the output buffer. The verifier panics on
//!     the second run with a structured diagnostic showing the hash
//!     mismatch.
//!
//! Note: actually triggering NON-deterministic GPU output (atomics
//! without ordering, etc) requires shipping a SPIR-V shader. To keep the
//! example dependency-free, scenario B fakes non-determinism by varying
//! CPU-side input between runs. The verifier itself does not care
//! whether the divergence comes from CPU or GPU; it just compares
//! captured outputs.
//!
//! Run with:
//! ```sh
//! cargo run --example determinism_demo --features debug-tools
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("determinism_demo requires --features debug-tools");

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ash::vk;
use ignis::{BufferInfo, Ignis, ManagedConfig, MemoryLocation, QueueType};

const TOTAL_STEPS: u32 = 5;

fn main() {
    println!();
    println!("    IGNIS DETERMINISM CHECKER DEMO");
    println!("    Verify GPU output is bit-identical across N runs.");
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
    // Step 1: context.
    step(1, "Create managed Ignis context");
    let ctx = Ignis::managed(
        ManagedConfig::new("determinism-demo", vk::API_VERSION_1_2)
            .enable_validation(false),
    )?;
    let dev_name = unsafe {
        std::ffi::CStr::from_ptr(ctx.device_properties().device_name.as_ptr())
    }
    .to_str()
    .unwrap_or("?");
    info(&format!("device: {dev_name}"));
    ok();

    // Step 2: deterministic scenario.
    step(2, "Scenario A: deterministic fill_buffer (4 KiB)");
    let det_a = ctx.create_determinism_checker(QueueType::Graphics)?;
    let dst_a = ctx.create_buffer(&BufferInfo {
        size: 4096,
        usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        location: MemoryLocation::GpuOnly,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
    })?;
    let dst_handle_a = dst_a.handle();
    let dst_size_a = dst_a.size();

    let res = det_a.run_with_seed(0x42, move |rec, _frame, captures| {
        rec.fill_buffer(dst_handle_a, 0, dst_size_a, 0xCAFE_BABE);
        captures.add_buffer("output", dst_handle_a, 0, dst_size_a);
        Ok(())
    })?;
    info(&format!(
        "first run hash: {:#018x} ({} bytes)",
        res.buffer_hashes[0].hash, res.buffer_hashes[0].size
    ));
    info("running 9 more times and verifying all 10 are identical...");
    det_a.verify_n_runs(10)?;
    info(&format!(
        "verified {} runs, all hashes identical",
        det_a.run_count()
    ));
    ok();

    // Step 3: print the hash chain to prove they all match.
    step(3, "Print hash chain for scenario A");
    for (i, r) in det_a.results().iter().enumerate() {
        info(&format!(
            "run {:>2}: hash = {:#018x}",
            i, r.buffer_hashes[0].hash
        ));
    }
    ok();

    // Step 4: non-deterministic scenario (fakes non-determinism via CPU
    // input). We catch the panic to keep the example running.
    step(4, "Scenario B: non-deterministic CPU input");
    let det_b = ctx.create_determinism_checker(QueueType::Graphics)?;

    // Staging buffer + GPU buffer pair. Each run, we re-write staging
    // with a counter that depends on a side-channel atomic so the
    // closure produces different output every invocation.
    let staging_b = ctx.create_buffer(&BufferInfo::staging(256))?;
    let gpu_b = ctx.create_buffer(&BufferInfo {
        size: 256,
        usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        location: MemoryLocation::GpuOnly,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
    })?;
    let staging_handle = staging_b.handle();
    let staging_ptr = staging_b
        .mapped_ptr()
        .expect("staging is host-visible");
    let gpu_handle = gpu_b.handle();

    // Closures stored by DeterminismChecker must be Send + Sync, so we
    // cannot capture *mut u8 directly. Cast to usize and back inside
    // the closure body. The pointer remains valid because staging_b
    // outlives the checker.
    let staging_ptr_addr = staging_ptr as usize;

    let counter = Arc::new(AtomicU32::new(0));
    let counter_for_closure = Arc::clone(&counter);

    info(&format!(
        "buffer pair: staging={:?} gpu={:?}",
        staging_handle, gpu_handle
    ));

    // Run once to capture the baseline.
    let baseline = det_b.run_with_seed(0xDEAD_BEEF, move |rec, _frame, captures| {
        // Each invocation writes a different byte pattern: byte 0 = run
        // counter. Subsequent runs see a new value, producing a
        // different hash.
        let v = counter_for_closure.fetch_add(1, Ordering::Relaxed) as u8;
        let staging_ptr = staging_ptr_addr as *mut u8;
        unsafe {
            std::ptr::write_bytes(staging_ptr, v, 256);
        }
        rec.copy_buffer(
            staging_handle,
            gpu_handle,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 256,
            }],
        );
        captures.add_buffer("output", gpu_handle, 0, 256);
        Ok(())
    })?;
    info(&format!(
        "baseline run 0: byte=0x00 hash={:#018x}",
        baseline.buffer_hashes[0].hash
    ));

    // verify_n_runs(2) will execute one more run, see that the hash
    // changed, and panic. We catch the panic so the example continues.
    info("attempting verify_n_runs(2) - this is expected to panic");
    println!();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        det_b.verify_n_runs(2)
    }));
    println!();
    match result {
        Ok(_) => {
            info("(unexpected) verifier did not panic; closure is deterministic after all?");
        }
        Err(_) => {
            info("verifier panicked as expected");
            info("see the IGN-DET diagnostic above for the hash mismatch report");
        }
    }
    ok();

    // Step 5: print all results including the divergent one.
    step(5, "Hash chain for scenario B");
    for (i, r) in det_b.results().iter().enumerate() {
        info(&format!(
            "run {:>2}: hash = {:#018x}",
            i, r.buffer_hashes[0].hash
        ));
    }
    info("scenario B's hash changes per run -> non-deterministic");
    info("scenario A's hash is stable -> deterministic");
    ok();

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