//! Cross-queue tracker demonstration.
//!
//! Builds three synthetic dependency graphs without ever submitting them
//! to a real queue (so we can construct deliberately broken scenarios
//! that would deadlock real hardware), records them into the tracker,
//! and prints the analysis report for each.
//!
//! Scenarios:
//!  1. Clean diamond. Two cross-queue edges, no issues.
//!  2. Orphans. One submission signals a semaphore nobody waits on,
//!     another waits on a semaphore nobody signals.
//!  3. Cycle. Two submissions wait on each other's signal -> deadlock.
//!
//! Run with:
//! ```sh
//! cargo run --example cross_queue_demo --features debug-tools
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("cross_queue_demo requires --features debug-tools");

use ash::vk;
use ignis::{Ignis, ManagedConfig};

const TOTAL_STEPS: u32 = 4;

fn main() {
    println!();
    println!("    IGNIS CROSS-QUEUE TRACKER DEMO");
    println!("    Detect orphan signals, orphan waits, and cycles in queue submission graphs.");
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
    step(1, "Create context (no validation, no real queue submits)");
    let ctx = Ignis::managed(
        ManagedConfig::new("cross-queue-demo", vk::API_VERSION_1_2).enable_validation(false),
    )?;
    let dev_name = unsafe {
        std::ffi::CStr::from_ptr(ctx.device_properties().device_name.as_ptr())
    }
    .to_str()
    .unwrap_or("?");
    info(&format!("device: {dev_name}"));
    info("note: this demo records synthetic submissions without submitting them.");
    info("real Vulkan would deadlock on scenario 3.");
    ok();

    // Scenario 1: clean diamond.
    step(2, "Scenario 1: clean diamond pattern");
    {
        let tracker = ctx.create_cross_queue_tracker();
        // Q0 = graphics, Q1 = compute. A on Q0 signals 1+2.
        // B and C on Q1 each take one branch, signal 3+4.
        // D on Q0 collects both.
        tracker.record_raw(0, 0, "geometry", &[], &[], &[1, 2], 0);
        tracker.record_raw(1, 0, "shadow_a", &[], &[1], &[3], 0);
        tracker.record_raw(1, 0, "shadow_b", &[], &[2], &[4], 0);
        tracker.record_raw(0, 0, "compose", &[], &[3, 4], &[], 0);

        let report = tracker.analyze();
        info(&format!(
            "submissions={} queues={} cross_queue_edges={} cycles={} orphans={}",
            report.submission_count,
            report.queue_count,
            report.cross_queue_edges.len(),
            report.cycles.len(),
            report.orphan_signals.len() + report.orphan_waits.len()
        ));
        println!();
        eprint!("{report}");
        println!();
        assert!(!report.has_issues(), "diamond should be clean");
        info("diamond pattern verified clean ✓");
    }
    ok();

    // Scenario 2: orphans.
    step(3, "Scenario 2: orphan signal + orphan wait");
    {
        let tracker = ctx.create_cross_queue_tracker();
        tracker.record_raw(0, 0, "main_pass", &[], &[], &[1], 0);
        tracker.record_raw(1, 0, "consumer", &[], &[1], &[], 0);

        // Orphan signal: post_pass signals 99, nobody waits.
        tracker.record_raw(0, 0, "post_pass", &[], &[], &[99], 0);

        // Orphan wait: lonely_consumer waits on 200, nobody signals.
        tracker.record_raw(1, 0, "lonely_consumer", &[], &[200], &[], 0);

        let report = tracker.analyze();
        info(&format!(
            "orphan_signals={} orphan_waits={}",
            report.orphan_signals.len(),
            report.orphan_waits.len()
        ));
        println!();
        eprint!("{report}");
        println!();
        assert_eq!(report.orphan_signals.len(), 1);
        assert_eq!(report.orphan_waits.len(), 1);
        assert!(!report.has_cycles());
        info("orphans correctly detected ✓");
    }
    ok();

    // Scenario 3: cycle (deadlock).
    step(4, "Scenario 3: cycle (would deadlock)");
    {
        let tracker = ctx.create_cross_queue_tracker();
        // A waits on sem_b, signals sem_a.
        // B waits on sem_a, signals sem_b.
        // Each waits on the other -> classic deadlock.
        tracker.record_raw(0, 0, "graphics_A", &[], &[2], &[1], 0);
        tracker.record_raw(1, 0, "compute_B", &[], &[1], &[2], 0);

        let report = tracker.analyze();
        info(&format!(
            "cycles={} chain_length={}",
            report.cycles.len(),
            report.longest_chain.len()
        ));
        println!();
        eprint!("{report}");
        println!();
        assert!(report.has_cycles(), "cycle should be detected");
        if report.has_cycles() {
            info("cycle correctly detected ✓");
            info("(in real Vulkan this would deadlock at vkQueueSubmit)");
        }
    }
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