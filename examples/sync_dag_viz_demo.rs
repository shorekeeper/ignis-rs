//! Sync DAG visualizer demonstration.
//!
//! Builds three dependency graphs (clean diamond, orphans, cycle) and
//! renders each in DOT, Mermaid, and SVG. SVG can be opened directly in
//! a browser; DOT requires `dot -Tsvg`; Mermaid renders inline on GitHub.
//!
//! Output files:
//!   sync_dag_diamond.dot, .mmd, .svg
//!   sync_dag_orphans.dot, .mmd, .svg
//!   sync_dag_cycle.dot,   .mmd, .svg
//!
//! Run with:
//! ```sh
//! cargo run --example sync_dag_viz_demo --features debug-tools
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("sync_dag_viz_demo requires --features debug-tools");

use ash::vk;
use ignis::{CrossQueueTracker, Ignis, ManagedConfig, SyncDagVisualizer};

const TOTAL_STEPS: u32 = 6;

fn main() {
    println!();
    println!("    IGNIS SYNC DAG VISUALIZER DEMO");
    println!("    Render cross-queue dependency graphs as DOT, Mermaid, and SVG.");
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
    step(1, "Create context (synthetic data only)");
    let _ctx = Ignis::managed(
        ManagedConfig::new("sync-dag-demo", vk::API_VERSION_1_2).enable_validation(false),
    )?;
    info("the visualizer is stateless; no GPU work is performed");
    ok();

    let viz = SyncDagVisualizer::new();

    // Scenario 1: diamond.
    step(2, "Render diamond pattern");
    let t1 = build_diamond();
    let r1 = t1.analyze();
    let s1 = t1.snapshot();
    save_all(&viz, &r1, &s1, "sync_dag_diamond")?;
    info("submissions: 4, cross-queue edges: 2, cycles: 0");
    ok();

    // Scenario 2: orphans.
    step(3, "Render graph with orphans");
    let t2 = build_orphans();
    let r2 = t2.analyze();
    let s2 = t2.snapshot();
    save_all(&viz, &r2, &s2, "sync_dag_orphans")?;
    info(&format!(
        "orphan signals: {}, orphan waits: {}",
        r2.orphan_signals.len(),
        r2.orphan_waits.len()
    ));
    ok();

    // Scenario 3: cycle.
    step(4, "Render graph with cycle");
    let t3 = build_cycle();
    let r3 = t3.analyze();
    let s3 = t3.snapshot();
    save_all(&viz, &r3, &s3, "sync_dag_cycle")?;
    info(&format!("cycles detected: {}", r3.cycles.len()));
    info("open sync_dag_cycle.svg in a browser to see the cycle in red");
    ok();

    // Step 5: combined HTML index. Solves the "many separate SVGs is
    // painful to navigate" problem.
    step(5, "Render combined HTML index");
    let graphs = [
        ("Diamond (clean)", &r1, s1.as_slice()),
        ("Orphans (warning)", &r2, s2.as_slice()),
        ("Cycle (deadlock)", &r3, s3.as_slice()),
    ];
    let html_path = "sync_dag_index.html";
    let bytes = viz
        .save_html_index("Ignis Sync DAG Demo", &graphs, html_path)
        .map_err(|_| ignis::Error::InvalidConfig("failed to write HTML index"))?;
    info(&format!("wrote {} ({} bytes)", html_path, bytes));
    info("open sync_dag_index.html in any browser to see all graphs on one page");
    info("each graph has its own section with stats badge and responsive SVG");
    ok();

    // Step 6: combined BMP. Single image with all graphs stacked
    // vertically, each with its own title, badge, and stats. Open in
    // any OS image viewer (no browser needed).
    step(6, "Render combined BMP (single scrollable image)");
    let combined_path = "sync_dag_combined.bmp";
    let bytes = viz
        .save_bmp_combined("Ignis Sync DAG Demo", &graphs, combined_path)
        .map_err(|_| ignis::Error::InvalidConfig("failed to write combined BMP"))?;
    info(&format!("wrote {} ({} bytes)", combined_path, bytes));
    info("open in Photos / Preview / xdg-open");
    info("contains all 3 graphs: diamond (OK), orphans (warning), cycle (deadlock)");
    ok();

    Ok(())
}

fn save_all(
    viz: &SyncDagVisualizer,
    report: &ignis::CrossQueueReport,
    subs: &[ignis::TrackedSubmission],
    base: &str,
) -> ignis::Result<()> {
    for ext in &["dot", "mmd", "svg", "bmp"] {
        let path = format!("{}.{}", base, ext);
        let bytes = viz
            .save(report, subs, &path)
            .map_err(|_| ignis::Error::InvalidConfig("failed to write visualizer output"))?;
        info(&format!("wrote {} ({} bytes)", path, bytes));
    }
    Ok(())
}

fn build_diamond() -> CrossQueueTracker {
    let t = CrossQueueTracker::new();
    t.record_raw(0, 0, "geometry", &[], &[], &[1, 2], 0);
    t.record_raw(1, 0, "shadow_a", &[], &[1], &[3], 0);
    t.record_raw(1, 0, "shadow_b", &[], &[2], &[4], 0);
    t.record_raw(0, 0, "compose", &[], &[3, 4], &[], 0);
    t
}

fn build_orphans() -> CrossQueueTracker {
    let t = CrossQueueTracker::new();
    t.record_raw(0, 0, "main_pass", &[], &[], &[1], 0);
    t.record_raw(1, 0, "consumer", &[], &[1], &[], 0);
    t.record_raw(0, 0, "leaks_signal", &[], &[], &[99], 0);
    t.record_raw(1, 0, "lonely_wait", &[], &[200], &[], 0);
    t
}

fn build_cycle() -> CrossQueueTracker {
    let t = CrossQueueTracker::new();
    t.record_raw(0, 0, "A", &[], &[2], &[1], 0);
    t.record_raw(1, 0, "B", &[], &[1], &[2], 0);
    t
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