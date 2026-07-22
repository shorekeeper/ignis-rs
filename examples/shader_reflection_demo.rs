//! SPIR-V reflection + filesystem hot-reload watcher demonstration.
//!
//! Steps performed by the example:
//!
//!  1. Reflect three pre-assembled SPIR-V modules (compute, vertex,
//!     fragment) using the same fixtures as the smoke tests. Show entry
//!     points, stages, and (for compute) workgroup local size.
//!  2. Hand-assemble a fixture with one uniform-buffer descriptor and
//!     show how reflection produces structured DescriptorBinding entries
//!     including set, binding, type, count, stage, and source name.
//!  3. Use `descriptor_set_layouts_from` to turn the reflection into
//!     ready-to-use VkDescriptorSetLayoutBinding arrays grouped by set
//!     index.
//!  4. Drive the ShaderWatcher: write a file, register a watch, modify
//!     the file twice, and confirm the callback fires for each change.
//!     The callback re-parses the bytes through `bytes_to_spirv` and
//!     `reflect`, demonstrating the full hot-reload cycle.
//!
//! No external Vulkan compiler is required; all SPIR-V is hand-assembled.
//!
//! Run with:
//! ```sh
//! cargo run --example shader_reflection_demo --features debug-tools
//! ```

#[cfg(not(feature = "debug-tools"))]
compile_error!("shader_reflection_demo requires --features debug-tools");

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ash::vk;
use ignis::{
    bytes_to_spirv, descriptor_set_layouts_from, reflect, ShaderReflection, ShaderWatcher,
};

const TOTAL_STEPS: u32 = 6;

// SPIR-V fixtures lifted from the smoke test. void main() {} compute,
// minimal vertex shader writing gl_Position, minimal fragment shader
// writing constant red.

#[rustfmt::skip]
const EMPTY_COMPUTE_SPV: &[u32] = &[
    0x07230203, 0x00010000, 0x00000000, 0x00000006, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000E, 0x00000000, 0x00000001,
    0x0005000F, 0x00000005, 0x00000004, 0x6E69616D, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001,
    0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200F8, 0x00000005,
    0x000100FD,
    0x00010038,
];

#[rustfmt::skip]
const MINIMAL_VERT_SPV: &[u32] = &[
    0x07230203, 0x00010000, 0x00000000, 0x00000011, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000E, 0x00000000, 0x00000001,
    0x0006000F, 0x00000000, 0x00000003, 0x6E69616D, 0x00000000, 0x00000008,
    0x00050048, 0x00000006, 0x00000000, 0x0000000B, 0x00000000,
    0x00030047, 0x00000006, 0x00000002,
    0x00020013, 0x00000001,
    0x00030021, 0x00000002, 0x00000001,
    0x00030016, 0x00000004, 0x00000020,
    0x00040017, 0x00000005, 0x00000004, 0x00000004,
    0x0003001E, 0x00000006, 0x00000005,
    0x00040020, 0x00000007, 0x00000003, 0x00000006,
    0x00040015, 0x0000000C, 0x00000020, 0x00000000,
    0x00040020, 0x0000000E, 0x00000003, 0x00000005,
    0x0004003B, 0x00000007, 0x00000008, 0x00000003,
    0x0004002B, 0x00000004, 0x00000009, 0x00000000,
    0x0004002B, 0x00000004, 0x0000000A, 0x3F800000,
    0x0007002C, 0x00000005, 0x0000000B, 0x00000009, 0x00000009, 0x00000009, 0x0000000A,
    0x0004002B, 0x0000000C, 0x0000000D, 0x00000000,
    0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
    0x000200F8, 0x0000000F,
    0x00050041, 0x0000000E, 0x00000010, 0x00000008, 0x0000000D,
    0x0003003E, 0x00000010, 0x0000000B,
    0x000100FD,
    0x00010038,
];

#[rustfmt::skip]
const MINIMAL_FRAG_SPV: &[u32] = &[
    0x07230203, 0x00010000, 0x00000000, 0x0000000C, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000E, 0x00000000, 0x00000001,
    0x0006000F, 0x00000004, 0x00000003, 0x6E69616D, 0x00000000, 0x00000007,
    0x00030010, 0x00000003, 0x00000007,
    0x00040047, 0x00000007, 0x0000001E, 0x00000000,
    0x00020013, 0x00000001,
    0x00030021, 0x00000002, 0x00000001,
    0x00030016, 0x00000004, 0x00000020,
    0x00040017, 0x00000005, 0x00000004, 0x00000004,
    0x00040020, 0x00000006, 0x00000003, 0x00000005,
    0x0004003B, 0x00000006, 0x00000007, 0x00000003,
    0x0004002B, 0x00000004, 0x00000008, 0x3F800000,
    0x0004002B, 0x00000004, 0x00000009, 0x00000000,
    0x0007002C, 0x00000005, 0x0000000A, 0x00000008, 0x00000009, 0x00000009, 0x00000008,
    0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
    0x000200F8, 0x0000000B,
    0x0003003E, 0x00000007, 0x0000000A,
    0x000100FD,
    0x00010038,
];

fn main() {
    println!();
    println!("    IGNIS SHADER REFLECTION + HOT-RELOAD DEMO");
    println!("    Pure-Rust SPIR-V reflection (no shaderc/spirv-cross) and");
    println!("    filesystem-polled shader hot-reload (no notify crate).");
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
    // Step 1: reflect the compute fixture.
    step(1, "Reflect compute shader (void main, local_size 1,1,1)");
    let r = reflect(EMPTY_COMPUTE_SPV)?;
    print_reflection(&r);
    assert_eq!(r.entry_points.len(), 1);
    assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::COMPUTE);
    assert_eq!(r.local_size, Some([1, 1, 1]));
    ok();

    // Step 2: reflect the vertex fixture.
    step(2, "Reflect vertex shader (writes gl_Position)");
    let r = reflect(MINIMAL_VERT_SPV)?;
    print_reflection(&r);
    assert_eq!(r.entry_points.len(), 1);
    assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::VERTEX);
    assert!(r.local_size.is_none());
    ok();

    // Step 3: reflect the fragment fixture.
    step(3, "Reflect fragment shader (constant red color)");
    let r = reflect(MINIMAL_FRAG_SPV)?;
    print_reflection(&r);
    assert_eq!(r.entry_points.len(), 1);
    assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::FRAGMENT);
    ok();

    // Step 4: reflect a fixture with a uniform buffer descriptor and
    // build VkDescriptorSetLayoutBinding arrays from the result.
    step(4, "Reflect UBO fixture and build VkDescriptorSetLayoutBindings");
    let ubo_spv = build_ubo_fixture();
    let r = reflect(&ubo_spv)?;
    print_reflection(&r);
    assert_eq!(r.descriptor_bindings.len(), 1);
    let b = &r.descriptor_bindings[0];
    info(&format!(
        "binding details: set={} binding={} type={:?} count={} stage={:?} name={:?}",
        b.set, b.binding, b.descriptor_type, b.count, b.stage, b.name
    ));
    assert_eq!(b.set, 0);
    assert_eq!(b.binding, 0);
    assert_eq!(b.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);

    let layouts = descriptor_set_layouts_from(&r);
    info(&format!(
        "descriptor_set_layouts_from grouped result into {} set(s)",
        layouts.len()
    ));
    for (set_idx, bindings) in &layouts {
        info(&format!("  set {set_idx}: {} binding(s)", bindings.len()));
        for binding in bindings {
            info(&format!(
                "    binding={} type={:?} count={}",
                binding.binding, binding.descriptor_type, binding.descriptor_count
            ));
        }
    }
    ok();

    // Step 5: hot-reload demonstration.
    step(5, "ShaderWatcher: write file, modify, observe callbacks");
    let path = std::env::temp_dir().join(format!("ignis_reload_demo_{}.spv", std::process::id()));
    write_spv(&path, EMPTY_COMPUTE_SPV)?;
    info(&format!("wrote initial SPV to {}", path.display()));

    let watcher = ShaderWatcher::new(Duration::from_millis(150));
    let count = Arc::new(AtomicU32::new(0));
    let last_stage: Arc<std::sync::Mutex<String>> =
        Arc::new(std::sync::Mutex::new("none".to_string()));

    let count_cb = Arc::clone(&count);
    let stage_cb = Arc::clone(&last_stage);
    watcher.watch(&path, move |bytes| {
        let n = count_cb.fetch_add(1, Ordering::SeqCst) + 1;
        // Re-parse the new bytes to demonstrate the full reload cycle.
        match bytes_to_spirv(bytes) {
            Ok(words) => match reflect(&words) {
                Ok(r) => {
                    let stage = r
                        .entry_points
                        .first()
                        .map(|e| format!("{:?}", e.stage))
                        .unwrap_or_else(|| "no entry points".to_string());
                    eprintln!(
                        "       [callback {n}] {} bytes, stage = {stage}",
                        bytes.len()
                    );
                    *stage_cb.lock().unwrap() = stage;
                }
                Err(e) => eprintln!("       [callback {n}] reflect error: {e}"),
            },
            Err(e) => eprintln!("       [callback {n}] bytes_to_spirv error: {e}"),
        }
    });
    info("watcher registered, sleeping for filesystem mtime granularity...");

    // Many filesystems have second-level mtime resolution. Wait long
    // enough to ensure the next write produces a distinguishable mtime.
    std::thread::sleep(Duration::from_millis(1100));
    write_spv(&path, MINIMAL_VERT_SPV)?;
    info("rewrote file with vertex shader bytes");

    std::thread::sleep(Duration::from_millis(1100));
    write_spv(&path, MINIMAL_FRAG_SPV)?;
    info("rewrote file with fragment shader bytes");

    // Wait for the polling thread to catch up. Poll interval is 150ms,
    // so 800ms of slack covers any scheduling jitter.
    std::thread::sleep(Duration::from_millis(800));

    let total = count.load(Ordering::SeqCst);
    info(&format!("total callbacks fired: {total}"));
    info(&format!(
        "last reflected stage: {}",
        last_stage.lock().unwrap()
    ));
    assert!(
        total >= 2,
        "expected at least 2 callbacks for 2 file modifications"
    );

    drop(watcher);
    let _ = std::fs::remove_file(&path);
    ok();

    // Step 6: summary.
    step(6, "Summary");
    info("reflection: extracts entry points, descriptors, push constants,");
    info("            vertex inputs, spec constants, and compute local size");
    info("watcher:    polls the filesystem and re-fires user callback on change");
    info("together:   complete shader hot-reload cycle without spirv-cross");
    info("            or any third-party file watcher dependency");
    ok();

    Ok(())
}

/// Print a reflection result in a compact form.
fn print_reflection(r: &ShaderReflection) {
    info(&format!("entry points: {}", r.entry_points.len()));
    for ep in &r.entry_points {
        info(&format!("  '{}' -> {:?}", ep.name, ep.stage));
    }
    if let Some(ls) = r.local_size {
        info(&format!("compute local size: [{}, {}, {}]", ls[0], ls[1], ls[2]));
    }
    if !r.descriptor_bindings.is_empty() {
        info(&format!(
            "descriptor bindings: {}",
            r.descriptor_bindings.len()
        ));
    }
    if !r.push_constant_ranges.is_empty() {
        info(&format!(
            "push constant ranges: {}",
            r.push_constant_ranges.len()
        ));
    }
    if !r.vertex_inputs.is_empty() {
        info(&format!("vertex inputs: {}", r.vertex_inputs.len()));
    }
    if !r.spec_constants.is_empty() {
        info(&format!(
            "specialization constants: {}",
            r.spec_constants.len()
        ));
    }
}

/// Serialize SPIR-V words (little-endian) and write them to disk.
fn write_spv(path: &std::path::Path, words: &[u32]) -> ignis::Result<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(path, &bytes).map_err(|_| ignis::Error::InvalidConfig("failed to write SPV"))
}

/// Hand-assembled fragment shader with one UBO at set=0 binding=0 named "ubo".
/// This is the same fixture used by `shader_reflection.rs` unit tests.
fn build_ubo_fixture() -> Vec<u32> {
    let mut s: Vec<u32> = Vec::with_capacity(64);
    s.extend_from_slice(&[0x07230203, 0x00010000, 0, 32, 0]);
    s.extend_from_slice(&[(2 << 16) | 17, 1]);
    s.extend_from_slice(&[(3 << 16) | 14, 0, 1]);
    s.extend_from_slice(&[(5 << 16) | 15, 4, 4, 0x6E69616D, 0x00000000]);
    s.extend_from_slice(&[(3 << 16) | 16, 4, 7]);
    s.extend_from_slice(&[(4 << 16) | 5, 4, 0x6E69616D, 0x00000000]);
    s.extend_from_slice(&[(3 << 16) | 5, 7, 0x004F4255]);
    s.extend_from_slice(&[(3 << 16) | 5, 9, 0x006F6275]);
    s.extend_from_slice(&[(3 << 16) | 71, 7, 2]);
    s.extend_from_slice(&[(5 << 16) | 72, 7, 0, 35, 0]);
    s.extend_from_slice(&[(4 << 16) | 71, 9, 34, 0]);
    s.extend_from_slice(&[(4 << 16) | 71, 9, 33, 0]);
    s.extend_from_slice(&[(2 << 16) | 19, 2]);
    s.extend_from_slice(&[(3 << 16) | 33, 3, 2]);
    s.extend_from_slice(&[(3 << 16) | 22, 5, 32]);
    s.extend_from_slice(&[(4 << 16) | 23, 6, 5, 4]);
    s.extend_from_slice(&[(3 << 16) | 30, 7, 6]);
    s.extend_from_slice(&[(4 << 16) | 32, 8, 2, 7]);
    s.extend_from_slice(&[(4 << 16) | 59, 8, 9, 2]);
    s.extend_from_slice(&[(5 << 16) | 54, 2, 4, 0, 3]);
    s.extend_from_slice(&[(2 << 16) | 248, 10]);
    s.extend_from_slice(&[(1 << 16) | 253]);
    s.extend_from_slice(&[(1 << 16) | 56]);
    s
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