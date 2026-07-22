//! Comprehensive ignis -> ignis-viz live link demo.
//!
//! Generates a busy synthetic workload that exercises every viewer view
//! and feature surface, including replay scrubber data depth, snapshot
//! save/load, all severity filters, and stress patterns.
//!
//!   - Frame Graph: 16+ core nodes plus optional SSAO and bloom
//!     subgraphs that toggle on a 15s/22s schedule. Long submit->pass
//!     edges create routing-only dummy layers, so enabling layer
//!     compaction in Settings visibly shrinks the graph horizontally
//!     with a 350ms ease-in-out transition. Periodic node label
//!     updates demonstrate the re-registration path.
//!   - Sync DAG: submissions on three queue families with nontrivial
//!     durations so the hover-correlation tooltip shows passes/allocs/
//!     frees that fall inside each submission's time window.
//!     Cycle and orphan marks tint queue lanes red and yellow on a
//!     periodic schedule so the user can see the marks appear and
//!     auto-clear without restarting the demo.
//!   - Memory: four named device memory blocks with churning transient
//!     allocations. Periodic resource renaming exercises the name
//!     resolver update path (handle keeps the same id, name string
//!     changes; viewer must reflect this in tooltips).
//!   - Pass timeline: variable pass durations per frame, with golden
//!     frame boundary lines inferred from submission cadence on the
//!     graphics queue (visible after ~3 graphics submissions).
//!   - GPU Timeline: per-pass GPU timestamps emitted with proper
//!     `VkPipelineStageFlags` so the view shows multiple lanes split
//!     across three queues. Per-queue colours match Sync DAG and Pass
//!     Profiler so a queue is visually identifiable across panels.
//!   - Validation: rolling stream of synthetic VUID diagnostics with
//!     realistic severity mix (mostly warnings, occasional errors,
//!     periodic info notes). The Info filter chip is fed independently
//!     so testing severity filtering works at any point. Each
//!     diagnostic is tagged with the graph node id it relates to so
//!     the Frame Graph and Subgraph views display a red `!` badge on
//!     affected nodes for ~15 seconds. Errors trigger auto-pin and
//!     auto-jump if those settings are on.
//!   - Pipeline Stats: per-pass GPU counter readbacks emitted every
//!     ~40 frames simulating real readback latency. Counters scale
//!     with workload (more vertices when SSAO is on, etc.).
//!   - Budget: per-heap usage and driver budget snapshots; staging
//!     heap utilization tracks the transient alloc count so the
//!     viewer's bar swings visibly. Budget pressure elevated during
//!     burst mode for visibility testing.
//!   - Canary Monitor: simulated `HardenedAllocator` corruption events
//!     every 45 seconds (alternating front/back guard, mostly
//!     warnings with a periodic error, occasional severe corruption
//!     of a large block). Aggregate stats snapshots stream every 3
//!     seconds so the top-card counters track. Click an event for
//!     the hex diff with `^^` markers.
//!   - Determinism: alternating session profiles. Every 90 seconds a
//!     "clean" 5-run "physics_smoke_test" session completes with
//!     zero divergences (all rows green). Every 120 seconds a
//!     "render_smoke_test" session has run #3 diverge on the
//!     shadow_atlas image; click the row for detail with an "Open
//!     diff bitmap" button (the bitmap path is synthetic; opening it
//!     returns a "file not found" from the OS, which is expected).
//!   - Names: every memory block, buffer, image, and sampler is
//!     registered with a debug name so the viewer can substitute it
//!     everywhere instead of showing raw hex handles. A subset is
//!     periodically renamed to test the resolver update path.
//!   - Burst Mode: every 90 seconds the simulator enters a 12-second
//!     burst where event rates triple. Useful for stress-testing the
//!     viewer's IPC drain path and verifying lossy-ring behavior.
//!
//! Tabs and replay UI features are exercised by the viewer regardless
//! of what the producer emits; the console hint section below points
//! the user at the relevant interactions.
//!
//! Usage:
//!     ignis-rs (terminal A): cargo run --example live_link_demo --features live-link
//!     ignis-viz (terminal B): cargo run -- --ipc ignis_demo --no-sim

#[cfg(not(feature = "live-link"))]
compile_error!("live_link_demo requires --features live-link");

use std::sync::Arc;
use std::time::{Duration, Instant};

use ignis::live_link::{
    LiveLink, NODE_KIND_PASS, NODE_KIND_RESOURCE, NODE_KIND_SUBMIT,
    RES_KIND_BUFFER, RES_KIND_DEVICE_MEMORY, RES_KIND_IMAGE, RES_KIND_SAMPLER,
    VAL_SEVERITY_ERROR, VAL_SEVERITY_INFO, VAL_SEVERITY_WARNING,
    PIPELINE_ISSUE_DESCRIPTOR_COUNT, PIPELINE_ISSUE_LAYOUT_COMPATIBILITY,
    PIPELINE_ISSUE_PUSH_CONSTANT_RANGE, PIPELINE_ISSUE_STAGE_INTERFACE,
};

const RING_NAME: &str = "ignis_demo";
const RING_CAPACITY: u32 = 8192;

// ---- VkPipelineStageFlags constants for GPU timestamp lanes ------------

const STAGE_VERTEX_SHADER: u32      = 0x0000_0008;
const STAGE_FRAGMENT_SHADER: u32    = 0x0000_0080;
const STAGE_COLOR_ATTACHMENT: u32   = 0x0000_0400;
const STAGE_COMPUTE_SHADER: u32     = 0x0000_0800;
const STAGE_TRANSFER: u32           = 0x0000_1000;

// ---- Sync severity constants (mirror viewer-side ipc.rs) ---------------

const SYNC_SEVERITY_ORPHAN: u32 = 1;
const SYNC_SEVERITY_CYCLE: u32 = 2;
// Shader printf coarse stage codes (mirror viewer-side ipc).
const PRINTF_STAGE_VS: u32 = 1;
const PRINTF_STAGE_FS: u32 = 2;
const PRINTF_STAGE_CS: u32 = 3;
const PRINTF_STAGE_RGEN: u32 = 4;

// Vulkan object types used by lifetime registrations.
const VK_TYPE_BUFFER: u32 = 9;
const VK_TYPE_IMAGE: u32 = 10;
const VK_TYPE_PIPELINE: u32 = 19;
const VK_TYPE_SAMPLER: u32 = 21;
const VK_TYPE_DESCRIPTOR_SET: u32 = 23;

// Aliasing access types.
const ACCESS_READ: u32 = 0;
const ACCESS_WRITE: u32 = 1;

// Resource kind tags used in descriptor issues.
const RES_KIND_OTHER: u32 = 255;

fn main() {
    println!("ignis live link demo (v0.14: full feature coverage)");
    println!();

    let link = match LiveLink::create(RING_NAME, RING_CAPACITY) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to create live link: {e}");
            std::process::exit(1);
        }
    };

    println!("shared memory ring '{}' is ready (capacity {}).",
        RING_NAME, RING_CAPACITY);
    println!("run 'ignis-viz --ipc {} --no-sim' in another terminal.", RING_NAME);
    println!();
    print_console_hints();

    setup_initial_graph(&link);
    let mem = setup_initial_allocations(&link);
    register_resource_names(&link, &mem);
    register_permanent_objects(&link);
    run_event_loop(&link, &mem);
}

fn print_console_hints() {

    println!("emitting events. Ctrl+C to stop.");
    println!();
}

// ---- Static topology ---------------------------------------------------

const CORE_NODES: &[(u32, u32, &str)] = &[
    (1,  NODE_KIND_PASS,     "vertex_skin"),
    (2,  NODE_KIND_PASS,     "shadow_cascade"),
    (3,  NODE_KIND_PASS,     "geometry_main"),
    (4,  NODE_KIND_RESOURCE, "skinned_vertex_buffer"),
    (5,  NODE_KIND_RESOURCE, "shadow_atlas"),
    (6,  NODE_KIND_RESOURCE, "gbuffer_albedo"),
    (7,  NODE_KIND_RESOURCE, "gbuffer_normal"),
    (8,  NODE_KIND_RESOURCE, "gbuffer_depth"),
    (9,  NODE_KIND_PASS,     "lighting_deferred"),
    (10, NODE_KIND_RESOURCE, "hdr_color"),
    (11, NODE_KIND_PASS,     "tonemap"),
    (12, NODE_KIND_RESOURCE, "ldr_color"),
    (13, NODE_KIND_PASS,     "ui_compose"),
    (14, NODE_KIND_RESOURCE, "swapchain"),
    (50, NODE_KIND_SUBMIT,   "graphics_queue"),
    (51, NODE_KIND_SUBMIT,   "compute_queue"),
    (52, NODE_KIND_SUBMIT,   "transfer_queue"),
];

const CORE_EDGES: &[(u32, u32)] = &[
    (1, 4),
    (2, 5),
    (4, 3), (3, 6), (3, 7), (3, 8),
    (5, 9), (6, 9), (7, 9), (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (50, 1), (50, 2), (50, 3), (50, 9), (50, 11), (50, 13),
];

const SSAO_NODES: &[(u32, u32, &str)] = &[
    (20, NODE_KIND_PASS,     "ssao_compute"),
    (21, NODE_KIND_RESOURCE, "ssao_target"),
    (22, NODE_KIND_PASS,     "ssao_blur"),
    (23, NODE_KIND_RESOURCE, "ssao_blurred"),
];
const SSAO_EDGES: &[(u32, u32)] = &[
    (8, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 9),
    (51, 20), (51, 22),
];

const BLOOM_NODES: &[(u32, u32, &str)] = &[
    (30, NODE_KIND_PASS,     "bloom_downsample"),
    (31, NODE_KIND_RESOURCE, "bloom_pyramid"),
    (32, NODE_KIND_PASS,     "bloom_upsample"),
    (33, NODE_KIND_RESOURCE, "bloom_result"),
];
const BLOOM_EDGES: &[(u32, u32)] = &[
    (10, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 11),
    (50, 30), (50, 32),
];

// Periodic label updates. The viewer should reflect the new label
// without losing the node's connections or pinned state.
const NODE_LABEL_UPDATES: &[(u32, &[&str])] = &[
    (3,  &["geometry_main", "geometry_main_v2", "geometry_consolidated"]),
    (9,  &["lighting_deferred", "lighting_pbr", "lighting_pbr_indirect"]),
    (13, &["ui_compose", "ui_compose_msaa", "ui_compose_taa"]),
];

fn setup_initial_graph(link: &Arc<LiveLink>) {
    for &(id, k, l) in CORE_NODES { link.record_node(id, k, l); }
    for &(a, b) in CORE_EDGES { link.record_edge(a, b); }
}

// ---- Initial allocations -----------------------------------------------

fn setup_initial_allocations(link: &Arc<LiveLink>) -> [u64; 4] {
    let handles: [u64; 4] = [
        0xCAFE_0000_AAAA_0001,
        0xCAFE_0000_AAAA_0002,
        0xCAFE_0000_AAAA_0003,
        0xCAFE_0000_AAAA_0004,
    ];
    let allocs: &[(usize, u64, u64, &str)] = &[
        (0, 0,                    8 * 1024 * 1024,   "renderer::shadow_atlas"),
        (0, 8 * 1024 * 1024,      32 * 1024 * 1024,  "renderer::gbuffer"),
        (0, 40 * 1024 * 1024,     16 * 1024 * 1024,  "renderer::hdr_color"),
        (0, 56 * 1024 * 1024,     4 * 1024 * 1024,   "renderer::ldr_color"),
        (1, 0,                    2 * 1024 * 1024,   "scene::vertex_buffer"),
        (1, 2 * 1024 * 1024,      1024 * 1024,       "scene::index_buffer"),
        (1, 3 * 1024 * 1024,      512 * 1024,        "scene::skin_matrices"),
        (1, 4 * 1024 * 1024,      256 * 1024,        "ui::glyph_atlas"),
        (2, 0,                    16 * 1024 * 1024,  "staging::upload_ring"),
        (2, 16 * 1024 * 1024,     4 * 1024 * 1024,   "staging::readback_ring"),
        (3, 0,                    1024 * 1024,       "descriptors::material_pool"),
        (3, 1024 * 1024,          512 * 1024,        "descriptors::frame_pool"),
    ];
    for &(idx, off, size, site) in allocs {
        link.record_allocation(handles[idx], off, size, site);
    }
    handles
}

// ---- Resource name registry --------------------------------------------

struct ResourceHandles {
    gbuffer_albedo_image: u64,
    gbuffer_normal_image: u64,
    gbuffer_depth_image: u64,
    hdr_color_image: u64,
    shadow_atlas_image: u64,

    scene_vertex_buffer: u64,
    scene_index_buffer: u64,
    skin_matrices_buffer: u64,
    glyph_atlas_image: u64,

    nearest_sampler: u64,
    linear_sampler: u64,
    shadow_sampler: u64,
}

fn synthetic_handles() -> ResourceHandles {
    ResourceHandles {
        gbuffer_albedo_image: 0x1AAA_0000_0000_0001,
        gbuffer_normal_image: 0x1AAA_0000_0000_0002,
        gbuffer_depth_image:  0x1AAA_0000_0000_0003,
        hdr_color_image:      0x1AAA_0000_0000_0004,
        shadow_atlas_image:   0x1AAA_0000_0000_0005,
        glyph_atlas_image:    0x1AAA_0000_0000_0006,

        scene_vertex_buffer:  0x2BBB_0000_0000_0001,
        scene_index_buffer:   0x2BBB_0000_0000_0002,
        skin_matrices_buffer: 0x2BBB_0000_0000_0003,

        nearest_sampler:      0x3CCC_0000_0000_0001,
        linear_sampler:       0x3CCC_0000_0000_0002,
        shadow_sampler:       0x3CCC_0000_0000_0003,
    }
}

fn register_resource_names(link: &Arc<LiveLink>, mem: &[u64; 4]) {
    link.record_resource_name(mem[0], RES_KIND_DEVICE_MEMORY, "heap::device_local_images");
    link.record_resource_name(mem[1], RES_KIND_DEVICE_MEMORY, "heap::device_local_buffers");
    link.record_resource_name(mem[2], RES_KIND_DEVICE_MEMORY, "heap::staging_ring");
    link.record_resource_name(mem[3], RES_KIND_DEVICE_MEMORY, "heap::descriptor_pool");

    let h = synthetic_handles();

    link.record_resource_name(h.gbuffer_albedo_image, RES_KIND_IMAGE, "gbuffer::albedo");
    link.record_resource_name(h.gbuffer_normal_image, RES_KIND_IMAGE, "gbuffer::normal");
    link.record_resource_name(h.gbuffer_depth_image,  RES_KIND_IMAGE, "gbuffer::depth");
    link.record_resource_name(h.hdr_color_image,      RES_KIND_IMAGE, "lighting::hdr_color");
    link.record_resource_name(h.shadow_atlas_image,   RES_KIND_IMAGE, "shadow::atlas_4k");
    link.record_resource_name(h.glyph_atlas_image,    RES_KIND_IMAGE, "ui::glyph_atlas_2k");

    link.record_resource_name(h.scene_vertex_buffer,  RES_KIND_BUFFER, "scene::vbo");
    link.record_resource_name(h.scene_index_buffer,   RES_KIND_BUFFER, "scene::ibo");
    link.record_resource_name(h.skin_matrices_buffer, RES_KIND_BUFFER, "scene::skin_palette");

    link.record_resource_name(h.nearest_sampler, RES_KIND_SAMPLER, "sampler::nearest_clamp");
    link.record_resource_name(h.linear_sampler,  RES_KIND_SAMPLER, "sampler::linear_repeat");
    link.record_resource_name(h.shadow_sampler,  RES_KIND_SAMPLER, "sampler::shadow_pcf");
}

// Periodic resource name updates. The viewer must reflect new names
// in tooltips and Memory View labels without losing track of the
// underlying handle. Used on a 90s schedule.
fn rename_resources(link: &Arc<LiveLink>, cycle: u64, mem: &[u64; 4]) {
    let h = synthetic_handles();
    match cycle % 4 {
        1 => {
            link.record_resource_name(h.gbuffer_albedo_image, RES_KIND_IMAGE,
                "gbuffer::albedo_v2");
            link.record_resource_name(mem[2], RES_KIND_DEVICE_MEMORY,
                "heap::staging_ring_consolidated");
        }
        2 => {
            link.record_resource_name(h.shadow_atlas_image, RES_KIND_IMAGE,
                "shadow::atlas_4k_csm");
            link.record_resource_name(h.scene_vertex_buffer, RES_KIND_BUFFER,
                "scene::vbo_packed");
        }
        3 => {
            link.record_resource_name(h.hdr_color_image, RES_KIND_IMAGE,
                "lighting::hdr_color_r11g11b10");
            link.record_resource_name(h.linear_sampler, RES_KIND_SAMPLER,
                "sampler::linear_repeat_aniso16");
        }
        _ => {
            // Restore originals so the cycle is observable.
            register_resource_names(link, mem);
        }
    }
}

// ---- Shader printf ----------------------------------------------------

struct PrintfSample {
    stage: u32,
    location: &'static str,
    fmt: &'static str,
}

const PRINTF_SAMPLES: &[PrintfSample] = &[
    PrintfSample { stage: PRINTF_STAGE_VS, location: "skin.vert:42",
        fmt: "skinned_pos: [{:.3}, {:.3}, {:.3}]" },
    PrintfSample { stage: PRINTF_STAGE_FS, location: "lighting.frag:128",
        fmt: "Ld at uv=({:.3},{:.3}) = {:.4}" },
    PrintfSample { stage: PRINTF_STAGE_FS, location: "lighting.frag:201",
        fmt: "WARN: NaN detected in BRDF output" },
    PrintfSample { stage: PRINTF_STAGE_CS, location: "ssao.comp:64",
        fmt: "tile={} occlusion={:.3}" },
    PrintfSample { stage: PRINTF_STAGE_CS, location: "bloom_down.comp:18",
        fmt: "lod={} sample_radius={}" },
    PrintfSample { stage: PRINTF_STAGE_VS, location: "geom.vert:88",
        fmt: "instance={} matrix_det={:.3}" },
    PrintfSample { stage: PRINTF_STAGE_FS, location: "tonemap.frag:55",
        fmt: "exposure={:.2} max_lum={:.4}" },
    PrintfSample { stage: PRINTF_STAGE_CS, location: "ssao.comp:64",
        fmt: "tile={} occlusion={:.3}" },
    PrintfSample { stage: PRINTF_STAGE_RGEN, location: "primary.rgen:34",
        fmt: "ray bounce={} energy={:.3}" },
    PrintfSample { stage: PRINTF_STAGE_FS, location: "ui.frag:12",
        fmt: "ASSERT: invalid glyph index {}" },
];

fn emit_shader_printf(link: &Arc<LiveLink>, frame: u64) {
    let s = &PRINTF_SAMPLES[(frame as usize) % PRINTF_SAMPLES.len()];
    let v0 = (frame as f32 * 0.013).sin();
    let v1 = (frame as f32 * 0.017).cos();
    let v2 = (frame as f32 * 0.011).sin();
    // Substitute "{}" placeholders with synthetic values; we use the
    // simplest possible interpolation so PRINTF_SAMPLES table stays
    // declarative and the resulting strings vary by frame (stress
    // testing the dedup path).
    let msg = if s.fmt.matches("{").count() == 0 {
        s.fmt.to_string()
    } else if s.fmt.contains("WARN") || s.fmt.contains("ASSERT") {
        s.fmt.to_string()
    } else if s.fmt.matches("{").count() == 1 {
        s.fmt.replace("{}", &format!("{}", frame % 256))
    } else if s.fmt.matches("{").count() == 2 {
        s.fmt.replace("{}", &format!("{}", frame % 64))
            .replacen("{:.3}", &format!("{:.3}", v0), 1)
            .replacen("{:.4}", &format!("{:.4}", v1.abs()), 1)
            .replacen("{:.2}", &format!("{:.2}", v2 * 4.0 + 8.0), 1)
    } else {
        format!("{} v0={:.3} v1={:.3} v2={:.3}", s.fmt, v0, v1, v2)
    };
    link.record_shader_printf(s.stage, 0x500, s.location, &msg);
}

// ---- Hang detection ---------------------------------------------------

fn emit_hang_event(link: &Arc<LiveLink>, hang_idx: u64) {
    // Synthetic breadcrumb trail. Last completed and first pending
    // ids vary per hang so the demo shows different "where it stuck"
    // results across simulated hangs.
    let trail: &[(u32, &str, bool)] = &[
        (1, "begin_frame", true),
        (2, "shadow_cascade", true),
        (3, "geometry_main", true),
        (4, "ssao_compute", true),
        (5, "lighting_deferred", hang_idx % 2 == 0),
        (6, "bloom_downsample", false),
        (7, "tonemap", false),
        (8, "ui_compose", false),
        (9, "present", false),
    ];
    let last_done = trail.iter().rev().find(|(_, _, c)| *c)
        .map(|(id, l, _)| (*id, *l)).unwrap_or((0, ""));
    let first_pending = trail.iter().find(|(_, _, c)| !*c)
        .map(|(id, l, _)| (*id, *l)).unwrap_or((0, ""));
    let elapsed_ns = 5_300_000_000_u64 + (hang_idx * 137_000_000);
    let parent = link.record_hang_detected(
        0xF00D_0000_BEEF_0000 + hang_idx,
        elapsed_ns,
        "main_render_submit",
        last_done.0, first_pending.0,
        last_done.1, first_pending.1,
        trail.len() as u32,
    );
    for (id, label, completed) in trail {
        link.record_breadcrumb(parent, *id, *completed, label);
    }
}

// ---- Device fault -----------------------------------------------------

fn emit_device_fault(link: &Arc<LiveLink>, fault_idx: u64) {
    let descriptions: &[&str] = &[
        "Page fault detected on GPU virtual address 0xCAFEBABE during \
         shader read. Faulting unit: SM 3, warp 7, lane 12. The address \
         maps to an unmapped region of the device's virtual memory \
         space. Most likely cause is a buffer device address decoded \
         from uninitialized push constant memory.",
        "Hardware exception in shader execution unit: undefined \
         instruction at SPIR-V offset 0x4A8. The faulting compute \
         shader was dispatched with a workgroup count exceeding the \
         device's maxComputeWorkGroupCount[0] limit, which the \
         driver routed to an invalid command buffer.",
        "Memory access violation: image store to a pixel outside the \
         attachment extent. The fragment shader emits gl_FragCoord-\
         indexed writes via imageStore() but the bound image view's \
         subresource range is smaller than the framebuffer extent. \
         Reduce viewport size or expand the storage image.",
    ];
    let desc = descriptions[(fault_idx as usize) % descriptions.len()];
    link.record_device_fault(
        desc,
        2 + (fault_idx % 3) as u32,
        1 + (fault_idx % 2) as u32,
        4096,
        true,
        fault_idx % 2 == 0,
        fault_idx % 3 == 0,
        12 + (fault_idx % 8) as u32,
        16,
        12 + (fault_idx % 4) as u32,
    );
}

// ---- Object lifetime --------------------------------------------------

const PERMANENT_OBJECTS: &[(u64, u32, &str, &str, u32, u32, &str)] = &[
    (0x4000_0000_0000_0001, VK_TYPE_PIPELINE, "shadow_pipeline",
        "src/render/shadow.rs", 42, 8, "ShadowPass::create_pipeline"),
    (0x4000_0000_0000_0002, VK_TYPE_PIPELINE, "geometry_pipeline",
        "src/render/geometry.rs", 88, 12, "GeometryPass::build"),
    (0x4000_0000_0000_0003, VK_TYPE_PIPELINE, "lighting_pipeline",
        "src/render/lighting.rs", 156, 8, "LightingPass::compile_deferred"),
    (0x4000_0000_0000_0010, VK_TYPE_BUFFER, "scene_constants_buffer",
        "src/scene/uniforms.rs", 24, 4, "Scene::init"),
    (0x4000_0000_0000_0020, VK_TYPE_SAMPLER, "fxaa_sampler",
        "src/postfx/fxaa.rs", 17, 6, "Fxaa::new"),
    (0x4000_0000_0000_0030, VK_TYPE_DESCRIPTOR_SET, "frame_set_persistent",
        "src/render/frame_set.rs", 71, 4, "FrameSet::create"),
    // The last two entries are never used and never destroyed - they
    // model the "orphan leak" pattern the Lifetime view's Orphans
    // chip is designed to surface.
    (0x4000_0000_0000_0100, VK_TYPE_BUFFER, "debug_overlay_buffer_v1",
        "src/dev/debug_overlay.rs", 312, 16, "DebugOverlay::init_v1"),
    (0x4000_0000_0000_0101, VK_TYPE_SAMPLER, "deprecated_lod_sampler",
        "src/legacy/lod.rs", 94, 8, "LodChain::create_old"),
];

fn register_permanent_objects(link: &Arc<LiveLink>) {
    for &(h, ty, name, file, line, col, func) in PERMANENT_OBJECTS {
        link.record_object_registered(h, ty, name, file, line, col, func);
    }
}

struct TransientObject { handle: u64, object_type: u32, age_frames: u32 }

fn churn_transient_object(
    link: &Arc<LiveLink>,
    transient: &mut Vec<TransientObject>,
    next_handle: &mut u64,
    frame: u64,
) {
    // Register a new transient object with a varied type so the
    // Lifetime view's by-type breakdown row populates with realistic
    // counts.
    let kinds: &[(u32, &str, &str, u32, &str)] = &[
        (VK_TYPE_BUFFER, "tmp::scratch_vbo", "src/render/scratch.rs", 88, "Scratch::alloc_vbo"),
        (VK_TYPE_IMAGE, "tmp::readback_target", "src/render/readback.rs", 32, "Readback::create_image"),
        (VK_TYPE_DESCRIPTOR_SET, "tmp::frame_set", "src/render/frame_set.rs", 142, "FrameSet::allocate_per_frame"),
        (VK_TYPE_PIPELINE, "tmp::variant_pipeline", "src/render/pipeline_cache.rs", 411, "Cache::specialize"),
    ];
    let pick = kinds[(frame as usize) % kinds.len()];
    *next_handle += 1;
    let h = 0x4100_0000_0000_0000 | *next_handle;
    link.record_object_registered(h, pick.0, pick.1, pick.2, pick.3, 4, pick.4);
    transient.push(TransientObject { handle: h, object_type: pick.0, age_frames: 0 });
}

fn destroy_old_transients(link: &Arc<LiveLink>, transient: &mut Vec<TransientObject>) {
    // Destroy objects older than ~600 frames. Vary usage_count so
    // the destroyed-history list mixes heavily-used and orphan-on-
    // destroy entries.
    let mut i = 0;
    while i < transient.len() {
        if transient[i].age_frames > 600 {
            let obj = transient.remove(i);
            let usage = if obj.handle & 0x7 == 0 {
                0
            } else {
                ((obj.handle.wrapping_mul(31)) % 4096) + 1
            };
            link.record_object_destroyed(obj.handle, obj.object_type, usage);
        } else {
            transient[i].age_frames += 1;
            i += 1;
        }
    }
}

// ---- Descriptor audit -------------------------------------------------

fn emit_descriptor_issue(link: &Arc<LiveLink>, idx: u64, h: &ResourceHandles) {
    let cases: &[(u64, u32, u32, u64, &str, &str)] = &[
        (0xD555_0000_0001, 0, RES_KIND_IMAGE, h.gbuffer_albedo_image,
            "frame_set_persistent",
            "ImageView for gbuffer::albedo destroyed during resize \
             but still bound at set 0 binding 0"),
        (0xD555_0000_0002, 2, RES_KIND_SAMPLER, h.linear_sampler,
            "material_set_marble",
            "Sampler unregistered when material was reloaded but \
             descriptor write was not refreshed"),
        (0xD555_0000_0003, 1, RES_KIND_BUFFER, h.skin_matrices_buffer,
            "skin_set",
            "Buffer destroyed by streaming subsystem, descriptor still \
             references the freed memory range"),
        (0xD555_0000_0004, 3, RES_KIND_OTHER, 0xDEAD_BEEF_F00D_0001,
            "debug_overlay_set",
            "Generic resource destruction not propagated to its \
             descriptor reference"),
    ];
    let c = cases[(idx as usize) % cases.len()];
    link.record_descriptor_issue(c.0, c.1, c.2, c.3, c.4, c.5);
}

// ---- Aliasing detector ------------------------------------------------

fn emit_aliasing_conflict(link: &Arc<LiveLink>, idx: u64, h: &ResourceHandles) {
    let cases: &[(u64, u32, u32, u32, u32, u32, &str, &str, &str, &str)] = &[
        (h.gbuffer_albedo_image, 5, 7, STAGE_COLOR_ATTACHMENT, STAGE_FRAGMENT_SHADER,
            ACCESS_READ, "gbuffer::albedo", "geometry_main", "lighting_deferred",
            "RAW: gbuffer written by geometry pass then sampled by lighting \
             without an intervening barrier"),
        (h.shadow_atlas_image, 3, 4, STAGE_COLOR_ATTACHMENT, STAGE_COLOR_ATTACHMENT,
            ACCESS_WRITE, "shadow::atlas_4k", "shadow_cascade", "shadow_cascade_2",
            "WAW: two shadow cascades write to overlapping atlas slices in \
             the same submission"),
        (h.hdr_color_image, 9, 10, STAGE_FRAGMENT_SHADER, STAGE_COMPUTE_SHADER,
            ACCESS_READ, "lighting::hdr_color", "tonemap", "bloom_downsample",
            "RAW: hdr_color sampled by tonemap then read by bloom compute, \
             both in the same recorder without a barrier"),
        (h.scene_vertex_buffer, 1, 2, STAGE_TRANSFER, STAGE_VERTEX_SHADER,
            ACCESS_READ, "scene::vbo", "buffer_upload", "vertex_skin",
            "RAW: vertex buffer staged by transfer queue and immediately \
             consumed by vertex shader without a release/acquire pair"),
    ];
    let c = cases[(idx as usize) % cases.len()];
    link.record_aliasing_conflict(c.0, c.1, c.2, c.3, c.4, c.5,
        c.6, c.7, c.8, c.9);
}

// ---- Pipeline audit ---------------------------------------------------

fn emit_pipeline_issue(link: &Arc<LiveLink>, idx: u64) {
    let cases: &[(u64, u32, &str, &str)] = &[
        (0x4000_0000_0000_0001, PIPELINE_ISSUE_DESCRIPTOR_COUNT,
            "shadow_pipeline",
            "Pipeline expects 3 descriptor sets but bind_descriptor_sets \
             provided only 2. Set 2 (frame_globals) is missing."),
        (0x4000_0000_0000_0002, PIPELINE_ISSUE_PUSH_CONSTANT_RANGE,
            "geometry_pipeline",
            "Push constant write at offset 64 size 32 falls outside the \
             declared layout range [0, 80)."),
        (0x4000_0000_0000_0003, PIPELINE_ISSUE_LAYOUT_COMPATIBILITY,
            "lighting_pipeline",
            "Bound pipeline layout has 4 set layouts; pipeline was created \
             with a layout having 5. Set index 4 is incompatible."),
        (0x4000_0000_0000_0001, PIPELINE_ISSUE_STAGE_INTERFACE,
            "shadow_pipeline",
            "Vertex shader output 'v_normal' (vec3) at location 1 is not \
             consumed by the fragment shader; fragment expects \
             'v_color' (vec4) at location 1 instead."),
    ];
    let c = cases[(idx as usize) % cases.len()];
    link.record_pipeline_issue(c.0, c.1, c.2, c.3);
}

// ---- Allocation profiler synthetic snapshot ---------------------------

/// Specification for one synthetic allocation site. The demo uses a
/// fixed table so the same call sites appear across snapshot epochs;
/// only their stats vary, modelling realistic patterns of live memory
/// (stable, oscillating, slow-leak).
struct AllocSiteSpec {
    function: &'static str,
    file: &'static str,
    line: u32,
    /// Baseline active bytes when the site is at rest.
    base_active_bytes: u64,
    /// Baseline active allocation count.
    base_active_allocs: u64,
    /// Per-epoch oscillation amplitude in bytes. The site's active
    /// bytes wander in `[base, base + oscillation]` so the viewer's
    /// mini-bars actually move between snapshots.
    oscillation: u64,
    /// Per-epoch growth (models a slow leak). Reset implicitly when
    /// the demo restarts.
    growth_per_epoch: u64,
    /// Per-epoch lifetime allocation count increment.
    allocs_per_epoch: u64,
}

const ALLOC_SITES: &[AllocSiteSpec] = &[
    AllocSiteSpec {
        function: "Renderer::create_gbuffer",
        file: "src/render/gbuffer.rs", line: 42,
        base_active_bytes: 32 * 1024 * 1024, base_active_allocs: 4,
        oscillation: 0, growth_per_epoch: 0, allocs_per_epoch: 0,
    },
    AllocSiteSpec {
        function: "TextureCache::upload",
        file: "src/render/texture_cache.rs", line: 188,
        base_active_bytes: 16 * 1024 * 1024, base_active_allocs: 12,
        oscillation: 4 * 1024 * 1024, growth_per_epoch: 0,
        allocs_per_epoch: 1,
    },
    AllocSiteSpec {
        function: "MeshLoader::create_vbo",
        file: "src/scene/mesh.rs", line: 312,
        base_active_bytes: 8 * 1024 * 1024, base_active_allocs: 6,
        oscillation: 0, growth_per_epoch: 0, allocs_per_epoch: 0,
    },
    AllocSiteSpec {
        function: "ShadowAtlas::allocate",
        file: "src/render/shadow.rs", line: 67,
        base_active_bytes: 8 * 1024 * 1024, base_active_allocs: 1,
        oscillation: 0, growth_per_epoch: 0, allocs_per_epoch: 0,
    },
    AllocSiteSpec {
        function: "StagingRing::reserve",
        file: "src/memory/staging.rs", line: 145,
        base_active_bytes: 4 * 1024 * 1024, base_active_allocs: 8,
        oscillation: 2 * 1024 * 1024, growth_per_epoch: 0,
        allocs_per_epoch: 4,
    },
    AllocSiteSpec {
        function: "Lighting::cubemap_array",
        file: "src/render/lighting.rs", line: 156,
        base_active_bytes: 4 * 1024 * 1024, base_active_allocs: 2,
        oscillation: 0, growth_per_epoch: 0, allocs_per_epoch: 0,
    },
    AllocSiteSpec {
        function: "Bloom::create_pyramid",
        file: "src/postfx/bloom.rs", line: 156,
        base_active_bytes: 2 * 1024 * 1024, base_active_allocs: 6,
        oscillation: 512 * 1024, growth_per_epoch: 0,
        allocs_per_epoch: 0,
    },
    AllocSiteSpec {
        function: "DescriptorPool::create_frame",
        file: "src/render/descriptor.rs", line: 204,
        base_active_bytes: 1024 * 1024, base_active_allocs: 32,
        oscillation: 256 * 1024, growth_per_epoch: 0,
        allocs_per_epoch: 8,
    },
    AllocSiteSpec {
        function: "PipelineCache::specialize",
        file: "src/render/pipeline_cache.rs", line: 411,
        base_active_bytes: 512 * 1024, base_active_allocs: 16,
        oscillation: 0, growth_per_epoch: 32 * 1024,
        allocs_per_epoch: 2,
    },
    AllocSiteSpec {
        function: "UniformBuffer::create_per_frame",
        file: "src/render/uniform.rs", line: 88,
        base_active_bytes: 256 * 1024, base_active_allocs: 24,
        oscillation: 64 * 1024, growth_per_epoch: 0,
        allocs_per_epoch: 6,
    },
    AllocSiteSpec {
        function: "UI::glyph_atlas_alloc",
        file: "src/ui/font.rs", line: 219,
        base_active_bytes: 256 * 1024, base_active_allocs: 4,
        oscillation: 0, growth_per_epoch: 0, allocs_per_epoch: 0,
    },
    AllocSiteSpec {
        function: "DebugOverlay::v1::buffer",
        file: "src/dev/debug_overlay.rs", line: 312,
        base_active_bytes: 64 * 1024, base_active_allocs: 1,
        oscillation: 0, growth_per_epoch: 8 * 1024,
        allocs_per_epoch: 1,
    },
];

/// Emit one full snapshot batch of allocation sites under one epoch.
///
/// Sites are computed, then ranked by `active_bytes` descending so the
/// producer's `site_index` matches what a real `AllocationProfiler`
/// would report. The viewer relies on this ordering: it preserves the
/// rank as row order to keep scrolling stable across epochs.
///
/// Each site's stats vary between epochs through three independent
/// dimensions:
///
/// - `oscillation`: short-term swing around the baseline. Models pools
///   whose live size depends on transient frame activity.
/// - `growth_per_epoch`: slow accumulation. Models long-running leaks
///   the user will notice in the Lifetime view's orphan list as well.
/// - `allocs_per_epoch`: lifetime allocation count growth. Models
///   high-frequency allocators where active bytes stay roughly stable
///   but cumulative count climbs.
fn emit_alloc_site_snapshot(link: &Arc<LiveLink>, epoch: u64) {
    let mut rows: Vec<(usize, u64, u64, u64, u64, u64)> = ALLOC_SITES
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let osc = if s.oscillation > 0 {
                ((epoch.wrapping_mul(11) + (i as u64) * 13) % 100)
                    * s.oscillation / 100
            } else {
                0
            };
            let growth = s.growth_per_epoch * epoch;
            let active_bytes = s.base_active_bytes + osc + growth;
            let active_allocs = s.base_active_allocs;
            let total_allocs = s.base_active_allocs * 4
                + epoch * s.allocs_per_epoch;
            let total_bytes = s.base_active_bytes * 4
                + epoch * s.allocs_per_epoch * 64 * 1024
                + growth * 3;
            let peak_active_bytes = active_bytes
                + s.oscillation
                + s.base_active_bytes / 8;
            (i, active_bytes, active_allocs, total_allocs,
                total_bytes, peak_active_bytes)
        })
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));

    for (rank, row) in rows.iter().enumerate() {
        let (idx, active_bytes, active_allocs, total_allocs,
             total_bytes, peak_active_bytes) = *row;
        let spec = &ALLOC_SITES[idx];
        let peak_active_allocs = active_allocs + 2 + (epoch / 30);
        link.record_alloc_site(
            epoch,
            rank as u32,
            spec.function,
            spec.file,
            spec.line,
            total_allocs,
            total_bytes,
            active_allocs,
            active_bytes,
            peak_active_allocs,
            peak_active_bytes,
        );
    }
}

// ---- Synthetic validation stream ---------------------------------------

struct VlSample {
    severity: u32,
    /// Graph node id this diagnostic relates to. 0 means none.
    node_id: u32,
    function: &'static str,
    vuid: &'static str,
    message: &'static str,
    /// Index into the resource handle table for the offending object.
    object_slot: u8,
}

const VL_SAMPLES: &[VlSample] = &[
    VlSample {
        severity: VAL_SEVERITY_WARNING,
        node_id: 3,
        function: "vkCmdDrawIndexed",
        vuid: "VUID-vkCmdDrawIndexed-None-04007",
        message: "Pipeline expects vertex attribute at location 3 but \
                  bound vertex buffer does not provide it. Reads will return zeros.",
        object_slot: 1,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 0,
        function: "vkAllocateMemory",
        vuid: "UNASSIGNED-BestPractices-vkAllocateMemory-small-allocation",
        message: "Small allocation of 64 KiB. Sub-allocate from a pool \
                  to reduce VkDeviceMemory pressure.",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_ERROR,
        node_id: 6,
        function: "vkCmdPipelineBarrier",
        vuid: "VUID-VkImageMemoryBarrier-oldLayout-01213",
        message: "oldLayout is COLOR_ATTACHMENT_OPTIMAL but image was \
                  not created with VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT. \
                  Layout transition is invalid.",
        object_slot: 2,
    },
    VlSample {
        severity: VAL_SEVERITY_WARNING,
        node_id: 0,
        function: "vkQueueSubmit",
        vuid: "UNASSIGNED-BestPractices-pipeline-stall",
        message: "Queue submit caused a CPU stall waiting for fence. \
                  Consider increasing frames-in-flight or batching submits.",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_ERROR,
        node_id: 13,
        function: "vkCmdBindDescriptorSets",
        vuid: "VUID-vkCmdBindDescriptorSets-pDescriptorSets-04616",
        message: "Descriptor set 0 binding 2 references a sampler that \
                  was destroyed but not unbound. Subsequent draws produce \
                  undefined results.",
        object_slot: 7,
    },
    VlSample {
        severity: VAL_SEVERITY_WARNING,
        node_id: 8,
        function: "vkCreateImage",
        vuid: "UNASSIGNED-BestPractices-CreateImage-Depth32Format",
        message: "Created VK_FORMAT_D32_SFLOAT depth image. Some drivers \
                  prefer D24_UNORM_S8_UINT for tiling efficiency on this hardware.",
        object_slot: 4,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 3,
        function: "vkCmdBeginRenderPass",
        vuid: "UNASSIGNED-CoreValidation-DrawState-RenderpassRedundant",
        message: "Render pass has a single subpass with one color and one \
                  depth attachment. Consider VK_KHR_dynamic_rendering on Vulkan 1.3.",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_ERROR,
        node_id: 9,
        function: "vkUpdateDescriptorSets",
        vuid: "VUID-VkWriteDescriptorSet-descriptorType-00322",
        message: "descriptorType is COMBINED_IMAGE_SAMPLER but bound image \
                  view's format is not filterable with the supplied sampler.",
        object_slot: 8,
    },
    VlSample {
        severity: VAL_SEVERITY_WARNING,
        node_id: 13,
        function: "vkCmdCopyBufferToImage",
        vuid: "UNASSIGNED-BestPractices-vkCmdCopyBufferToImage-pre-transition",
        message: "Image is in TRANSFER_DST_OPTIMAL but barrier was \
                  inserted with srcStageMask=ALL_COMMANDS. Tighten to \
                  TOP_OF_PIPE for upload-only paths.",
        object_slot: 6,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 0,
        function: "vkBeginCommandBuffer",
        vuid: "UNASSIGNED-CoreValidation-CommandBufferReuse",
        message: "Command buffer reset and re-recorded 1024 times this \
                  session. No issues, just a usage note.",
        object_slot: 0,
    },
];

// Dedicated info-level stream so the Info severity filter chip is
// always populated. Without these the filter would look broken
// because info events from VL_SAMPLES come through too rarely.
const VL_INFO_SAMPLES: &[VlSample] = &[
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 0,
        function: "vkEnumerateInstanceLayerProperties",
        vuid: "UNASSIGNED-Loader-Layer",
        message: "Instance layer 'VK_LAYER_KHRONOS_validation' loaded \
                  successfully (version 1.3.275).",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 0,
        function: "vkCreateInstance",
        vuid: "UNASSIGNED-Loader-Instance",
        message: "Vulkan instance created with API version 1.3. \
                  Available device extensions: 87.",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 11,
        function: "vkCmdSetViewport",
        vuid: "UNASSIGNED-Performance-DynamicState",
        message: "Pipeline uses dynamic viewport state. Consider static \
                  viewport for shader cache hits across resolutions.",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 1,
        function: "vkBindBufferMemory",
        vuid: "UNASSIGNED-CoreValidation-BindMemory",
        message: "Buffer bound to memory region with required alignment \
                  16, actual offset alignment 256.",
        object_slot: 1,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 0,
        function: "vkCreateDevice",
        vuid: "UNASSIGNED-Feature-Probe",
        message: "Optional feature 'samplerAnisotropy' is supported and \
                  enabled. maxSamplerAnisotropy = 16.0.",
        object_slot: 0,
    },
    VlSample {
        severity: VAL_SEVERITY_INFO,
        node_id: 0,
        function: "vkGetPhysicalDeviceMemoryProperties",
        vuid: "UNASSIGNED-MemoryProbe",
        message: "Device exposes 4 memory heaps, 11 memory types. \
                  Resizable BAR: not available on this configuration.",
        object_slot: 0,
    },
];

fn handle_for_slot(h: &ResourceHandles, slot: u8) -> u64 {
    match slot {
        1 => h.scene_vertex_buffer,
        2 => h.gbuffer_albedo_image,
        3 => h.gbuffer_normal_image,
        4 => h.gbuffer_depth_image,
        5 => h.hdr_color_image,
        6 => h.glyph_atlas_image,
        7 => h.nearest_sampler,
        8 => h.linear_sampler,
        9 => h.shadow_sampler,
        _ => 0,
    }
}

const VK_OBJECT_TYPE_BUFFER: u32 = 9;
const VK_OBJECT_TYPE_IMAGE: u32 = 10;
const VK_OBJECT_TYPE_SAMPLER: u32 = 21;

fn object_type_for_slot(slot: u8) -> u32 {
    match slot {
        1 => VK_OBJECT_TYPE_BUFFER,
        2..=6 => VK_OBJECT_TYPE_IMAGE,
        7..=9 => VK_OBJECT_TYPE_SAMPLER,
        _ => 0,
    }
}

// ---- Pipeline statistics emission --------------------------------------

fn emit_pipeline_stats(
    link: &Arc<LiveLink>,
    frame: u64,
    ssao: bool,
    bloom: bool,
    burst: bool,
) {
    let j = (frame * 17) % 4_096;
    // During burst mode, vertex/fragment counts triple to simulate a
    // crowded scene. The viewer's pipeline stats panel should show
    // visible spikes synchronized with the burst window.
    let mul: u64 = if burst { 3 } else { 1 };

    link.record_pipeline_stats(
        "shadow_cascade",
        80_000 * mul + j, 0, 0,
        80_000 * mul + j, 26_000 * mul, 26_000 * mul, 24_500 * mul,
        0, 0, 0, 0,
    );
    link.record_pipeline_stats(
        "geometry_main",
        80_000 * mul + j,
        12_500_000 * mul + (frame * 137) % 1_000_000, 0,
        80_000 * mul + j, 26_000 * mul, 26_000 * mul, 24_800 * mul,
        0, 0, 0, 0,
    );
    link.record_pipeline_stats(
        "lighting_deferred",
        4, 2_073_600 * mul + (frame * 53) % 50_000, 0,
        4, 1, 1, 1, 0, 0, 0, 0,
    );
    link.record_pipeline_stats(
        "tonemap",
        4, 2_073_600, 0,
        4, 1, 1, 1, 0, 0, 0, 0,
    );
    link.record_pipeline_stats(
        "ui_compose",
        500 + j / 100,
        80_000 * mul + (frame * 23) % 30_000, 0,
        500, 200, 200, 195, 0, 0, 0, 0,
    );

    if ssao {
        link.record_pipeline_stats(
            "ssao_compute",
            0, 0,
            518_400 * mul + (frame * 47) % 20_000,
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        link.record_pipeline_stats(
            "ssao_blur",
            0, 0, 518_400 * mul,
            0, 0, 0, 0, 0, 0, 0, 0,
        );
    }
    if bloom {
        link.record_pipeline_stats(
            "bloom_downsample",
            4, 518_400 * mul + (frame * 41) % 20_000, 0,
            4, 1, 1, 1, 0, 0, 0, 0,
        );
        link.record_pipeline_stats(
            "bloom_upsample",
            4, 2_073_600 * mul + (frame * 37) % 80_000, 0,
            4, 1, 1, 1, 0, 0, 0, 0,
        );
    }
}

// ---- Memory budget emission --------------------------------------------

const HEAP_FLAG_DEVICE_LOCAL: u32 = 1;

fn emit_budget(link: &Arc<LiveLink>, frame: u64, transient_count: usize, burst: bool) {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    // Burst mode pushes more memory pressure to make the bars swing
    // visibly during the demo.
    let burst_bias = if burst { 32 * MIB } else { 0 };

    let used0 = 60 * MIB + (frame * 31) % (4 * MIB) + burst_bias;
    link.record_budget(0, HEAP_FLAG_DEVICE_LOCAL, used0, 7 * GIB, 8 * GIB);

    let used1 = 4 * MIB + (frame * 17) % (256 * 1024);
    link.record_budget(1, HEAP_FLAG_DEVICE_LOCAL, used1, 7 * GIB, 8 * GIB);

    let used2 = 20 * MIB + (transient_count as u64) * 384 * 1024 + burst_bias / 2;
    link.record_budget(2, 0, used2, 14 * GIB, 16 * GIB);

    let used3 = 1_500_000 + ((frame / 100) % 4) * 256 * 1024;
    link.record_budget(3, 0, used3, 256 * MIB, 256 * MIB);
}

// ---- GPU timeline emission ---------------------------------------------

fn emit_gpu_timeline(
    link: &Arc<LiveLink>,
    frame: u64,
    ssao: bool,
    bloom: bool,
    elapsed_ns: u64,
    burst: bool,
) {
    let frame_start = elapsed_ns.saturating_sub(17_000_000);
    let mut t = frame_start;
    let mul: u64 = if burst { 2 } else { 1 };

    let dur_skin = (80_000 + (frame * 17) % 20_000) * mul;
    link.record_gpu_timestamp(0, 0, STAGE_VERTEX_SHADER, t, dur_skin, "vertex_skin");
    t += dur_skin;

    let dur_shadow = (600_000 + (frame * 31) % 200_000) * mul;
    link.record_gpu_timestamp(0, 0, STAGE_COLOR_ATTACHMENT, t, dur_shadow, "shadow_cascade");
    t += dur_shadow;

    let dur_geo = (1_200_000 + (frame * 23) % 400_000) * mul;
    link.record_gpu_timestamp(0, 0, STAGE_COLOR_ATTACHMENT, t, dur_geo, "geometry_main");
    t += dur_geo;

    if ssao {
        let mut tc = frame_start + 200_000;
        let dur_ssao = (350_000 + (frame * 19) % 100_000) * mul;
        link.record_gpu_timestamp(1, 0, STAGE_COMPUTE_SHADER, tc, dur_ssao, "ssao_compute");
        tc += dur_ssao;
        let dur_blur = (180_000 + (frame * 13) % 50_000) * mul;
        link.record_gpu_timestamp(1, 0, STAGE_COMPUTE_SHADER, tc, dur_blur, "ssao_blur");
    }

    let dur_light = (800_000 + (frame * 29) % 200_000) * mul;
    link.record_gpu_timestamp(0, 0, STAGE_FRAGMENT_SHADER, t, dur_light, "lighting_deferred");
    t += dur_light;

    if bloom {
        let dur_bd = (220_000 + (frame * 11) % 80_000) * mul;
        link.record_gpu_timestamp(0, 0, STAGE_COMPUTE_SHADER, t, dur_bd, "bloom_downsample");
        t += dur_bd;
        let dur_bu = (240_000 + (frame * 17) % 80_000) * mul;
        link.record_gpu_timestamp(0, 0, STAGE_FRAGMENT_SHADER, t, dur_bu, "bloom_upsample");
        t += dur_bu;
    }

    let dur_tone = (150_000 + (frame * 7) % 30_000) * mul;
    link.record_gpu_timestamp(0, 0, STAGE_FRAGMENT_SHADER, t, dur_tone, "tonemap");
    t += dur_tone;

    let dur_ui = (120_000 + (frame * 13) % 40_000) * mul;
    link.record_gpu_timestamp(0, 0, STAGE_FRAGMENT_SHADER, t, dur_ui, "ui_compose");

    if frame % 4 == 0 {
        let xfer_start = frame_start + 4_000_000;
        let dur_xfer = (200_000 + (frame * 41) % 300_000) * mul;
        link.record_gpu_timestamp(
            2, 0, STAGE_TRANSFER, xfer_start, dur_xfer, "buffer_upload");
    }
}

// ---- Sync DAG cycle / orphan synthetic emission ------------------------

fn emit_sync_marks(link: &Arc<LiveLink>, elapsed_secs: u64) {
    let in_cycle_window = elapsed_secs % 60 < 10;
    let in_orphan_window = elapsed_secs % 90 < 8;

    if in_cycle_window {
        link.record_sync_cycle(
            0, 0,
            SYNC_SEVERITY_CYCLE, 12_000,
            1, 0,
            "cycle 0: graphics_main -> compute_async -> graphics_main \
             (semaphore handoff loop)",
        );
        link.record_sync_cycle(
            1, 0,
            SYNC_SEVERITY_CYCLE, 12_000,
            0, 0,
            "cycle 0: compute_async -> graphics_main -> compute_async \
             (semaphore handoff loop)",
        );
    }

    if in_orphan_window {
        link.record_sync_cycle(
            2, 0,
            SYNC_SEVERITY_ORPHAN, 12_000,
            0, 0,
            "orphan signal: sem 0xCAFE_BABE signaled by transfer_upload \
             but never waited on within the recorded window",
        );
    }
}

// ---- Canary corruption synthetic emission ------------------------------

const CANARY_PATTERN_GOOD: [u8; 16] = [
    0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
    0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD, 0xCD,
];

struct CanarySample {
    /// 0 = front guard, 1 = back guard.
    region: u32,
    /// 0 info, 1 warn, 2 err.
    severity: u32,
    /// Memory handle slot (index into the synthetic_handles table).
    handle_kind: CanaryHandleKind,
    user_offset: u64,
    user_size: u64,
    /// Index of first corrupted byte within the guard region.
    first_byte: u32,
    /// Total corrupted bytes in this guard region.
    corrupted_count: u32,
    /// Actual byte at `first_byte` (for the hex diff).
    actual_byte: u8,
    /// Detection context.
    source: &'static str,
    /// Free-form description.
    description: &'static str,
}

#[derive(Copy, Clone)]
enum CanaryHandleKind {
    GbufferAlbedo,
    GbufferDepth,
    HdrColor,
    StagingRing,
    SceneVbo,
    ShadowAtlas,
    GlyphAtlas,
}

fn canary_handle(kind: CanaryHandleKind, h: &ResourceHandles, mem: &[u64; 4]) -> u64 {
    match kind {
        CanaryHandleKind::GbufferAlbedo => h.gbuffer_albedo_image,
        CanaryHandleKind::GbufferDepth => h.gbuffer_depth_image,
        CanaryHandleKind::HdrColor => h.hdr_color_image,
        CanaryHandleKind::StagingRing => mem[2],
        CanaryHandleKind::SceneVbo => h.scene_vertex_buffer,
        CanaryHandleKind::ShadowAtlas => h.shadow_atlas_image,
        CanaryHandleKind::GlyphAtlas => h.glyph_atlas_image,
    }
}

const CANARY_SAMPLES: &[CanarySample] = &[
    CanarySample {
        region: 1,
        severity: 1,
        handle_kind: CanaryHandleKind::GbufferAlbedo,
        user_offset: 8 * 1024 * 1024,
        user_size: 4 * 1024 * 1024,
        first_byte: 0,
        corrupted_count: 1,
        actual_byte: 0xFF,
        source: "Allocator::free()",
        description: "Single-byte overwrite at the start of the back \
                      guard. Likely a one-past-end write in a memcpy \
                      sized with strlen() instead of sizeof() on the \
                      writer side.",
    },
    CanarySample {
        region: 0,
        severity: 1,
        handle_kind: CanaryHandleKind::HdrColor,
        user_offset: 40 * 1024 * 1024,
        user_size: 16 * 1024 * 1024,
        first_byte: 60,
        corrupted_count: 4,
        actual_byte: 0x42,
        source: "verify_all_live()",
        description: "Front guard tail bytes overwritten. ASCII 0x42 \
                      ('B') in the corrupted region suggests a string \
                      literal got memcpy'd into the wrong target.",
    },
    CanarySample {
        region: 1,
        severity: 2,
        handle_kind: CanaryHandleKind::SceneVbo,
        user_offset: 0,
        user_size: 2 * 1024 * 1024,
        first_byte: 0,
        corrupted_count: 64,
        actual_byte: 0xDD,
        source: "Allocator::free()",
        description: "Entire back guard overwritten with 0xDD (MSVC \
                      freed-heap pattern). Likely a use-after-free where \
                      the freed block's pattern bled into the live \
                      buffer's tail through a stale pointer.",
    },
    CanarySample {
        region: 0,
        severity: 1,
        handle_kind: CanaryHandleKind::StagingRing,
        user_offset: 16 * 1024 * 1024,
        user_size: 4 * 1024 * 1024,
        first_byte: 8,
        corrupted_count: 2,
        actual_byte: 0x00,
        source: "quarantine eviction",
        description: "Quarantine re-verification found two zero bytes \
                      in the front guard. Producer side appears to be \
                      racing on a memset that crosses the guard boundary.",
    },
    CanarySample {
        region: 1,
        severity: 1,
        handle_kind: CanaryHandleKind::GbufferDepth,
        user_offset: 8 * 1024 * 1024 + 4 * 1024 * 1024 + 16 * 1024 * 1024,
        user_size: 4 * 1024 * 1024,
        first_byte: 16,
        corrupted_count: 8,
        actual_byte: 0x7F,
        source: "Allocator::free()",
        description: "8-byte run starting at offset 16 in the back guard. \
                      Pattern 0x7F suggests a float-as-byte-stream write \
                      from a debug overlay path.",
    },
    CanarySample {
        region: 1,
        severity: 2,
        handle_kind: CanaryHandleKind::ShadowAtlas,
        user_offset: 0,
        user_size: 8 * 1024 * 1024,
        first_byte: 4,
        corrupted_count: 32,
        actual_byte: 0x55,
        source: "Allocator::free()",
        description: "Shadow atlas back guard corrupted with alternating \
                      0x55 pattern. May indicate a shader writing past \
                      the framebuffer extent due to viewport mismatch.",
    },
    CanarySample {
        region: 0,
        severity: 0,
        handle_kind: CanaryHandleKind::GlyphAtlas,
        user_offset: 4 * 1024 * 1024,
        user_size: 256 * 1024,
        first_byte: 50,
        corrupted_count: 2,
        actual_byte: 0xCC,
        source: "verify_all_live()",
        description: "Two-byte corruption near user data boundary on the \
                      front guard. Pattern 0xCC suggests an MSVC stack \
                      allocator pattern leaking through. Low severity \
                      because writes are inside guard bounds.",
    },
];

fn emit_canary(
    link: &Arc<LiveLink>,
    sample: &CanarySample,
    h: &ResourceHandles,
    mem: &[u64; 4],
) {
    let handle = canary_handle(sample.handle_kind, h, mem);
    let canary_word = 0xDEAD_BEEF_CAFE_BABE_u64;

    let mut actual = CANARY_PATTERN_GOOD;
    let start = sample.first_byte as usize;
    let end = (start + sample.corrupted_count as usize).min(actual.len());
    for byte in actual.iter_mut().take(end).skip(start) {
        *byte = sample.actual_byte;
    }

    link.record_canary_corruption(
        handle,
        sample.user_offset,
        sample.user_size,
        64,
        canary_word,
        sample.region,
        sample.severity,
        sample.first_byte,
        sample.corrupted_count,
        canary_word.to_le_bytes()[(sample.first_byte as usize) % 8],
        sample.actual_byte,
        &CANARY_PATTERN_GOOD,
        &actual,
        sample.source,
        sample.description,
    );
}

fn emit_hardened_stats(link: &Arc<LiveLink>, elapsed: Duration, total_corruptions: u64) {
    let secs = elapsed.as_secs();
    let total_allocs = 12 + secs * 2;
    let total_frees = total_allocs.saturating_sub(8);
    let active = (total_allocs - total_frees).max(8);
    let q_entries = (secs / 7) % 24;
    let q_bytes = q_entries * 384 * 1024;
    let active_bytes = active * 256 * 1024 + 60 * 1024 * 1024;
    let peak_allocs = active.max(40 + secs / 60);
    let peak_bytes = active_bytes.max(80 * 1024 * 1024);

    link.record_hardened_stats(
        total_allocs,
        total_frees,
        active,
        active_bytes,
        q_entries,
        q_bytes,
        total_corruptions,
        peak_allocs,
        peak_bytes,
    );
}

// ---- Determinism synthetic emission ------------------------------------

const DETERMINISM_RUNS: u32 = 5;
const BASELINE_AGGREGATE: u64 = 0xA1B2_C3D4_E5F6_7788;
const DIVERGENT_AGGREGATE: u64 = 0xA1B2_C3D4_E5F6_9999;
const PHYSICS_AGGREGATE: u64 = 0xDEAD_F00D_BEEF_CAFE;
const DETERMINISM_BUFFER_CAPTURES: u32 = 3;
const DETERMINISM_IMAGE_CAPTURES: u32 = 4;

fn emit_render_smoke_test(link: &Arc<LiveLink>, session_idx: u64) {
    for run in 0..DETERMINISM_RUNS {
        let seed = 0xC0FFEE_0000_0000_u64 ^ (session_idx as u64) ^ (run as u64);
        let frame_idx = run;
        let diverges = run == 3;
        let aggregate = if diverges { DIVERGENT_AGGREGATE } else { BASELINE_AGGREGATE };
        link.record_determinism_run(
            run,
            seed,
            frame_idx,
            DETERMINISM_BUFFER_CAPTURES,
            DETERMINISM_IMAGE_CAPTURES,
            aggregate,
            !diverges,
            "render_smoke_test",
        );

        if diverges {
            let bitmap_path = format!(
                "ignis_diff_session{}_run{}_shadow_atlas.bmp",
                session_idx, run);
            link.record_determinism_divergence(
                run,
                1,
                BASELINE_AGGREGATE,
                DIVERGENT_AGGREGATE,
                4096, 4096,
                "shadow_atlas",
                &bitmap_path,
            );
        }

        std::thread::sleep(Duration::from_millis(40));
    }
}

// Clean session: every run matches baseline, no divergences. Useful for
// demonstrating the all-green case in the determinism table.
fn emit_physics_smoke_test(link: &Arc<LiveLink>, session_idx: u64) {
    for run in 0..DETERMINISM_RUNS {
        let seed = 0xBAD_FEED_0000_0000_u64 ^ (session_idx as u64) ^ (run as u64);
        link.record_determinism_run(
            run,
            seed,
            run,
            8,
            2,
            PHYSICS_AGGREGATE,
            true,
            "physics_smoke_test",
        );
        std::thread::sleep(Duration::from_millis(35));
    }
}

// ---- Node label updates ------------------------------------------------

fn emit_node_label_updates(link: &Arc<LiveLink>, cycle: u64) {
    for &(node_id, names) in NODE_LABEL_UPDATES {
        let name = names[(cycle as usize) % names.len()];
        let kind = CORE_NODES.iter()
            .find(|(id, _, _)| *id == node_id)
            .map(|(_, k, _)| *k)
            .unwrap_or(NODE_KIND_PASS);
        link.record_node(node_id, kind, name);
    }
}

// ---- Main loop ---------------------------------------------------------

fn run_event_loop(link: &Arc<LiveLink>, mem: &[u64; 4]) {
    let pass_ids: &[(u32, &str, u64)] = &[
        (1,  "vertex_skin",      80_000),
        (2,  "shadow_cascade",   600_000),
        (3,  "geometry_main",    1_200_000),
        (9,  "lighting_deferred", 800_000),
        (11, "tonemap",          150_000),
        (13, "ui_compose",       120_000),
    ];
    let optional_ssao_passes: &[(u32, &str, u64)] = &[
        (20, "ssao_compute", 350_000),
        (22, "ssao_blur",    180_000),
    ];
    let optional_bloom_passes: &[(u32, &str, u64)] = &[
        (30, "bloom_downsample", 220_000),
        (32, "bloom_upsample",   240_000),
    ];

    let core_pulse_edges: &[(u32, u32)] = CORE_EDGES;
    let mut frame: u64 = 0;
    let mut ssao_visible = false;
    let mut bloom_visible = false;
    let mut transient: Vec<(u64, u64, u64, String)> = Vec::new();
    let res = synthetic_handles();
    let mut transient_objects: Vec<TransientObject> = Vec::new();
    let mut next_object_handle: u64 = 0;
    let mut alloc_site_epoch: u64 = 0;
    let mut alloc_site_total: u64 = 0;let mut printf_total: u64 = 0;
    let mut hang_total: u64 = 0;
    let mut fault_total: u64 = 0;
    let mut object_total: u64 = 0;
    let mut descriptor_issues_total: u64 = 0;
    let mut aliasing_total: u64 = 0;
    let mut pipeline_issues_total: u64 = 0;
    let mut hang_idx: u64 = 0;
    let mut fault_idx: u64 = 0;
    let mut descriptor_idx: u64 = 0;
    let mut aliasing_idx: u64 = 0;
    let mut pipeline_idx: u64 = 0;
    let mut alloc_site_epoch: u64 = 0;
    let mut alloc_site_total: u64 = 0;
    let mut vl_cursor: usize = 0;
    let mut vl_info_cursor: usize = 0;
    let mut vl_total: u64 = 0;
    let mut pstats_total: u64 = 0;
    let mut budget_total: u64 = 0;
    let mut gpu_total: u64 = 0;
    let mut sync_total: u64 = 0;
    let mut canary_total: u64 = 0;
    let mut hstats_total: u64 = 0;
    let mut det_total: u64 = 0;
    let mut det_divergences: u64 = 0;
    let mut canary_cursor: usize = 0;
    let mut det_session_idx: u64 = 0;
    let mut clean_det_session_idx: u64 = 0;
    let mut rename_cycle: u64 = 0;
    let mut label_update_cycle: u64 = 0;

    let start = Instant::now();
    let mut last_ssao_toggle = start;
    let mut last_bloom_toggle = start;
    let mut last_sync_emit = start;
    let mut last_canary_emit = start;
    let mut last_hstats_emit = start;
    let mut last_det_session = start;
    let mut last_clean_det_session = start;
    let mut last_vl_info_emit = start;
    let mut last_resource_rename = start;
    let mut last_label_update = start;
    let mut last_burst_start = start;
    let mut last_hang = start;
    let mut last_fault = start;
    let mut last_descriptor_issue = start;
    let mut last_aliasing = start;
    let mut last_pipeline_issue = start;
    let mut last_alloc_site_emit = start;

    // Burst window state. When `burst_active`, event rates are tripled
    // for a fixed duration. The viewer's stats sparklines should show
    // visible spikes synchronized with the burst windows.
    let mut burst_active = false;
    let mut burst_started_at = start;

    loop {
        frame += 1;
        link.heartbeat();
        let elapsed = start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // Burst mode: active for 12 seconds every 90 seconds.
        if !burst_active && last_burst_start.elapsed() >= Duration::from_secs(90) {
            burst_active = true;
            burst_started_at = Instant::now();
            last_burst_start = Instant::now();
            println!("\n  >> BURST MODE ON for 12s (3x event rate)");
        }
        if burst_active && burst_started_at.elapsed() >= Duration::from_secs(12) {
            burst_active = false;
            println!("\n  >> burst mode off");
        }

        if last_ssao_toggle.elapsed() >= Duration::from_secs(15) {
            last_ssao_toggle = Instant::now();
            ssao_visible = !ssao_visible;
            if ssao_visible {
                for &(id, k, l) in SSAO_NODES { link.record_node(id, k, l); }
                for &(a, b) in SSAO_EDGES { link.record_edge(a, b); }
                println!("\n  >> SSAO subgraph appeared (watch the layout animation)");
            } else {
                for &(id, _, _) in SSAO_NODES { link.record_node_remove(id); }
                println!("\n  >> SSAO subgraph removed (watch the layout animation)");
            }
        }

        if last_bloom_toggle.elapsed() >= Duration::from_secs(22) {
            last_bloom_toggle = Instant::now();
            bloom_visible = !bloom_visible;
            if bloom_visible {
                for &(id, k, l) in BLOOM_NODES { link.record_node(id, k, l); }
                for &(a, b) in BLOOM_EDGES { link.record_edge(a, b); }
                println!("\n  >> bloom subgraph appeared (watch the layout animation)");
            } else {
                for &(id, _, _) in BLOOM_NODES { link.record_node_remove(id); }
                println!("\n  >> bloom subgraph removed (watch the layout animation)");
            }
        }

        // Periodic node label updates. Tests the re-registration path
        // where a node id stays the same but its label changes.
        if last_label_update.elapsed() >= Duration::from_secs(60) {
            last_label_update = Instant::now();
            label_update_cycle += 1;
            emit_node_label_updates(link, label_update_cycle);
            println!("\n  >> updated labels on {} nodes",
                NODE_LABEL_UPDATES.len());
        }

        // Periodic resource renaming.
        if last_resource_rename.elapsed() >= Duration::from_secs(90) {
            last_resource_rename = Instant::now();
            rename_cycle += 1;
            rename_resources(link, rename_cycle, mem);
            println!("\n  >> renamed resources (cycle {})", rename_cycle);
        }

        // Pass durations. Burst mode does not affect per-pass duration
        // here because the workload variation is already significant;
        // it only shifts pipeline stats to make the spike visible.
        for &(id, name, base) in pass_ids {
            let jitter = (frame * 17 + id as u64 * 31) % 400_000;
            let dur = base + jitter;
            link.record_pass(id, name, dur);
        }
        if ssao_visible {
            for &(id, name, base) in optional_ssao_passes {
                let dur = base + ((frame * 13) % 200_000);
                link.record_pass(id, name, dur);
            }
        }
        if bloom_visible {
            for &(id, name, base) in optional_bloom_passes {
                let dur = base + ((frame * 11) % 150_000);
                link.record_pass(id, name, dur);
            }
        }

        for &(from, to) in core_pulse_edges {
            let active = ((frame as i64 + from as i64 * 7 + to as i64 * 3) % 24) < 5;
            link.record_edge_toggle(from, to, active);
        }

        // Submission rates. Burst mode emits an extra mid-frame submit
        // to bias the cadence and exercise the frame inference logic.
        if frame % 16 == 0 {
            link.record_submission(0, 0, "graphics_main",
                1_200_000 + (frame * 31) % 2_500_000);
        }
        if burst_active && frame % 8 == 0 {
            link.record_submission(0, 0, "graphics_burst_aux",
                400_000 + (frame * 19) % 800_000);
        }
        if frame % 24 == 0 {
            link.record_submission(1, 0, "compute_async",
                400_000 + (frame * 17) % 1_500_000);
        }
        if frame % 40 == 0 {
            link.record_submission(2, 0, "transfer_upload",
                200_000 + (frame * 13) % 800_000);
        }

        // Transient allocations. During burst mode we churn faster.
        let alloc_cadence = if burst_active { 8 } else { 18 };
        let max_transient = if burst_active { 16 } else { 12 };
        if frame % alloc_cadence == 0 && transient.len() < max_transient {
            let mem_handle = mem[2];
            let offset = 20 * 1024 * 1024 + (transient.len() as u64) * 512 * 1024;
            let size = 64 * 1024 + (frame * 7) % (256 * 1024);
            let site = format!("frame::scratch_{:03}", transient.len());
            link.record_allocation(mem_handle, offset, size, &site);
            transient.push((mem_handle, offset, size, site));
        }
        let free_cadence = if burst_active { 30 } else { 90 };
        if frame % free_cadence == 0 && !transient.is_empty() {
            let (m, o, s, _) = transient.remove(0);
            link.record_free(m, o, s);
        }

        if frame % 200 == 0 {
            let off = 2 * 1024 * 1024 + ((frame / 200) % 4) * 256 * 1024;
            link.record_allocation(mem[3], off, 256 * 1024,
                "descriptors::transient_set");
        }
        if frame % 250 == 0 {
            let off = 2 * 1024 * 1024 + ((frame / 250 - 1) % 4) * 256 * 1024;
            link.record_free(mem[3], off, 256 * 1024);
        }

        if frame % 40 == 0 {
            emit_pipeline_stats(link, frame, ssao_visible, bloom_visible, burst_active);
            let mut n: u64 = 5;
            if ssao_visible { n += 2; }
            if bloom_visible { n += 2; }
            pstats_total += n;
        }

        if frame % 50 == 0 {
            emit_budget(link, frame, transient.len(), burst_active);
            budget_total += 4;
        }

        if frame % 8 == 0 {
            emit_gpu_timeline(link, frame, ssao_visible, bloom_visible,
                elapsed_ns, burst_active);
            let mut n: u64 = 6;
            if ssao_visible { n += 2; }
            if bloom_visible { n += 2; }
            if frame % 4 == 0 { n += 1; }
            gpu_total += n;
        }

        // Sync DAG cycle / orphan emission at 1Hz.
        if last_sync_emit.elapsed() >= Duration::from_secs(1) {
            last_sync_emit = Instant::now();
            let secs = elapsed.as_secs();
            let was_active_cycle = secs % 60 < 10;
            let was_active_orphan = secs % 90 < 8;
            emit_sync_marks(link, secs);
            if was_active_cycle { sync_total += 2; }
            if was_active_orphan { sync_total += 1; }
        }

        // Canary corruption events on a 45s schedule.
        if last_canary_emit.elapsed() >= Duration::from_secs(45) {
            last_canary_emit = Instant::now();
            let sample = &CANARY_SAMPLES[canary_cursor];
            canary_cursor = (canary_cursor + 1) % CANARY_SAMPLES.len();
            emit_canary(link, sample, &res, mem);
            canary_total += 1;
            let region = if sample.region == 0 { "front" } else { "back" };
            let sev = match sample.severity {
                2 => "ERROR",
                1 => "warn",
                _ => "info",
            };
            println!(
                "\n  >> canary corruption [{}] {} guard, {} bytes corrupted",
                sev, region, sample.corrupted_count,
            );
        }

        // Hardened allocator stats every 3s.
        if last_hstats_emit.elapsed() >= Duration::from_secs(3) {
            last_hstats_emit = Instant::now();
            emit_hardened_stats(link, elapsed, canary_total);
            hstats_total += 1;
        }

        // Determinism: divergent session every 2 minutes.
        if last_det_session.elapsed() >= Duration::from_secs(120) {
            last_det_session = Instant::now();
            det_session_idx += 1;
            println!(
                "\n  >> render_smoke_test session #{} (5 runs, run #3 diverges)",
                det_session_idx,
            );
            emit_render_smoke_test(link, det_session_idx);
            det_total += DETERMINISM_RUNS as u64;
            det_divergences += 1;
        }

        // Determinism: clean session every 90s.
        if last_clean_det_session.elapsed() >= Duration::from_secs(90) {
            last_clean_det_session = Instant::now();
            clean_det_session_idx += 1;
            println!(
                "\n  >> physics_smoke_test session #{} (5 clean runs)",
                clean_det_session_idx,
            );
            emit_physics_smoke_test(link, clean_det_session_idx);
            det_total += DETERMINISM_RUNS as u64;
        }

        // High-frequency shader printf to populate the dedicated view.
        // Six samples per second exercises the dedup path nicely.
        let printf_cadence = if burst_active { 3 } else { 5 };
        if frame % printf_cadence == 0 {
            emit_shader_printf(link, frame);
            printf_total += 1;
        }

        // Object lifetime churn: register a new transient every 50
        // frames, age existing ones, destroy any older than ~600
        // frames. Permanent objects (registered at startup) accumulate
        // as leaks the user can spot in the Live and Orphans modes.
        if frame % 50 == 0 {
            churn_transient_object(link, &mut transient_objects,
                &mut next_object_handle, frame);
            object_total += 1;
        }
        if frame % 30 == 0 {
            destroy_old_transients(link, &mut transient_objects);
        }

        // Hang detection: rare but dramatic. Every 180s simulate a
        // hang event with a full breadcrumb trail.
        if last_hang.elapsed() >= Duration::from_secs(180) {
            last_hang = Instant::now();
            hang_idx += 1;
            emit_hang_event(link, hang_idx);
            hang_total += 1;
            println!("\n  >> SIMULATED HANG #{} (5.3s timeout, breadcrumb \
                trail of 9 entries)", hang_idx);
        }

        // Device fault: very rare. Every 300s simulate a fault snapshot.
        if last_fault.elapsed() >= Duration::from_secs(300) {
            last_fault = Instant::now();
            fault_idx += 1;
            emit_device_fault(link, fault_idx);
            fault_total += 1;
            println!("\n  >> SIMULATED DEVICE FAULT #{} (synthetic \
                vendor description streamed via continuation chunks)",
                fault_idx);
        }

        // Descriptor audit issues every 60s.
        if last_descriptor_issue.elapsed() >= Duration::from_secs(60) {
            last_descriptor_issue = Instant::now();
            emit_descriptor_issue(link, descriptor_idx, &res);
            descriptor_idx += 1;
            descriptor_issues_total += 1;
        }

        // Aliasing conflicts every 45s.
        if last_aliasing.elapsed() >= Duration::from_secs(45) {
            last_aliasing = Instant::now();
            emit_aliasing_conflict(link, aliasing_idx, &res);
            aliasing_idx += 1;
            aliasing_total += 1;
        }

        // Pipeline audit issues every 70s.
        if last_pipeline_issue.elapsed() >= Duration::from_secs(70) {
            last_pipeline_issue = Instant::now();
            emit_pipeline_issue(link, pipeline_idx);
            pipeline_idx += 1;
            pipeline_issues_total += 1;
        }

        // Allocation profiler snapshot every 1.5s. Each snapshot is one
        // batch of records sharing the same epoch; the viewer atomically
        // swaps in the new batch when the epoch advances. The cadence
        // matches what a real AllocationProfiler would produce when
        // bridged via bridge_alloc_profiler_to_live_link.
        if last_alloc_site_emit.elapsed() >= Duration::from_millis(1500) {
            last_alloc_site_emit = Instant::now();
            alloc_site_epoch += 1;
            emit_alloc_site_snapshot(link, alloc_site_epoch);
            alloc_site_total += ALLOC_SITES.len() as u64;
        }

        // Validation events: warning/info mix on the main cadence.
        let vl_cadence = if burst_active { 12 } else { 25 };
        if frame % vl_cadence == 0 {
            let mut tries = 0;
            while tries < VL_SAMPLES.len() {
                let s = &VL_SAMPLES[vl_cursor];
                vl_cursor = (vl_cursor + 1) % VL_SAMPLES.len();
                if s.severity != VAL_SEVERITY_ERROR {
                    let handle = handle_for_slot(&res, s.object_slot);
                    let object_type = object_type_for_slot(s.object_slot);
                    link.record_validation(
                        s.severity, s.node_id,
                        s.function, s.vuid, s.message,
                        object_type, handle,
                    );
                    vl_total += 1;
                    break;
                }
                tries += 1;
            }
        }
        // Validation errors every ~6s.
        if frame % 180 == 0 {
            for _ in 0..VL_SAMPLES.len() {
                let s = &VL_SAMPLES[vl_cursor];
                vl_cursor = (vl_cursor + 1) % VL_SAMPLES.len();
                if s.severity == VAL_SEVERITY_ERROR {
                    let handle = handle_for_slot(&res, s.object_slot);
                    let object_type = object_type_for_slot(s.object_slot);
                    link.record_validation(
                        s.severity, s.node_id,
                        s.function, s.vuid, s.message,
                        object_type, handle,
                    );
                    vl_total += 1;
                    break;
                }
            }
        }

        // Dedicated info-level VL stream so the Info filter chip is
        // populated regardless of the warning/error cadence above.
        if last_vl_info_emit.elapsed() >= Duration::from_secs(30) {
            last_vl_info_emit = Instant::now();
            let s = &VL_INFO_SAMPLES[vl_info_cursor];
            vl_info_cursor = (vl_info_cursor + 1) % VL_INFO_SAMPLES.len();
            link.record_validation(
                s.severity, s.node_id,
                s.function, s.vuid, s.message,
                0, 0,
            );
            vl_total += 1;
        }

        // Periodic burst of multiple VL events at once. Tests viewer
        // backpressure handling.
        if frame % 600 == 0 {
            let burst: &[VlSample] = &[
                VlSample {
                    severity: VAL_SEVERITY_WARNING,
                    node_id: 9,
                    function: "vkCmdDispatch",
                    vuid: "UNASSIGNED-BestPractices-Workgroup-Underutilization",
                    message: "Workgroup of 1024 threads dispatched but only \
                              42 are active. Compact or reduce local size.",
                    object_slot: 0,
                },
                VlSample {
                    severity: VAL_SEVERITY_WARNING,
                    node_id: 9,
                    function: "vkCmdDispatch",
                    vuid: "UNASSIGNED-BestPractices-pipeline-stall",
                    message: "Compute dispatch waits on prior fragment \
                              shader. Consider async compute queue.",
                    object_slot: 0,
                },
                VlSample {
                    severity: VAL_SEVERITY_WARNING,
                    node_id: 0,
                    function: "vkQueueSubmit",
                    vuid: "UNASSIGNED-BestPractices-Submission-ReducedBatching",
                    message: "Three submits within 0.4 ms. Batch into one \
                              vkQueueSubmit call to reduce kernel transitions.",
                    object_slot: 0,
                },
                VlSample {
                    severity: VAL_SEVERITY_INFO,
                    node_id: 0,
                    function: "vkAllocateCommandBuffers",
                    vuid: "UNASSIGNED-Best-Practices-CommandBuffer-Allocation",
                    message: "Allocated 32 command buffers in one call. \
                              Reasonable batching for current workload.",
                    object_slot: 0,
                },
            ];
            for s in burst {
                link.record_validation(
                    s.severity, s.node_id,
                    s.function, s.vuid, s.message,
                    0, 0,
                );
                vl_total += 1;
            }
        }

        // Status line every 100 frames.
        if frame % 100 == 0 {
            let ssao = if ssao_visible { "on " } else { "off" };
            let bloom = if bloom_visible { "on " } else { "off" };
            let burst = if burst_active { " BURST" } else { "      " };
            print!(
                "\r  uptime: {:>4}s  frames: {:>7}  ssao={} bloom={}{} \
                 transient={:>2}  vl={:>5}  pstats={:>5}  budget={:>5}  \
                 gpu={:>5}  sync={:>4}  canary={:>3}  hstats={:>4}  \
                 det={:>3} ({}+{} sessions)  printf={:>6}  obj={:>4} \
                 (live={:>2})  desc={:>3}  alias={:>3}  pipe={:>3}  \
                 hang={:>2}  fault={:>2}  sites={:>5} (epoch={:>3})",
                start.elapsed().as_secs(), frame, ssao, bloom, burst,
                transient.len(), vl_total, pstats_total, budget_total,
                gpu_total, sync_total, canary_total, hstats_total,
                det_total, det_divergences, clean_det_session_idx,
                printf_total,
                object_total,
                transient_objects.len() + PERMANENT_OBJECTS.len(),
                descriptor_issues_total,
                aliasing_total,
                pipeline_issues_total,
                hang_total,
                fault_total,
                alloc_site_total,
                alloc_site_epoch,
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }

        std::thread::sleep(Duration::from_millis(33));
    }
}