//! Panel renderers: memory layout and resource timeline.
//!
//! Each renderer takes a snapshot of its data source and a target
//! [`Framebuffer`]. Renderers know nothing about Vulkan; they only paint
//! into CPU memory.
//!
//! The timeline panel iterates `ResourceTrace` in place via `for_each`
//! to avoid allocating a fresh Vec of TraceEvent (each containing a
//! String) every frame. At 6000 live events × 60 fps that would burn
//! ~360k heap allocations per second purely for visualization.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk::Handle;

use super::raster::{palette, Color, Framebuffer};
use crate::resource_trace::{ResourceTrace, TraceEventKind};
use crate::AllocationProfiler;

/// Render the memory layout panel into the framebuffer at (x, y, w, h).
pub fn render_memory_panel(
    fb: &mut Framebuffer,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    profiler: &Arc<AllocationProfiler>,
) {
    fb.rect(x, y, w, h, palette::PANEL_BG);
    fb.rect_outline(x, y, w, h, palette::FRAME);

    // Header.
    let allocs = profiler.live_allocations();
    let total_bytes: u64 = allocs.iter().map(|a| a.size).sum();
    fb.text(x + 8, y + 6, "MEMORY LAYOUT", palette::TEXT_HEAD);
    let header = format!(
        "{} blocks   {} live allocs   {}",
        count_unique_blocks(&allocs),
        allocs.len(),
        format_bytes(total_bytes),
    );
    fb.text(x + 8, y + 18, &header, palette::TEXT_DIM);

    // Group by memory handle.
    let blocks = group_by_memory(&allocs);

    let body_top = y + 36;
    let body_bottom = y + h - 8;
    let row_height = 36;

    // Adapt label width to panel width: at the default 1280px window we
    // want ~220px (about 27 chars), at very wide windows up to 480px
    // (about 60 chars), enough to show a full VkDeviceMemory hex handle
    // plus the surrounding "VkDeviceMemory(...)" text.
    let label_width = (w / 4).clamp(220, 480);
    let max_label_chars = ((label_width / 8).max(8)) as usize;

    let bar_left = x + 8 + label_width + 8;
    let bar_right = x + w - 8;
    let bar_width = (bar_right - bar_left).max(0);

    let mut row_y = body_top;
    for block in &blocks {
        if row_y + row_height > body_bottom {
            fb.text(
                x + 8,
                body_bottom - 10,
                "(more blocks not shown)",
                palette::TEXT_DIM,
            );
            break;
        }

        let label = format!("VkDeviceMemory({:#x})", block.memory_raw);
        fb.text(x + 8, row_y, &short(&label, max_label_chars), palette::TEXT);
        let info = format!(
            "type {} | {} allocs | {}",
            block.memory_type_index,
            block.allocs.len(),
            format_bytes(block.max_extent),
        );
        fb.text(x + 8, row_y + 12, &short(&info, max_label_chars), palette::TEXT_DIM);

        let bar_y = row_y + 4;
        let bar_h = 22;
        fb.rect(bar_left, bar_y, bar_width, bar_h, palette::BAR_BG);
        fb.rect_outline(bar_left, bar_y, bar_width, bar_h, palette::FRAME);

        let denom = block.max_extent.max(1) as i64;
        for a in &block.allocs {
            let off_px = ((a.offset as i64) * (bar_width as i64) / denom) as i32;
            let mut w_px = ((a.size as i64) * (bar_width as i64) / denom) as i32;
            if w_px < 1 {
                w_px = 1;
            }
            let color = palette::ALLOC_COLORS
                [(block.memory_type_index as usize) % palette::ALLOC_COLORS.len()];
            fb.rect(bar_left + off_px, bar_y + 1, w_px, bar_h - 2, color);
        }

        row_y += row_height;
    }

    if blocks.is_empty() {
        fb.text(x + 8, body_top + 8, "(no live allocations)", palette::TEXT_DIM);
    }
}

/// Render the resource timeline panel.
pub fn render_timeline_panel(
    fb: &mut Framebuffer,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    trace: &Arc<ResourceTrace>,
    time_window_ns: u64,
) {
    fb.rect(x, y, w, h, palette::PANEL_BG);
    fb.rect_outline(x, y, w, h, palette::FRAME);

    let stats = trace.stats();
    fb.text(x + 8, y + 6, "RESOURCE TIMELINE", palette::TEXT_HEAD);
    let header = format!(
        "alloc={} free={} submit={} pass={} trans={} custom={}   total={} (lifetime {})",
        stats.allocations,
        stats.frees,
        stats.submissions,
        stats.passes,
        stats.transitions,
        stats.custom,
        stats.total,
        stats.lifetime_total,
    );
    fb.text(x + 8, y + 18, &short(&header, ((w - 16) / 8) as usize), palette::TEXT_DIM);

    const LANES: &[(&str, Color)] = &[
        ("alloc", palette::EVT_ALLOC),
        ("free", palette::EVT_FREE),
        ("submit", palette::EVT_SUBMIT),
        ("transition", palette::EVT_TRANSITION),
        ("pass", palette::EVT_PASS),
        ("custom", palette::EVT_CUSTOM),
    ];

    let body_top = y + 38;
    let body_bottom = y + h - 8;
    let lane_height = ((body_bottom - body_top) / LANES.len() as i32).max(12);
    let track_left = x + 8 + 90;
    let track_right = x + w - 8;
    let track_width = (track_right - track_left).max(0);

    for (i, (label, color)) in LANES.iter().enumerate() {
        let ly = body_top + (i as i32) * lane_height;
        fb.text(x + 8, ly + 4, label, *color);
        fb.rect(track_left, ly + 4, track_width, lane_height - 8, palette::BAR_BG);
    }

    if track_width <= 0 {
        return;
    }

    // Sample the trace's "now" once so all events on this frame use a
    // stable window. for_each iterates under the trace's internal lock
    // so no clones are made.
    let now = trace.now_ns();
    let window_start = now.saturating_sub(time_window_ns);
    let window_span = time_window_ns.max(1) as f64;
    let track_w_f = track_width as f64;

    trace.for_each(|event| {
        if event.timestamp_ns < window_start {
            return;
        }
        let lane_idx = match event.kind {
            TraceEventKind::Allocation { .. } => 0,
            TraceEventKind::Free { .. } => 1,
            TraceEventKind::Submission { .. } => 2,
            TraceEventKind::Transition { .. } => 3,
            TraceEventKind::Pass { .. } => 4,
            TraceEventKind::Custom { .. } => 5,
        };
        let ly = body_top + (lane_idx as i32) * lane_height;
        let lane_color = LANES[lane_idx].1;

        let t_norm = ((event.timestamp_ns - window_start) as f64 / window_span)
            .clamp(0.0, 1.0);
        let ex = track_left + (t_norm * track_w_f) as i32;

        let dur_ns = match &event.kind {
            TraceEventKind::Submission { duration_ns, .. }
            | TraceEventKind::Pass { duration_ns, .. } => *duration_ns,
            _ => 0,
        };
        if dur_ns > 0 {
            let bar_w = ((dur_ns as f64 / window_span) * track_w_f).max(2.0) as i32;
            fb.rect(ex, ly + 6, bar_w.min(track_width), lane_height - 12, lane_color);
        } else {
            fb.rect(ex, ly + 6, 2, lane_height - 12, lane_color);
        }
    });

    let time_label = format!("window: last {}", format_duration_ns(time_window_ns));
    let tw = Framebuffer::text_width(&time_label);
    fb.text(track_right - tw, body_bottom - 8, &time_label, palette::TEXT_DIM);
}

struct BlockInfo {
    memory_raw: u64,
    memory_type_index: u32,
    allocs: Vec<crate::LiveAllocation>,
    max_extent: u64,
}

fn group_by_memory(allocs: &[crate::LiveAllocation]) -> Vec<BlockInfo> {
    let mut blocks: HashMap<u64, BlockInfo> = HashMap::new();
    for a in allocs {
        let raw = a.memory.as_raw();
        let entry = blocks.entry(raw).or_insert_with(|| BlockInfo {
            memory_raw: raw,
            memory_type_index: a.memory_type_index,
            allocs: Vec::new(),
            max_extent: 0,
        });
        let extent = a.offset + a.size;
        if extent > entry.max_extent {
            entry.max_extent = extent;
        }
        entry.allocs.push(a.clone());
    }
    let mut list: Vec<BlockInfo> = blocks.into_values().collect();
    for b in list.iter_mut() {
        b.allocs.sort_by_key(|a| a.offset);
    }
    list.sort_by_key(|b| b.memory_raw);
    list
}

fn count_unique_blocks(allocs: &[crate::LiveAllocation]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for a in allocs {
        seen.insert(a.memory.as_raw());
    }
    seen.len()
}

fn short(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(2)).collect();
        format!("{truncated}..")
    }
}

fn format_bytes(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", b as f64 / 1_073_741_824.0)
    } else if b >= 1024 * 1024 {
        format!("{:.1} MiB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

fn format_duration_ns(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.1}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{}ms", ns / 1_000_000)
    } else if ns >= 1_000 {
        format!("{}us", ns / 1_000)
    } else {
        format!("{}ns", ns)
    }
}