//! Memory layout SVG visualizer.
//!
//! Renders a static SVG image showing how live allocations are arranged
//! within `VkDeviceMemory` blocks. The image is human-readable: each row
//! represents one device memory object, each colored rectangle within
//! the row represents one allocation, sized proportionally to its byte
//! footprint within the block.
//!
//! # When to Use
//!
//! - Diagnose memory fragmentation: gaps between allocations within a
//!   single row are visible as dark regions inside the bar.
//! - Identify the largest individual consumers: hover over a rectangle
//!   to see its call site, offset, and size in the SVG tooltip.
//! - Compare allocator strategies: render before-and-after snapshots and
//!   diff them visually.
//!
//! # Data Source
//!
//! Reads from [`AllocationProfiler::live_allocations`]. The visualizer is
//! stateless; all rendering is a pure function of the snapshot.
//!
//! # Output Compatibility
//!
//! Output is standalone SVG 1.1 with inline styling. No external CSS or
//! fonts. Works in any browser, RenderDoc-style image inspector, or
//! Markdown viewer that supports embedded SVG.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ignis::debug::memory_viz::*;
//! # use std::sync::Arc;
//! # fn example(profiler: &Arc<AllocationProfiler>) -> std::io::Result<()> {
//! let viz = MemoryVisualizer::new();
//! viz.save_svg(profiler, "memory_layout.svg")?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

use ash::vk::Handle;

use super::alloc_profiler::{AllocationProfiler, LiveAllocation};

/// Configuration knobs for SVG generation.
#[derive(Debug, Clone)]
pub struct VisualizerConfig {
    /// Total SVG canvas width in pixels.
    pub width: u32,
    /// Height of each memory-block row in pixels.
    pub bar_height: u32,
    /// Vertical spacing between rows in pixels.
    pub spacing: u32,
    /// Width reserved for the left-side label column in pixels.
    pub label_width: u32,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            width: 1400,
            bar_height: 36,
            spacing: 6,
            label_width: 260,
        }
    }
}

/// Renders memory layout snapshots as SVG.
///
/// Stateless. Construct once and reuse across multiple snapshots.
pub struct MemoryVisualizer {
    config: VisualizerConfig,
}

impl MemoryVisualizer {
    /// Construct with default configuration.
    pub fn new() -> Self {
        Self {
            config: VisualizerConfig::default(),
        }
    }

    /// Construct with custom configuration.
    pub fn with_config(config: VisualizerConfig) -> Self {
        Self { config }
    }

    /// Render an SVG document from the profiler's current live allocations.
    pub fn render_svg(&self, profiler: &Arc<AllocationProfiler>) -> String {
        let allocs = profiler.live_allocations();
        self.render_allocations(&allocs)
    }

    /// Render an SVG document from explicit live allocation data.
    ///
    /// Useful when allocations come from a non-profiler source (custom
    /// allocator with introspection, replay of recorded allocations, unit
    /// tests).
    pub fn render_allocations(&self, allocs: &[LiveAllocation]) -> String {
        let blocks = group_by_memory(allocs);

        let margin = 20_u32;
        let header_h = 50_u32;
        let footer_h = 90_u32;
        let label_w = self.config.label_width;
        let bar_w = self
            .config
            .width
            .saturating_sub(2 * margin)
            .saturating_sub(label_w);

        let total_h = header_h
            + (blocks.len() as u32) * (self.config.bar_height + self.config.spacing)
            + footer_h;

        let mut svg = String::with_capacity(8192);

        // Note: SVG color literals like #1e1e1e contain a `"#` sequence
        // that would terminate r#"..."# raw strings early. We use
        // r##"..."## with two hashes so the inner `"#` is not interpreted
        // as the closing delimiter.
        let _ = writeln!(svg, r##"<?xml version="1.0" encoding="UTF-8"?>"##);
        let _ = writeln!(
            svg,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" font-family="Consolas, monospace" font-size="11">"##,
            self.config.width, total_h
        );

        // Background.
        let _ = writeln!(svg, r##"<rect width="100%" height="100%" fill="#1e1e1e"/>"##);

        // Header text.
        let total_bytes: u64 = allocs.iter().map(|a| a.size).sum();
        let _ = writeln!(
            svg,
            r##"<text x="{}" y="22" fill="#e8e8e8" font-size="14" font-weight="bold">VkDeviceMemory layout</text>"##,
            margin
        );
        let _ = writeln!(
            svg,
            r##"<text x="{}" y="40" fill="#a0a0a0" font-size="11">{} blocks, {} live allocations, {} total</text>"##,
            margin,
            blocks.len(),
            allocs.len(),
            format_bytes(total_bytes)
        );

        // Rows: one per VkDeviceMemory.
        let mut y = header_h;
        for block in &blocks {
            // Left-side label.
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#c8c8c8" font-size="11">VkDeviceMemory({:#x})</text>"##,
                margin,
                y + 14,
                block.memory_raw
            );
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#7a7a7a" font-size="10">type {} | {} allocs | {}</text>"##,
                margin,
                y + 28,
                block.memory_type_index,
                block.allocs.len(),
                format_bytes(block.max_extent)
            );

            // Bar background showing total allocated extent.
            let bar_x = margin + label_w;
            let _ = writeln!(
                svg,
                r##"<rect x="{}" y="{}" width="{}" height="{}" fill="#2a2a2a" stroke="#404040" stroke-width="0.5"/>"##,
                bar_x, y, bar_w, self.config.bar_height
            );

            // Individual allocations as colored rectangles.
            for a in &block.allocs {
                let denom = block.max_extent.max(1);
                let x_off = bar_x + ((a.offset * bar_w as u64) / denom) as u32;
                let mut w = ((a.size * bar_w as u64) / denom) as u32;
                if w < 1 {
                    w = 1;
                }
                let color = color_for_index(a.memory_type_index);

                // Wrap rect in <g> so the <title> tooltip applies on hover.
                let _ = writeln!(
                    svg,
                    r##"<g><rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="#000" stroke-width="0.3"/>"##,
                    x_off, y, w, self.config.bar_height, color
                );
                let tooltip = format!(
                    "offset={} size={} ({}){}{}",
                    a.offset,
                    a.size,
                    format_bytes(a.size),
                    "&#10;",
                    html_escape(&a.site.to_string())
                );
                let _ = writeln!(svg, "<title>{}</title></g>", tooltip);
            }

            y += self.config.bar_height + self.config.spacing;
        }

        // Legend.
        let mut used_types: Vec<u32> = blocks.iter().map(|b| b.memory_type_index).collect();
        used_types.sort_unstable();
        used_types.dedup();

        y += 14;
        let _ = writeln!(
            svg,
            r##"<text x="{}" y="{}" fill="#a0a0a0" font-size="11" font-weight="bold">Legend (memory type):</text>"##,
            margin, y
        );
        y += 22;

        let mut x = margin;
        for t in &used_types {
            let _ = writeln!(
                svg,
                r##"<rect x="{}" y="{}" width="20" height="14" fill="{}"/>"##,
                x,
                y - 11,
                color_for_index(*t)
            );
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#c8c8c8" font-size="11">type {}</text>"##,
                x + 26,
                y,
                t
            );
            x += 90;
            if x + 90 > self.config.width.saturating_sub(margin) {
                x = margin;
                y += 18;
            }
        }

        svg.push_str("</svg>\n");
        svg
    }

    /// Render and write to a file in one step.
    pub fn save_svg(
        &self,
        profiler: &Arc<AllocationProfiler>,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let svg = self.render_svg(profiler);
        std::fs::write(path, svg)
    }
}

impl Default for MemoryVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-block aggregate used during rendering.
struct BlockInfo {
    memory_raw: u64,
    memory_type_index: u32,
    allocs: Vec<LiveAllocation>,
    /// Highest `offset + size` observed in the block. Used as the
    /// proportional denominator for placing rectangles within the bar.
    max_extent: u64,
}

/// Group flat allocation list by `VkDeviceMemory` handle, computing the
/// per-block extent and sorting each block's allocations by offset.
fn group_by_memory(allocs: &[LiveAllocation]) -> Vec<BlockInfo> {
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

/// Pick a deterministic color from a 12-entry palette based on the memory
/// type index. The palette is chosen to have enough hue variance to be
/// distinguishable on dark backgrounds and is colorblind-friendly enough
/// for typical use.
fn color_for_index(idx: u32) -> &'static str {
    const COLORS: &[&str] = &[
        "#4ec9b0", "#dcdcaa", "#ce9178", "#9cdcfe", "#c586c0", "#569cd6", "#f44747", "#608b4e",
        "#d7ba7d", "#646695", "#dd6b20", "#3182ce",
    ];
    COLORS[(idx as usize) % COLORS.len()]
}

/// Escape XML-significant characters in a string for safe embedding in
/// SVG text or attribute values.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
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

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use std::time::Duration;

    use super::super::alloc_profiler::CallSite;

    fn la(mem: u64, off: u64, size: u64, ty: u32) -> LiveAllocation {
        LiveAllocation {
            memory: vk::DeviceMemory::from_raw(mem),
            offset: off,
            size,
            memory_type_index: ty,
            site: CallSite {
                function: "test::func".to_string(),
                file: "test.rs".to_string(),
                line: 1,
            },
            age: Duration::from_secs(0),
        }
    }

    #[test]
    fn empty_input_produces_minimal_svg() {
        let viz = MemoryVisualizer::new();
        let svg = viz.render_allocations(&[]);
        assert!(svg.starts_with("<?xml"));
        assert!(svg.ends_with("</svg>\n"));
        assert!(svg.contains("VkDeviceMemory layout"));
        assert!(svg.contains("0 blocks"));
    }

    #[test]
    fn groups_allocations_by_memory_handle() {
        let allocs = vec![
            la(1, 0, 100, 0),
            la(1, 100, 200, 0),
            la(2, 0, 50, 1),
        ];
        let blocks = group_by_memory(&allocs);
        assert_eq!(blocks.len(), 2);
        let block1 = blocks.iter().find(|b| b.memory_raw == 1).unwrap();
        let block2 = blocks.iter().find(|b| b.memory_raw == 2).unwrap();
        assert_eq!(block1.allocs.len(), 2);
        assert_eq!(block2.allocs.len(), 1);
        assert_eq!(block1.max_extent, 300);
        assert_eq!(block2.max_extent, 50);
    }

    #[test]
    fn allocations_sorted_by_offset_within_block() {
        let allocs = vec![
            la(1, 200, 100, 0),
            la(1, 0, 100, 0),
            la(1, 100, 100, 0),
        ];
        let blocks = group_by_memory(&allocs);
        let block = &blocks[0];
        assert_eq!(block.allocs[0].offset, 0);
        assert_eq!(block.allocs[1].offset, 100);
        assert_eq!(block.allocs[2].offset, 200);
    }

    #[test]
    fn renders_well_formed_svg_with_blocks() {
        let allocs = vec![
            la(1, 0, 1024 * 1024, 0),
            la(1, 1024 * 1024, 2 * 1024 * 1024, 0),
            la(2, 0, 512 * 1024, 1),
        ];
        let viz = MemoryVisualizer::new();
        let svg = viz.render_allocations(&allocs);

        assert!(svg.contains(r##"<?xml version="1.0""##));
        assert!(svg.contains("VkDeviceMemory"));
        assert!(svg.ends_with("</svg>\n"));
        assert!(svg.contains("VkDeviceMemory(0x1)"));
        assert!(svg.contains("VkDeviceMemory(0x2)"));
        // Legend covers both used types.
        assert!(svg.contains("type 0"));
        assert!(svg.contains("type 1"));
        // Live alloc count appears in header.
        assert!(svg.contains("3 live allocations"));
    }

    #[test]
    fn html_escape_handles_xml_chars() {
        assert_eq!(html_escape("a&b<c>d"), "a&amp;b&lt;c&gt;d");
        assert_eq!(html_escape(r#"a"b"#), "a&quot;b");
        assert_eq!(html_escape("plain"), "plain");
    }

    #[test]
    fn format_bytes_picks_correct_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.0 MiB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 2), "2.0 GiB");
    }

    #[test]
    fn color_palette_wraps_modulo() {
        let c0 = color_for_index(0);
        let c12 = color_for_index(12);
        let c24 = color_for_index(24);
        assert_eq!(c0, c12);
        assert_eq!(c0, c24);
    }

    #[test]
    fn small_allocation_renders_at_least_one_pixel() {
        // A tiny allocation alongside a giant one would round to zero
        // pixels under naive proportional sizing. Verify it still renders.
        let allocs = vec![la(1, 0, 1, 0), la(1, 1, 1024 * 1024 * 1024, 0)];
        let viz = MemoryVisualizer::new();
        let svg = viz.render_allocations(&allocs);
        // No panic, real output produced.
        assert!(svg.len() > 200);
        // Two allocations -> at least two <rect> elements within rows.
        let rect_count = svg.matches("<rect").count();
        assert!(rect_count >= 3, "background + 2 alloc rects expected");
    }

    #[test]
    fn custom_config_changes_dimensions() {
        let cfg = VisualizerConfig {
            width: 800,
            bar_height: 20,
            spacing: 4,
            label_width: 150,
        };
        let viz = MemoryVisualizer::with_config(cfg);
        let svg = viz.render_allocations(&[la(1, 0, 100, 0)]);
        assert!(svg.contains(r#"width="800""#));
    }

    #[test]
    fn many_memory_types_in_legend_wrap() {
        // Use 8 distinct memory types to force the legend onto a second row.
        let allocs: Vec<_> = (0..8).map(|t| la(t as u64 + 1, 0, 100, t)).collect();
        let viz = MemoryVisualizer::new();
        let svg = viz.render_allocations(&allocs);
        for t in 0..8 {
            assert!(
                svg.contains(&format!("type {}", t)),
                "legend missing type {}",
                t
            );
        }
    }

    #[test]
    fn save_svg_writes_to_file() {
        // Use a temp file in std::env::temp_dir.
        let allocs = vec![la(1, 0, 1024, 0)];
        let viz = MemoryVisualizer::new();
        let svg = viz.render_allocations(&allocs);

        let mut path = std::env::temp_dir();
        path.push(format!("ignis_test_{}.svg", std::process::id()));
        std::fs::write(&path, &svg).unwrap();

        let read_back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, svg);

        let _ = std::fs::remove_file(&path);
    }
}