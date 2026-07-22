//! Sync DAG visualizer: render cross-queue dependency graphs.
//!
//! Consumes a [`CrossQueueReport`] and emits one of three formats:
//!
//! - [`SyncDagVisualizer::to_dot`]: Graphviz DOT. Open with
//!   `dot -Tsvg input.dot -o out.svg` or paste into any online DOT viewer.
//!   Produces clean clustered output where each queue becomes a subgraph.
//! - [`SyncDagVisualizer::to_mermaid`]: Mermaid graph syntax. Embed
//!   directly in markdown documents (GitHub renders Mermaid natively).
//! - [`SyncDagVisualizer::to_svg`]: Standalone SVG with no external
//!   dependencies. Lane-per-queue layout, curved cross-queue edges,
//!   cycles outlined in red. Open in any browser.
//!
//! All three formats highlight cycles and orphans visually so the same
//! issues that [`CrossQueueReport`] catches textually are immediately
//! obvious in the picture.
//!
//! [`CrossQueueReport`]: super::cross_queue::CrossQueueReport

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write;

use super::cross_queue::{CrossQueueEdge, CrossQueueReport, TrackedSubmission};
use super::raster_common::{self, palette, Color, Framebuffer};

/// Visualizer configuration.
#[derive(Debug, Clone)]
pub struct SyncDagVizConfig {
    /// Total SVG canvas width in pixels. Mermaid/DOT ignore this.
    pub width: u32,
    /// Vertical pixels per queue lane.
    pub lane_height: u32,
    /// Horizontal pixels between submission columns.
    pub column_width: u32,
    /// Pixel padding around the canvas.
    pub padding: u32,
    /// Whether to draw same-queue dependency edges. Set false to reduce
    /// clutter when only cross-queue edges matter.
    pub draw_same_queue_edges: bool,
}

impl Default for SyncDagVizConfig {
    fn default() -> Self {
        Self {
            width: 1400,
            lane_height: 100,
            column_width: 180,
            padding: 30,
            draw_same_queue_edges: true,
        }
    }
}

/// Computed layout for a single DAG. Both SVG and BMP renderers
/// consume this so positions are guaranteed identical between formats.
struct DagLayout {
    /// Per-submission center positions, relative to the layout's
    /// origin (top-left at (0, 0)).
    positions: HashMap<u64, (i32, i32)>,
    /// Queues in display order (top to bottom).
    queue_keys: Vec<(u32, u32)>,
    /// Total bounding box width.
    width: i32,
    /// Total bounding box height.
    height: i32,
    /// Outer padding.
    pad: i32,
    /// Width of the per-lane label area on the left.
    lane_label_width: i32,
    /// Height of one queue lane.
    lane_h: i32,
}

fn compute_layout(
    submissions: &[TrackedSubmission],
    config: &SyncDagVizConfig,
) -> DagLayout {
    let mut by_queue: HashMap<(u32, u32), Vec<&TrackedSubmission>> = HashMap::new();
    for s in submissions {
        by_queue.entry(s.queue_id()).or_default().push(s);
    }
    let mut queue_keys: Vec<(u32, u32)> = by_queue.keys().copied().collect();
    queue_keys.sort();
    for v in by_queue.values_mut() {
        v.sort_by_key(|s| s.seq);
    }

    let mut positions: HashMap<u64, (i32, i32)> = HashMap::new();
    let lane_width = config.column_width as i32;
    let lane_label_width = 120i32;
    let pad = config.padding as i32;
    let lane_h = config.lane_height as i32;

    for (lane_idx, qkey) in queue_keys.iter().enumerate() {
        let y = pad + lane_idx as i32 * lane_h + lane_h / 2;
        for (col_idx, s) in by_queue[qkey].iter().enumerate() {
            let x = pad + lane_label_width + col_idx as i32 * lane_width;
            positions.insert(s.seq, (x, y));
        }
    }

    let max_cols = by_queue
        .values()
        .map(|v| v.len() as i32)
        .max()
        .unwrap_or(1);
    let width =
        (pad * 2 + lane_label_width + max_cols * lane_width).max(config.width as i32);
    let height = pad * 2 + queue_keys.len() as i32 * lane_h + 80;

    DagLayout {
        positions,
        queue_keys,
        width,
        height,
        pad,
        lane_label_width,
        lane_h,
    }
}

/// Stateless visualizer. Build once, render many.
pub struct SyncDagVisualizer {
    config: SyncDagVizConfig,
}

impl SyncDagVisualizer {
    /// Construct with default configuration.
    pub fn new() -> Self {
        Self {
            config: SyncDagVizConfig::default(),
        }
    }

    /// Construct with custom configuration.
    pub fn with_config(config: SyncDagVizConfig) -> Self {
        Self { config }
    }

    /// Render a Graphviz DOT description.
    pub fn to_dot(&self, report: &CrossQueueReport, submissions: &[TrackedSubmission]) -> String {
        let mut o = String::with_capacity(2048);
        let _ = writeln!(o, "digraph SyncDag {{");
        let _ = writeln!(o, "  rankdir=LR;");
        let _ = writeln!(
            o,
            "  graph [bgcolor=\"#1e1e1e\", fontcolor=\"#e8e8e8\", fontname=\"monospace\"];"
        );
        let _ = writeln!(
            o,
            "  node [shape=box, style=\"filled,rounded\", fillcolor=\"#2a2a2a\", \
             color=\"#404040\", fontcolor=\"#e8e8e8\", fontname=\"monospace\"];"
        );
        let _ = writeln!(
            o,
            "  edge [color=\"#9cdcfe\", fontcolor=\"#a0a0a0\", fontname=\"monospace\", fontsize=10];"
        );

        // Group submissions by queue.
        let mut by_queue: HashMap<(u32, u32), Vec<&TrackedSubmission>> = HashMap::new();
        for s in submissions {
            by_queue.entry(s.queue_id()).or_default().push(s);
        }
        let mut queue_keys: Vec<(u32, u32)> = by_queue.keys().copied().collect();
        queue_keys.sort();

        // Identify nodes participating in any cycle for highlighting.
        let cycle_nodes: std::collections::HashSet<u64> = report
            .cycles
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();

        // Identify orphan-source nodes for highlighting.
        let orphan_signal_nodes: std::collections::HashSet<u64> = report
            .orphan_signals
            .iter()
            .map(|o| o.from_seq)
            .collect();
        let orphan_wait_nodes: std::collections::HashSet<u64> = report
            .orphan_waits
            .iter()
            .map(|o| o.to_seq)
            .collect();

        for (qf, qi) in &queue_keys {
            let _ = writeln!(o, "  subgraph cluster_q{}_{} {{", qf, qi);
            let _ = writeln!(o, "    label=\"Queue {}/{}\";", qf, qi);
            let _ = writeln!(o, "    style=filled;");
            let _ = writeln!(o, "    color=\"#3a3a3a\";");
            let _ = writeln!(o, "    fillcolor=\"#252525\";");
            let _ = writeln!(o, "    fontcolor=\"#c8c8c8\";");

            for s in &by_queue[&(*qf, *qi)] {
                let mut color = "#9cdcfe";
                if cycle_nodes.contains(&s.seq) {
                    color = "#f44747";
                } else if orphan_signal_nodes.contains(&s.seq) {
                    color = "#dcdcaa";
                } else if orphan_wait_nodes.contains(&s.seq) {
                    color = "#dcdcaa";
                }
                let label = format!("#{}\\n{}", s.seq, escape_dot(&s.label));
                let _ = writeln!(
                    o,
                    "    n{} [label=\"{}\", color=\"{}\"];",
                    s.seq, label, color
                );
            }
            let _ = writeln!(o, "  }}");
        }

        // Edges.
        let edges_iter = report
            .cross_queue_edges
            .iter()
            .chain(if self.config.draw_same_queue_edges {
                report.same_queue_edges.iter()
            } else {
                [].iter()
            });

        for e in edges_iter {
            let same_q = e.from_queue == e.to_queue;
            let in_cycle = cycle_nodes.contains(&e.from_seq) && cycle_nodes.contains(&e.to_seq);
            let color = if in_cycle {
                "#f44747"
            } else if same_q {
                "#608b4e"
            } else {
                "#c586c0"
            };
            let _ = writeln!(
                o,
                "  n{} -> n{} [label=\"sem {:#x}\", color=\"{}\"];",
                e.from_seq, e.to_seq, e.via_semaphore, color
            );
        }

        let _ = writeln!(o, "}}");
        o
    }

    /// Render a Mermaid graph block (suitable for markdown embedding).
    pub fn to_mermaid(
        &self,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
    ) -> String {
        let mut o = String::with_capacity(2048);
        let _ = writeln!(o, "graph LR");

        let cycle_nodes: std::collections::HashSet<u64> = report
            .cycles
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();
        let orphan_nodes: std::collections::HashSet<u64> = report
            .orphan_signals
            .iter()
            .map(|o| o.from_seq)
            .chain(report.orphan_waits.iter().map(|o| o.to_seq))
            .collect();

        for s in submissions {
            let cls = if cycle_nodes.contains(&s.seq) {
                ":::cycle"
            } else if orphan_nodes.contains(&s.seq) {
                ":::orphan"
            } else {
                ""
            };
            let label = escape_mermaid(&format!("#{} {} (Q{}/{})", s.seq, s.label, s.queue_family, s.queue_index));
            let _ = writeln!(o, "  n{}[\"{}\"]{}", s.seq, label, cls);
        }

        let edges_iter = report
            .cross_queue_edges
            .iter()
            .chain(if self.config.draw_same_queue_edges {
                report.same_queue_edges.iter()
            } else {
                [].iter()
            });

        for e in edges_iter {
            let in_cycle =
                cycle_nodes.contains(&e.from_seq) && cycle_nodes.contains(&e.to_seq);
            let arrow = if in_cycle { "==>" } else { "-->" };
            let _ = writeln!(
                o,
                "  n{} {}|sem {:#x}| n{}",
                e.from_seq, arrow, e.via_semaphore, e.to_seq
            );
        }

        // Class definitions.
        let _ = writeln!(o);
        let _ = writeln!(o, "  classDef cycle fill:#f44747,stroke:#a82020,color:#fff;");
        let _ = writeln!(o, "  classDef orphan fill:#dcdcaa,stroke:#a89e3a,color:#000;");

        o
    }

    /// Render a standalone SVG. No external dependencies; opens in any browser.
    pub fn to_svg(
        &self,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
    ) -> String {
        self.to_svg_with_prefix(report, submissions, "")
    }

    /// Render an SVG with an optional id-prefix applied to every
    /// internal element id. Required when embedding multiple SVGs in
    /// the same HTML document so arrowhead marker definitions don't
    /// collide between graphs.
    fn to_svg_with_prefix(
        &self,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
        id_prefix: &str,
    ) -> String {
        // Group by queue.
        let mut by_queue: HashMap<(u32, u32), Vec<&TrackedSubmission>> = HashMap::new();
        for s in submissions {
            by_queue.entry(s.queue_id()).or_default().push(s);
        }
        let mut queue_keys: Vec<(u32, u32)> = by_queue.keys().copied().collect();
        queue_keys.sort();
        for v in by_queue.values_mut() {
            v.sort_by_key(|s| s.seq);
        }

        // Layout: each submission gets a column based on its position
        // within its queue's ordered list.
        let mut positions: HashMap<u64, (i32, i32)> = HashMap::new();
        let lane_width = self.config.column_width as i32;
        let lane_label_width = 120i32;
        let pad = self.config.padding as i32;
        let lane_h = self.config.lane_height as i32;

        for (lane_idx, qkey) in queue_keys.iter().enumerate() {
            let y = pad + lane_idx as i32 * lane_h + lane_h / 2;
            for (col_idx, s) in by_queue[qkey].iter().enumerate() {
                let x = pad + lane_label_width + col_idx as i32 * lane_width;
                positions.insert(s.seq, (x, y));
            }
        }

        // Total dimensions.
        let max_cols = by_queue
            .values()
            .map(|v| v.len() as i32)
            .max()
            .unwrap_or(1);
        let total_w = (pad * 2 + lane_label_width + max_cols * lane_width)
            .max(self.config.width as i32);
        let total_h = pad * 2 + queue_keys.len() as i32 * lane_h + 80;

        let cycle_nodes: std::collections::HashSet<u64> = report
            .cycles
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();
        let orphan_signal_nodes: std::collections::HashSet<u64> = report
            .orphan_signals
            .iter()
            .map(|o| o.from_seq)
            .collect();
        let orphan_wait_nodes: std::collections::HashSet<u64> = report
            .orphan_waits
            .iter()
            .map(|o| o.to_seq)
            .collect();

        let mut svg = String::with_capacity(8192);
        let _ = writeln!(svg, r##"<?xml version="1.0" encoding="UTF-8"?>"##);
        // viewBox + width="100%" makes the SVG responsive: browser
        // scales the picture to fill the available window width while
        // preserving aspect ratio. max-width caps it at the natural
        // pixel size so we never up-scale beyond what the layout
        // engine intended (avoids blurry text on very wide displays).
        let _ = writeln!(
            svg,
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block;max-width:{w}px;margin:0 auto;background:#1e1e1e" font-family="Consolas, monospace" font-size="11">"##,
            w = total_w,
            h = total_h
        );

        // Background.
        let _ = writeln!(svg, r##"<rect width="100%" height="100%" fill="#1e1e1e"/>"##);

        // Arrowhead marker definitions. IDs are prefixed so multiple
        // graphs can coexist on the same HTML page without collision.
        let _ = writeln!(svg, "<defs>");
        for (id, color) in &[
            ("arrow_normal", "#9cdcfe"),
            ("arrow_cross", "#c586c0"),
            ("arrow_cycle", "#f44747"),
        ] {
            let _ = writeln!(
                svg,
                r##"<marker id="{}{}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/></marker>"##,
                id_prefix, id, color
            );
        }
        let _ = writeln!(svg, "</defs>");

        // Title and stats.
        let _ = writeln!(
            svg,
            r##"<text x="{}" y="20" fill="#e8e8e8" font-size="14" font-weight="bold">Sync DAG: {} submissions across {} queue(s)</text>"##,
            pad,
            submissions.len(),
            queue_keys.len()
        );

        // Lane backgrounds and labels.
        for (lane_idx, qkey) in queue_keys.iter().enumerate() {
            let y = pad + lane_idx as i32 * lane_h;
            let _ = writeln!(
                svg,
                r##"<rect x="{}" y="{}" width="{}" height="{}" fill="#252525" stroke="#3a3a3a" stroke-width="1"/>"##,
                pad,
                y,
                total_w - pad * 2,
                lane_h - 4
            );
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#c8c8c8" font-size="12" font-weight="bold">Queue {}/{}</text>"##,
                pad + 8,
                y + lane_h / 2,
                qkey.0,
                qkey.1
            );
        }

        // Edges. Draw before nodes so nodes overlay the arrows cleanly.
        let edges_iter: Vec<&CrossQueueEdge> = report
            .cross_queue_edges
            .iter()
            .chain(if self.config.draw_same_queue_edges {
                report.same_queue_edges.iter()
            } else {
                [].iter()
            })
            .collect();

        for e in &edges_iter {
            let Some(&(x1, y1)) = positions.get(&e.from_seq) else {
                continue;
            };
            let Some(&(x2, y2)) = positions.get(&e.to_seq) else {
                continue;
            };
            let in_cycle =
                cycle_nodes.contains(&e.from_seq) && cycle_nodes.contains(&e.to_seq);
            let same_q = e.from_queue == e.to_queue;
            let (color, marker) = if in_cycle {
                ("#f44747", "arrow_cycle")
            } else if same_q {
                ("#608b4e", "arrow_normal")
            } else {
                ("#c586c0", "arrow_cross")
            };

            // For same-lane edges draw a straight horizontal arrow.
            // For cross-lane edges use a quadratic Bezier so we don't
            // overlap node boxes.
            let path = if same_q {
                format!(
                    r##"M {} {} L {} {}"##,
                    x1 + 60,
                    y1,
                    x2 - 60,
                    y2
                )
            } else {
                let mid_x = (x1 + x2) / 2;
                let mid_y = (y1 + y2) / 2;
                let bend = (y2 - y1).abs().min(50);
                format!(
                    "M {} {} Q {} {} {} {}",
                    x1 + 60,
                    y1,
                    mid_x,
                    mid_y - bend,
                    x2 - 60,
                    y2
                )
            };
            let _ = writeln!(
                svg,
                r##"<path d="{}" stroke="{}" stroke-width="1.5" fill="none" marker-end="url(#{}{})"/>"##,
                path, color, id_prefix, marker
            );
        }

        // Nodes.
        for s in submissions {
            let Some(&(x, y)) = positions.get(&s.seq) else {
                continue;
            };
            let in_cycle = cycle_nodes.contains(&s.seq);
            let in_orphan_sig = orphan_signal_nodes.contains(&s.seq);
            let in_orphan_wait = orphan_wait_nodes.contains(&s.seq);

            let (fill, stroke) = if in_cycle {
                ("#5a1a1a", "#f44747")
            } else if in_orphan_sig || in_orphan_wait {
                ("#3a3320", "#dcdcaa")
            } else {
                ("#2a2a2a", "#608b4e")
            };

            let box_w = 120i32;
            let box_h = 36i32;
            let bx = x - box_w / 2;
            let by = y - box_h / 2;
            let _ = writeln!(
                svg,
                r##"<g><rect x="{}" y="{}" width="{}" height="{}" rx="4" fill="{}" stroke="{}" stroke-width="2"/>"##,
                bx, by, box_w, box_h, fill, stroke
            );
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#e8e8e8" font-size="11" font-weight="bold" text-anchor="middle">#{}</text>"##,
                x,
                y - 4,
                s.seq
            );
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#c8c8c8" font-size="10" text-anchor="middle">{}</text>"##,
                x,
                y + 10,
                escape_xml(&truncate_string(&s.label, 16))
            );
            let _ = writeln!(
                svg,
                "<title>seq={} label={} queue={}/{} waits={} signals={}</title></g>",
                s.seq,
                escape_xml(&s.label),
                s.queue_family,
                s.queue_index,
                s.wait_semaphores.len(),
                s.signal_semaphores.len()
            );
        }

        // Legend at the bottom.
        let legend_y = pad + queue_keys.len() as i32 * lane_h + 24;
        let mut lx = pad;
        for (label, color) in &[
            ("normal", "#608b4e"),
            ("cross-queue", "#c586c0"),
            ("cycle", "#f44747"),
            ("orphan", "#dcdcaa"),
        ] {
            let _ = writeln!(
                svg,
                r##"<rect x="{}" y="{}" width="14" height="14" fill="{}"/>"##,
                lx,
                legend_y - 11,
                color
            );
            let _ = writeln!(
                svg,
                r##"<text x="{}" y="{}" fill="#c8c8c8" font-size="11">{}</text>"##,
                lx + 20,
                legend_y,
                label
            );
            lx += 120;
        }

        let _ = writeln!(svg, "</svg>");
        svg
    }

    /// Render a DAG to a CPU-side BGRA framebuffer. The framebuffer can
    /// be saved as BMP via [`save_bmp`](Self::save_bmp), passed to
    /// [`to_bmp_combined`](Self::to_bmp_combined) for multi-graph
    /// composition, or processed by user code for custom output.
    pub fn to_bmp(
        &self,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
    ) -> Framebuffer {
        let layout = compute_layout(submissions, &self.config);
        let mut fb = Framebuffer::new(layout.width as u32, layout.height as u32);
        fb.clear(palette::BG);
        self.render_graph_to_fb(&mut fb, 0, 0, &layout, report, submissions);
        fb
    }

    /// Render multiple DAGs into a single tall framebuffer suitable for
    /// scrolling in any image viewer. Each graph gets its own header
    /// row with title, badge (OK / ORPHAN / CYCLE), and stats line.
    ///
    /// This is the answer to "I have 200 graphs, opening 200 SVGs is
    /// painful": one BMP, scroll through it.
    pub fn to_bmp_combined(
        &self,
        title: &str,
        graphs: &[(&str, &CrossQueueReport, &[TrackedSubmission])],
    ) -> Framebuffer {
        let global_header_h: i32 = 60;
        let per_graph_header_h: i32 = 44;
        let graph_spacing: i32 = 20;

        // Compute layouts up front to know total dimensions.
        let layouts: Vec<DagLayout> = graphs
            .iter()
            .map(|(_, _, subs)| compute_layout(subs, &self.config))
            .collect();

        let max_w = layouts.iter().map(|l| l.width).max().unwrap_or(800);
        let total_h: i32 = global_header_h
            + layouts
                .iter()
                .map(|l| per_graph_header_h + l.height + graph_spacing)
                .sum::<i32>();

        let mut fb = Framebuffer::new(max_w as u32, total_h as u32);
        fb.clear(palette::BG);

        // Global header.
        fb.text(20, 20, title, palette::TEXT_HEAD);
        let global_stats = format!("{} graphs in this report", graphs.len());
        fb.text(20, 36, &global_stats, palette::TEXT_DIM);
        // Underline the global header for visual separation.
        fb.rect(0, global_header_h - 2, max_w, 1, palette::FRAME);

        // Render each graph in sequence.
        let mut current_y = global_header_h;
        for (i, ((name, report, subs), layout)) in graphs.iter().zip(layouts.iter()).enumerate() {
            self.draw_section_header(&mut fb, 0, current_y, i + 1, name, report);
            current_y += per_graph_header_h;
            self.render_graph_to_fb(&mut fb, 0, current_y, layout, report, subs);
            current_y += layout.height + graph_spacing;
        }

        fb
    }

    /// Save a single graph as a BMP file.
    pub fn save_bmp(
        &self,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<u64> {
        let fb = self.to_bmp(report, submissions);
        raster_common::save_bmp(&fb, path)
    }

    /// Save a multi-graph BMP file. See [`to_bmp_combined`](Self::to_bmp_combined).
    pub fn save_bmp_combined(
        &self,
        title: &str,
        graphs: &[(&str, &CrossQueueReport, &[TrackedSubmission])],
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<u64> {
        let fb = self.to_bmp_combined(title, graphs);
        raster_common::save_bmp(&fb, path)
    }

    /// Draw the per-section header (title + badge + stats line) for
    /// `to_bmp_combined`. Drawing is done at framebuffer coordinates
    /// (ox, oy); the header occupies `per_graph_header_h` pixels.
    fn draw_section_header(
        &self,
        fb: &mut Framebuffer,
        ox: i32,
        oy: i32,
        index: usize,
        name: &str,
        report: &CrossQueueReport,
    ) {
        let section_title = format!("[{}] {}", index, name);
        fb.text(ox + 20, oy + 8, &section_title, palette::TEXT_HEAD);

        // Badge: OK (green) / ORPHAN (yellow) / CYCLE (red).
        let (badge_text, badge_color, text_dark) = if report.has_cycles() {
            ("CYCLE", palette::BADGE_ERROR, false)
        } else if report.has_orphans() {
            ("ORPHAN", palette::BADGE_WARN, true)
        } else {
            ("OK", palette::BADGE_OK, false)
        };

        let badge_x = ox + 20 + Framebuffer::text_width(&section_title) + 16;
        let badge_w = Framebuffer::text_width(badge_text) + 12;
        let badge_h = 14;
        let badge_y = oy + 6;
        fb.rect(badge_x, badge_y, badge_w, badge_h, badge_color);
        let text_color = if text_dark {
            palette::TEXT_DARK
        } else {
            palette::TEXT_HEAD
        };
        fb.text(badge_x + 6, badge_y + 3, badge_text, text_color);

        // Stats line.
        let stats = format!(
            "submissions={}  queues={}  edges={}  cycles={}  orphans={}",
            report.submission_count,
            report.queue_count,
            report.cross_queue_edges.len() + report.same_queue_edges.len(),
            report.cycles.len(),
            report.orphan_signals.len() + report.orphan_waits.len(),
        );
        fb.text(ox + 20, oy + 26, &stats, palette::TEXT_DIM);
    }

    /// Core graph renderer. Draws lane backgrounds, edges, and nodes
    /// into `fb` translated by (ox, oy). Used by both `to_bmp` and
    /// `to_bmp_combined`.
    fn render_graph_to_fb(
        &self,
        fb: &mut Framebuffer,
        ox: i32,
        oy: i32,
        layout: &DagLayout,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
    ) {
        // Title line above the lanes.
        let title = format!(
            "Sync DAG: {} submissions across {} queue(s)",
            submissions.len(),
            layout.queue_keys.len()
        );
        fb.text(ox + layout.pad, oy + 8, &title, palette::TEXT_HEAD);

        // Lane backgrounds + labels.
        for (lane_idx, qkey) in layout.queue_keys.iter().enumerate() {
            let y = oy + layout.pad + lane_idx as i32 * layout.lane_h;
            let lane_w = layout.width - layout.pad * 2;
            fb.rect(
                ox + layout.pad,
                y,
                lane_w,
                layout.lane_h - 4,
                palette::LANE_BG,
            );
            fb.rect_outline(
                ox + layout.pad,
                y,
                lane_w,
                layout.lane_h - 4,
                palette::FRAME,
            );
            let label = format!("Queue {}/{}", qkey.0, qkey.1);
            fb.text(
                ox + layout.pad + 8,
                y + layout.lane_h / 2 - 4,
                &label,
                palette::TEXT,
            );
        }

        // Build sets of nodes that appear in cycles or as orphans for
        // visual highlighting.
        let cycle_nodes: HashSet<u64> = report
            .cycles
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();
        let orphan_signal_nodes: HashSet<u64> = report
            .orphan_signals
            .iter()
            .map(|o| o.from_seq)
            .collect();
        let orphan_wait_nodes: HashSet<u64> = report
            .orphan_waits
            .iter()
            .map(|o| o.to_seq)
            .collect();

        // Edges. Drawn before nodes so node boxes overlay arrow tails
        // cleanly, leaving only arrowheads visible on the node edge.
        let edges_iter: Vec<&CrossQueueEdge> = report
            .cross_queue_edges
            .iter()
            .chain(if self.config.draw_same_queue_edges {
                report.same_queue_edges.iter()
            } else {
                [].iter()
            })
            .collect();

        for e in &edges_iter {
            let Some(&(x1, y1)) = layout.positions.get(&e.from_seq) else {
                continue;
            };
            let Some(&(x2, y2)) = layout.positions.get(&e.to_seq) else {
                continue;
            };
            let in_cycle =
                cycle_nodes.contains(&e.from_seq) && cycle_nodes.contains(&e.to_seq);
            let same_q = e.from_queue == e.to_queue;
            let color = if in_cycle {
                palette::EDGE_CYCLE
            } else if same_q {
                palette::EDGE_NORMAL
            } else {
                palette::EDGE_CROSS
            };

            // Pull endpoints inward by 60 pixels (half a node) so the
            // arrow doesn't disappear under the node box.
            let sx = ox + x1 + 60;
            let sy = oy + y1;
            let ex = ox + x2 - 60;
            let ey = oy + y2;

            if same_q {
                fb.line_thick(sx, sy, ex, ey, color);
                let dx = (ex - sx) as f32;
                let dy = (ey - sy) as f32;
                fb.arrowhead(ex, ey, dx, dy, 8, color);
            } else {
                // Quadratic Bezier with control point above the midpoint.
                let mid_x = (sx + ex) / 2;
                let mid_y = (sy + ey) / 2;
                let bend = (ey - sy).abs().min(50);
                let cx = mid_x;
                let cy = mid_y - bend;
                fb.bezier_quad_thick(sx, sy, cx, cy, ex, ey, color);
                // Tangent at t=1 of B(t) = (1-t)^2 P0 + 2(1-t)t C + t^2 P1
                // is 2(P1 - C). Use this for arrowhead direction.
                let dx = 2.0 * (ex - cx) as f32;
                let dy = 2.0 * (ey - cy) as f32;
                fb.arrowhead(ex, ey, dx, dy, 8, color);
            }
        }

        // Nodes. Each is a 120x36 rounded-ish rectangle (raster_common
        // does sharp corners; "rounded" is left to SVG only).
        let box_w = 120i32;
        let box_h = 36i32;
        for s in submissions {
            let Some(&(x, y)) = layout.positions.get(&s.seq) else {
                continue;
            };
            let in_cycle = cycle_nodes.contains(&s.seq);
            let in_orphan = orphan_signal_nodes.contains(&s.seq)
                || orphan_wait_nodes.contains(&s.seq);
            let (fill, stroke) = if in_cycle {
                (palette::CYCLE_FILL, palette::CYCLE_STROKE)
            } else if in_orphan {
                (palette::ORPHAN_FILL, palette::ORPHAN_STROKE)
            } else {
                (palette::NODE_FILL, palette::NODE_STROKE)
            };
            let bx = ox + x - box_w / 2;
            let by = oy + y - box_h / 2;
            fb.rect(bx, by, box_w, box_h, fill);
            fb.rect_outline_thick(bx, by, box_w, box_h, 2, stroke);

            let seq_text = format!("#{}", s.seq);
            fb.text_centered(ox + x, by + 4, &seq_text, palette::TEXT_HEAD);
            let label = truncate_string(&s.label, 14);
            fb.text_centered(ox + x, by + 18, &label, palette::TEXT);
        }

        // Legend at the bottom of the graph region.
        let legend_y =
            oy + layout.pad + layout.queue_keys.len() as i32 * layout.lane_h + 24;
        let mut lx = ox + layout.pad;
        for (label, color) in &[
            ("normal", palette::EDGE_NORMAL),
            ("cross-queue", palette::EDGE_CROSS),
            ("cycle", palette::EDGE_CYCLE),
            ("orphan", palette::ORPHAN_STROKE),
        ] {
            fb.rect(lx, legend_y - 11, 14, 14, *color);
            fb.text(lx + 20, legend_y - 8, label, palette::TEXT);
            lx += 120;
        }
    }

    /// Render multiple graphs into a single HTML document with sticky
    /// navigation and embedded SVGs. Solves the "200 fragmented SVG
    /// files" problem: one file lists every graph, click a link to
    /// jump to it, all SVGs render responsively in the right pane.
    ///
    /// Each tuple is `(name, report, submissions)`. The name appears
    /// in the table of contents and as the section heading.
    pub fn to_html_index(
        &self,
        title: &str,
        graphs: &[(&str, &CrossQueueReport, &[TrackedSubmission])],
    ) -> String {
        let mut o = String::with_capacity(graphs.len() * 4096 + 4096);

        // ── Aggregate stats for the page header ──
        let total_subs: usize = graphs.iter().map(|(_, r, _)| r.submission_count).sum();
        let total_cycles: usize = graphs.iter().map(|(_, r, _)| r.cycles.len()).sum();
        let total_orphans: usize = graphs
            .iter()
            .map(|(_, r, _)| r.orphan_signals.len() + r.orphan_waits.len())
            .sum();

        let _ = writeln!(o, "<!DOCTYPE html>");
        let _ = writeln!(o, "<html lang=\"en\">");
        let _ = writeln!(o, "<head>");
        let _ = writeln!(o, "<meta charset=\"UTF-8\">");
        let _ = writeln!(
            o,
            "<title>{}</title>",
            escape_xml(title)
        );
        // Embedded stylesheet keeps the file standalone; no external
        // CDN, works offline, ports easily into a zip artifact.
        let _ = writeln!(o, "<style>");
        o.push_str(HTML_INDEX_CSS);
        let _ = writeln!(o, "</style>");
        let _ = writeln!(o, "</head>");
        let _ = writeln!(o, "<body>");
        let _ = writeln!(o, "<div class=\"layout\">");

        // ── Sticky navigation pane ──
        let _ = writeln!(o, "<nav>");
        let _ = writeln!(o, "<h3>Graphs ({})</h3>", graphs.len());
        let _ = writeln!(o, "<ul>");
        for (i, (name, report, _)) in graphs.iter().enumerate() {
            let badge_class = badge_class_for(report);
            let badge_text = badge_text_for(report);
            let _ = writeln!(
                o,
                "<li><a href=\"#g{i}\">{name}<span class=\"badge {badge_class}\">{badge_text}</span></a></li>",
                name = escape_xml(name),
            );
        }
        let _ = writeln!(o, "</ul>");
        let _ = writeln!(o, "</nav>");

        // ── Main content pane ──
        let _ = writeln!(o, "<main>");
        let _ = writeln!(
            o,
            "<header><h1>{}</h1>",
            escape_xml(title)
        );
        let _ = writeln!(
            o,
            "<div class=\"summary\">{} graphs &middot; {} total submissions &middot; {} cycles &middot; {} orphans</div>",
            graphs.len(),
            total_subs,
            total_cycles,
            total_orphans
        );
        let _ = writeln!(o, "</header>");

        for (i, (name, report, subs)) in graphs.iter().enumerate() {
            let badge_class = badge_class_for(report);
            let badge_text = badge_text_for(report);
            let _ = writeln!(o, "<section id=\"g{i}\">");
            let _ = writeln!(
                o,
                "<h2>{name}<span class=\"badge {badge_class}\">{badge_text}</span></h2>",
                name = escape_xml(name)
            );
            let _ = writeln!(
                o,
                "<div class=\"stats\">submissions={} &middot; queues={} &middot; cross-queue edges={} &middot; same-queue edges={} &middot; cycles={} &middot; orphans={}</div>",
                report.submission_count,
                report.queue_count,
                report.cross_queue_edges.len(),
                report.same_queue_edges.len(),
                report.cycles.len(),
                report.orphan_signals.len() + report.orphan_waits.len()
            );

            // Render the SVG with a per-graph id prefix. Strip the XML
            // prolog so it embeds cleanly in HTML body content.
            let prefix = format!("g{i}_");
            let svg = self.to_svg_with_prefix(report, subs, &prefix);
            for line in svg.lines() {
                if line.starts_with("<?xml") {
                    continue;
                }
                o.push_str(line);
                o.push('\n');
            }

            let _ = writeln!(o, "</section>");
        }

        let _ = writeln!(o, "</main>");
        let _ = writeln!(o, "</div>");
        let _ = writeln!(o, "</body>");
        let _ = writeln!(o, "</html>");
        o
    }

    /// Render multiple graphs to an HTML index file on disk.
    pub fn save_html_index(
        &self,
        title: &str,
        graphs: &[(&str, &CrossQueueReport, &[TrackedSubmission])],
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<u64> {
        let body = self.to_html_index(title, graphs);
        std::fs::write(path.as_ref(), &body)?;
        Ok(body.len() as u64)
    }

    /// Render to a file, choosing format from extension (`.dot`, `.mmd`,
    /// `.svg`). Returns the number of bytes written.
    pub fn save(
        &self,
        report: &CrossQueueReport,
        submissions: &[TrackedSubmission],
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<u64> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("svg")
            .to_ascii_lowercase();
        match ext.as_str() {
            "dot" | "gv" => {
                let body = self.to_dot(report, submissions);
                std::fs::write(path, &body)?;
                Ok(body.len() as u64)
            }
            "mmd" | "mermaid" => {
                let body = self.to_mermaid(report, submissions);
                std::fs::write(path, &body)?;
                Ok(body.len() as u64)
            }
            "bmp" => self.save_bmp(report, submissions, path),
            _ => {
                let body = self.to_svg(report, submissions);
                std::fs::write(path, &body)?;
                Ok(body.len() as u64)
            }
        }
    }
}

impl Default for SyncDagVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn escape_mermaid(s: &str) -> String {
    s.replace('"', "&quot;").replace('\n', " ")
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn truncate_string(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", truncated)
}

fn badge_class_for(report: &CrossQueueReport) -> &'static str {
    if report.has_cycles() {
        "badge-error"
    } else if report.has_orphans() {
        "badge-warn"
    } else {
        "badge-clean"
    }
}

fn badge_text_for(report: &CrossQueueReport) -> &'static str {
    if report.has_cycles() {
        "CYCLE"
    } else if report.has_orphans() {
        "ORPHAN"
    } else {
        "OK"
    }
}

/// Embedded CSS for the HTML index. Dark theme, sticky nav, responsive SVGs.
/// Kept as a constant so the index file stays single-file portable.
const HTML_INDEX_CSS: &str = r#"
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #1e1e1e;
    color: #e8e8e8;
    font-family: Consolas, "Cascadia Code", monospace;
    font-size: 13px;
}
.layout {
    display: grid;
    grid-template-columns: 260px 1fr;
    min-height: 100vh;
}
nav {
    position: sticky;
    top: 0;
    align-self: start;
    height: 100vh;
    overflow-y: auto;
    padding: 16px 12px;
    background: #252525;
    border-right: 1px solid #3a3a3a;
}
nav h3 {
    margin: 0 0 12px 0;
    color: #fff;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
nav ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
nav li { margin: 0; }
nav a {
    color: #c8c8c8;
    text-decoration: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px;
    border-radius: 3px;
    transition: background 0.1s;
}
nav a:hover { background: #2a2a2a; color: #fff; }
main {
    padding: 24px 32px;
    overflow-x: hidden;
}
header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #3a3a3a;
}
header h1 {
    margin: 0 0 8px 0;
    color: #fff;
    font-size: 22px;
}
.summary {
    color: #a0a0a0;
    font-size: 12px;
}
section {
    margin-bottom: 32px;
    padding: 20px;
    background: #252525;
    border-radius: 6px;
    border: 1px solid #3a3a3a;
}
section h2 {
    margin: 0 0 8px 0;
    color: #9cdcfe;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.stats {
    color: #a0a0a0;
    font-size: 11px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px dashed #3a3a3a;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 0.5px;
}
.badge-clean { background: #608b4e; color: #fff; }
.badge-warn { background: #dcdcaa; color: #1e1e1e; }
.badge-error { background: #f44747; color: #fff; }
section svg {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
    border: 1px solid #3a3a3a;
}
@media (max-width: 800px) {
    .layout { grid-template-columns: 1fr; }
    nav { position: static; height: auto; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::cross_queue::CrossQueueTracker;

    fn build_diamond() -> (CrossQueueReport, Vec<TrackedSubmission>) {
        let t = CrossQueueTracker::new();
        t.record_raw(0, 0, "geometry", &[], &[], &[1, 2], 0);
        t.record_raw(1, 0, "shadow_a", &[], &[1], &[3], 0);
        t.record_raw(1, 0, "shadow_b", &[], &[2], &[4], 0);
        t.record_raw(0, 0, "compose", &[], &[3, 4], &[], 0);
        let report = t.analyze();
        let snap = t.snapshot();
        (report, snap)
    }

    #[test]
    fn dot_output_contains_clusters_and_edges() {
        let (report, subs) = build_diamond();
        let viz = SyncDagVisualizer::new();
        let dot = viz.to_dot(&report, &subs);
        assert!(dot.starts_with("digraph SyncDag"));
        assert!(dot.contains("subgraph cluster_q0_0"));
        assert!(dot.contains("subgraph cluster_q1_0"));
        assert!(dot.contains("n1 -> n2") || dot.contains("n1 -> n3"));
        assert!(dot.ends_with("}\n"));
    }

    #[test]
    fn mermaid_output_has_class_definitions() {
        let (report, subs) = build_diamond();
        let viz = SyncDagVisualizer::new();
        let m = viz.to_mermaid(&report, &subs);
        assert!(m.starts_with("graph LR"));
        assert!(m.contains("classDef cycle"));
        assert!(m.contains("classDef orphan"));
    }

    #[test]
    fn svg_output_is_well_formed() {
        let (report, subs) = build_diamond();
        let viz = SyncDagVisualizer::new();
        let svg = viz.to_svg(&report, &subs);
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.ends_with("</svg>\n"));
        // 4 nodes + lane backgrounds + legend rects.
        let rect_count = svg.matches("<rect").count();
        assert!(rect_count >= 4);
    }

    #[test]
    fn cycle_renders_in_red() {
        let t = CrossQueueTracker::new();
        t.record_raw(0, 0, "A", &[], &[2], &[1], 0);
        t.record_raw(1, 0, "B", &[], &[1], &[2], 0);
        let report = t.analyze();
        let subs = t.snapshot();
        assert!(!report.cycles.is_empty());

        let viz = SyncDagVisualizer::new();
        let svg = viz.to_svg(&report, &subs);
        // Cycle stroke color should appear at least once.
        assert!(svg.contains("#f44747"));
    }

    #[test]
    fn save_picks_format_from_extension() {
        let (report, subs) = build_diamond();
        let viz = SyncDagVisualizer::new();
        let dir = std::env::temp_dir();
        let dot_path = dir.join(format!("ignis_dag_test_{}.dot", std::process::id()));
        let svg_path = dir.join(format!("ignis_dag_test_{}.svg", std::process::id()));

        viz.save(&report, &subs, &dot_path).unwrap();
        viz.save(&report, &subs, &svg_path).unwrap();
        let dot_body = std::fs::read_to_string(&dot_path).unwrap();
        let svg_body = std::fs::read_to_string(&svg_path).unwrap();
        assert!(dot_body.starts_with("digraph"));
        assert!(svg_body.starts_with("<?xml"));

        let _ = std::fs::remove_file(&dot_path);
        let _ = std::fs::remove_file(&svg_path);
    }
    #[test]
    fn bmp_output_is_valid_file() {
        let (report, subs) = build_diamond();
        let viz = SyncDagVisualizer::new();
        let path = std::env::temp_dir().join(format!(
            "ignis_dag_bmp_test_{}.bmp",
            std::process::id()
        ));
        let bytes = viz.save_bmp(&report, &subs, &path).unwrap();
        assert!(bytes > 100);
        let content = std::fs::read(&path).unwrap();
        assert_eq!(&content[0..2], b"BM");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bmp_combined_includes_all_graphs() {
        let (r1, s1) = build_diamond();
        let (r2, s2) = build_diamond();
        let viz = SyncDagVisualizer::new();
        let graphs = [
            ("first", &r1, s1.as_slice()),
            ("second", &r2, s2.as_slice()),
        ];
        let fb = viz.to_bmp_combined("Test Combined", &graphs);
        // Combined image should be at least 2x as tall as a single graph.
        let single = viz.to_bmp(&r1, &s1);
        assert!(
            fb.height() >= single.height() * 2,
            "combined height {} should be >= 2x single {}",
            fb.height(),
            single.height()
        );
    }
}