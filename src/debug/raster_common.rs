//! Shared rasterizer primitives for debug visualizers.
//!
//! Provides a CPU-side BGRA framebuffer with line/curve/rect/text/triangle
//! drawing, an embedded 8x8 monospace font, and a BMP encoder. Used by
//! the sync DAG bitmap renderer and reusable by any future diagnostic
//! that needs to ship "open the file in Photos" output.
//!
//! Performance is not a goal here. The rasterizer is meant for offline
//! diagnostic output (one image per analysis run, not per frame). Hot
//! paths use `slice::fill` for solid rectangles, which is the only
//! commonly-hit fast path. Lines use Bresenham, curves are sampled at
//! 32 steps with line segments between samples.
//!
//! # BMP-only output
//!
//! BMP is chosen over PNG because:
//! 1. No external dependencies (PNG needs deflate + CRC32).
//! 2. Universally viewable on every desktop OS (Photos / Preview / xdg-open).
//! 3. File size is acceptable for diagnostic output (a 2000x3000 BMP is
//!    ~17 MiB; same image as PNG would be ~500 KiB, but disk is cheap
//!    and analysis runs are not continuous).
//!
//! If PNG is needed later, `bgra_to_bgr` and `Framebuffer::bytes`
//! provide the raw inputs a PNG encoder would consume.
//!
//! # Font
//!
//! The 8x8 font is a copy of the IBM PC CGA-style ASCII set used by
//! [`debug_window`](crate::debug_window). Duplicated rather than shared
//! because the two features (`debug-tools` and `debug-window`) are
//! independently selectable.

use std::path::Path;

/// 32-bit BGRA color packed into a single u32. Alpha is stored but the
/// rasterizer treats every draw call as opaque (no alpha blending).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color(pub u8, pub u8, pub u8, pub u8); // B, G, R, A

impl Color {
    /// Construct an opaque color from RGB components.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self(b, g, r, 0xFF)
    }

    /// Pack as a little-endian u32 in BGRA byte order.
    #[inline(always)]
    fn packed(self) -> u32 {
        u32::from_le_bytes([self.0, self.1, self.2, self.3])
    }
}

/// Standard palette used by debug visualizers.
pub mod palette {
    use super::Color;

    /// Default canvas background.
    pub const BG: Color = Color::rgb(0x1E, 0x1E, 0x1E);
    /// Background fill for inset panels.
    pub const PANEL_BG: Color = Color::rgb(0x25, 0x25, 0x25);
    /// Color for thin frame borders and separator lines.
    pub const FRAME: Color = Color::rgb(0x40, 0x40, 0x40);
    /// Primary text color.
    pub const TEXT: Color = Color::rgb(0xE8, 0xE8, 0xE8);
    /// Dimmed/secondary text color.
    pub const TEXT_DIM: Color = Color::rgb(0x90, 0x90, 0x90);
    /// Heading text color (brightest).
    pub const TEXT_HEAD: Color = Color::rgb(0xFF, 0xFF, 0xFF);
    /// Dark text used on light backgrounds (badges).
    pub const TEXT_DARK: Color = Color::rgb(0x1E, 0x1E, 0x1E);

    /// Fill color for normal graph nodes.
    pub const NODE_FILL: Color = Color::rgb(0x2A, 0x2A, 0x2A);
    /// Stroke color for normal graph nodes.
    pub const NODE_STROKE: Color = Color::rgb(0x60, 0x8B, 0x4E);
    /// Fill color for nodes participating in a dependency cycle.
    pub const CYCLE_FILL: Color = Color::rgb(0x5A, 0x1A, 0x1A);
    /// Stroke color for nodes participating in a dependency cycle.
    pub const CYCLE_STROKE: Color = Color::rgb(0xF4, 0x47, 0x47);
    /// Fill color for nodes with orphan signal/wait semaphores.
    pub const ORPHAN_FILL: Color = Color::rgb(0x3A, 0x33, 0x20);
    /// Stroke color for nodes with orphan signal/wait semaphores.
    pub const ORPHAN_STROKE: Color = Color::rgb(0xDC, 0xDC, 0xAA);

    /// Edge color for normal same-queue dependencies.
    pub const EDGE_NORMAL: Color = Color::rgb(0x60, 0x8B, 0x4E);
    /// Edge color for cross-queue dependencies.
    pub const EDGE_CROSS: Color = Color::rgb(0xC5, 0x86, 0xC0);
    /// Edge color for edges participating in a cycle.
    pub const EDGE_CYCLE: Color = Color::rgb(0xF4, 0x47, 0x47);

    /// Background color for queue lanes in DAG visualizations.
    pub const LANE_BG: Color = Color::rgb(0x25, 0x25, 0x25);
    /// Badge color for OK/clean status.
    pub const BADGE_OK: Color = Color::rgb(0x60, 0x8B, 0x4E);
    /// Badge color for warning status (orphans present).
    pub const BADGE_WARN: Color = Color::rgb(0xDC, 0xDC, 0xAA);
    /// Badge color for error status (cycles present).
    pub const BADGE_ERROR: Color = Color::rgb(0xF4, 0x47, 0x47);
}

/// CPU-side BGRA framebuffer.
pub struct Framebuffer {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Framebuffer {
    /// Allocate a fresh BGRA buffer initialized to zero (transparent black).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            pixels: vec![0u8; (width as usize) * (height as usize) * 4],
            width,
            height,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Raw BGRA bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.pixels
    }

    /// Borrow the pixel buffer as u32 slice for fast bulk operations.
    #[inline(always)]
    fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        let len = self.pixels.len() / 4;
        // SAFETY: pixels was allocated as Vec<u8> with length 4*pixel_count.
        // u32 alignment is 4; Vec<u8>'s allocator gives at least 4-byte
        // alignment up to its element alignment, which is sufficient on
        // every platform Rust supports.
        unsafe {
            std::slice::from_raw_parts_mut(self.pixels.as_mut_ptr() as *mut u32, len)
        }
    }

    /// Fill the entire framebuffer with one color.
    pub fn clear(&mut self, c: Color) {
        let p = c.packed();
        self.as_u32_slice_mut().fill(p);
    }

    /// Set a single pixel. Out-of-bounds is silently dropped.
    #[inline(always)]
    pub fn put(&mut self, x: i32, y: i32, c: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let stride = self.width as usize;
        let idx = (y as usize) * stride + (x as usize);
        self.as_u32_slice_mut()[idx] = c.packed();
    }

    /// Filled axis-aligned rectangle. Negative or out-of-bounds rects
    /// are clipped to the framebuffer.
    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, c: Color) {
        if w <= 0 || h <= 0 {
            return;
        }
        let x0 = x.max(0).min(self.width as i32) as usize;
        let y0 = y.max(0).min(self.height as i32) as usize;
        let x1 = (x + w).max(0).min(self.width as i32) as usize;
        let y1 = (y + h).max(0).min(self.height as i32) as usize;
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let p = c.packed();
        let stride = self.width as usize;
        let buf = self.as_u32_slice_mut();
        for yy in y0..y1 {
            let row_start = yy * stride + x0;
            let row_end = row_start + (x1 - x0);
            buf[row_start..row_end].fill(p);
        }
    }

    /// 1-pixel-thick rectangle outline.
    pub fn rect_outline(&mut self, x: i32, y: i32, w: i32, h: i32, c: Color) {
        if w <= 0 || h <= 0 {
            return;
        }
        self.rect(x, y, w, 1, c);
        self.rect(x, y + h - 1, w, 1, c);
        self.rect(x, y, 1, h, c);
        self.rect(x + w - 1, y, 1, h, c);
    }

    /// N-pixel-thick rectangle outline. Each layer is offset inward by
    /// one pixel from the previous.
    pub fn rect_outline_thick(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        thickness: i32,
        c: Color,
    ) {
        for t in 0..thickness {
            self.rect_outline(x + t, y + t, w - 2 * t, h - 2 * t, c);
        }
    }

    /// Bresenham line from (x0,y0) to (x1,y1).
    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Color) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x = x0;
        let mut y = y0;
        loop {
            self.put(x, y, c);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }

    /// 2-pixel-thick line. Draws Bresenham + 1 perpendicular offset.
    pub fn line_thick(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Color) {
        self.line(x0, y0, x1, y1, c);
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        if dx > dy {
            self.line(x0, y0 + 1, x1, y1 + 1, c);
        } else {
            self.line(x0 + 1, y0, x1 + 1, y1, c);
        }
    }

    /// Quadratic Bezier curve sampled at 32 steps and connected with
    /// line segments. Adequate for diagnostic rendering at moderate
    /// curvatures.
    pub fn bezier_quad(
        &mut self,
        x0: i32,
        y0: i32,
        cx: i32,
        cy: i32,
        x1: i32,
        y1: i32,
        c: Color,
    ) {
        const STEPS: usize = 32;
        let mut prev_x = x0;
        let mut prev_y = y0;
        for i in 1..=STEPS {
            let t = i as f32 / STEPS as f32;
            let omt = 1.0 - t;
            let bx = omt * omt * x0 as f32
                + 2.0 * omt * t * cx as f32
                + t * t * x1 as f32;
            let by = omt * omt * y0 as f32
                + 2.0 * omt * t * cy as f32
                + t * t * y1 as f32;
            let px = bx as i32;
            let py = by as i32;
            self.line(prev_x, prev_y, px, py, c);
            prev_x = px;
            prev_y = py;
        }
    }

    /// 2-pixel-thick quadratic Bezier.
    pub fn bezier_quad_thick(
        &mut self,
        x0: i32,
        y0: i32,
        cx: i32,
        cy: i32,
        x1: i32,
        y1: i32,
        c: Color,
    ) {
        self.bezier_quad(x0, y0, cx, cy, x1, y1, c);
        self.bezier_quad(x0, y0 + 1, cx, cy + 1, x1, y1 + 1, c);
    }

    /// Filled triangle via scanline rasterization.
    pub fn triangle_filled(
        &mut self,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        c: Color,
    ) {
        let mut pts = [(x0, y0), (x1, y1), (x2, y2)];
        pts.sort_by_key(|p| p.1);
        let (xa, ya) = pts[0];
        let (xb, yb) = pts[1];
        let (xc, yc) = pts[2];
        for y in ya..=yc {
            let left_edge = if y < yb {
                lerp_x(xa, ya, xb, yb, y)
            } else {
                lerp_x(xb, yb, xc, yc, y)
            };
            let right_edge = lerp_x(xa, ya, xc, yc, y);
            let l = left_edge.min(right_edge);
            let r = left_edge.max(right_edge);
            for x in l..=r {
                self.put(x, y, c);
            }
        }
    }

    /// Draw an arrowhead at (tip_x, tip_y) pointing in direction (dx, dy).
    /// `size` is the length of the arrowhead from tip to base.
    pub fn arrowhead(
        &mut self,
        tip_x: i32,
        tip_y: i32,
        dx: f32,
        dy: f32,
        size: i32,
        c: Color,
    ) {
        let len = (dx * dx + dy * dy).sqrt().max(0.001);
        let ux = dx / len;
        let uy = dy / len;
        let px = -uy;
        let py = ux;
        let s = size as f32;
        let bx = tip_x as f32 - ux * s;
        let by = tip_y as f32 - uy * s;
        let lx = (bx + px * s * 0.5) as i32;
        let ly = (by + py * s * 0.5) as i32;
        let rx = (bx - px * s * 0.5) as i32;
        let ry = (by - py * s * 0.5) as i32;
        self.triangle_filled(tip_x, tip_y, lx, ly, rx, ry, c);
    }

    /// Draw an 8x8 bitmap glyph at (x, y).
    pub fn glyph(&mut self, x: i32, y: i32, ch: char, c: Color) {
        let bytes = glyph_bytes(ch);
        for (row, byte) in bytes.iter().enumerate() {
            for col in 0..8 {
                if (byte >> (7 - col)) & 1 != 0 {
                    self.put(x + col as i32, y + row as i32, c);
                }
            }
        }
    }

    /// Draw a string starting at (x, y) using the 8x8 bitmap font.
    pub fn text(&mut self, x: i32, y: i32, s: &str, c: Color) {
        let mut cx = x;
        for ch in s.chars() {
            self.glyph(cx, y, ch, c);
            cx += 8;
        }
    }

    /// Draw a string centered horizontally at `cx`.
    pub fn text_centered(&mut self, cx: i32, y: i32, s: &str, c: Color) {
        let w = (s.chars().count() as i32) * 8;
        self.text(cx - w / 2, y, s, c);
    }

    /// Width in pixels of a string rendered with [`text`].
    pub fn text_width(s: &str) -> i32 {
        (s.chars().count() as i32) * 8
    }
}

/// Linear interpolation of x along the edge from (x0,y0) to (x1,y1) at
/// vertical position y. Used by `triangle_filled`.
fn lerp_x(x0: i32, y0: i32, x1: i32, y1: i32, y: i32) -> i32 {
    if y1 == y0 {
        return x0;
    }
    x0 + (x1 - x0) * (y - y0) / (y1 - y0)
}

/// Look up the 8x8 bitmap for a character. Unknown characters render as
/// a solid block so accidental usage is visually obvious.
pub fn glyph_bytes(c: char) -> [u8; 8] {
    GLYPHS
        .iter()
        .find(|(g, _)| *g == c)
        .map(|(_, bytes)| *bytes)
        .unwrap_or(UNKNOWN)
}

const UNKNOWN: [u8; 8] = [0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x00];

/// 8x8 monospace font. Mirrors the IBM PC CGA character ROM aesthetic.
/// Same data as [`debug_window::font`](crate::debug_window) but
/// duplicated to keep this module independent of the `debug-window`
/// feature gate.
#[rustfmt::skip]
const GLYPHS: &[(char, [u8; 8])] = &[
    (' ', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    ('!', [0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00, 0x00]),
    ('"', [0x6C, 0x6C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    ('#', [0x6C, 0xFE, 0x6C, 0x6C, 0x6C, 0xFE, 0x6C, 0x00]),
    ('$', [0x18, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x18, 0x00]),
    ('%', [0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00, 0x00]),
    ('&', [0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00]),
    ('\'', [0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00]),
    ('(', [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00]),
    (')', [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00]),
    ('*', [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00]),
    ('+', [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00]),
    (',', [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30]),
    ('-', [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00]),
    ('.', [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00]),
    ('/', [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00]),

    ('0', [0x7C, 0xCE, 0xDE, 0xF6, 0xE6, 0xC6, 0x7C, 0x00]),
    ('1', [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00]),
    ('2', [0x7C, 0xC6, 0x06, 0x1C, 0x30, 0x66, 0xFE, 0x00]),
    ('3', [0x7C, 0xC6, 0x06, 0x3C, 0x06, 0xC6, 0x7C, 0x00]),
    ('4', [0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x1E, 0x00]),
    ('5', [0xFE, 0xC0, 0xC0, 0xFC, 0x06, 0xC6, 0x7C, 0x00]),
    ('6', [0x38, 0x60, 0xC0, 0xFC, 0xC6, 0xC6, 0x7C, 0x00]),
    ('7', [0xFE, 0xC6, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00]),
    ('8', [0x7C, 0xC6, 0xC6, 0x7C, 0xC6, 0xC6, 0x7C, 0x00]),
    ('9', [0x7C, 0xC6, 0xC6, 0x7E, 0x06, 0x0C, 0x78, 0x00]),

    (':', [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00]),
    (';', [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30]),
    ('<', [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00]),
    ('=', [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00]),
    ('>', [0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00]),
    ('?', [0x7C, 0xC6, 0x0C, 0x18, 0x18, 0x00, 0x18, 0x00]),
    ('@', [0x7C, 0xC6, 0xDE, 0xDE, 0xDC, 0xC0, 0x7C, 0x00]),

    ('A', [0x38, 0x6C, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0x00]),
    ('B', [0xFC, 0x66, 0x66, 0x7C, 0x66, 0x66, 0xFC, 0x00]),
    ('C', [0x3C, 0x66, 0xC0, 0xC0, 0xC0, 0x66, 0x3C, 0x00]),
    ('D', [0xF8, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0xF8, 0x00]),
    ('E', [0xFE, 0x62, 0x68, 0x78, 0x68, 0x62, 0xFE, 0x00]),
    ('F', [0xFE, 0x62, 0x68, 0x78, 0x68, 0x60, 0xF0, 0x00]),
    ('G', [0x3C, 0x66, 0xC0, 0xC0, 0xCE, 0x66, 0x3E, 0x00]),
    ('H', [0xC6, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00]),
    ('I', [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00]),
    ('J', [0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78, 0x00]),
    ('K', [0xE6, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0xE6, 0x00]),
    ('L', [0xF0, 0x60, 0x60, 0x60, 0x62, 0x66, 0xFE, 0x00]),
    ('M', [0xC6, 0xEE, 0xFE, 0xFE, 0xD6, 0xC6, 0xC6, 0x00]),
    ('N', [0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00]),
    ('O', [0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00]),
    ('P', [0xFC, 0x66, 0x66, 0x7C, 0x60, 0x60, 0xF0, 0x00]),
    ('Q', [0x7C, 0xC6, 0xC6, 0xC6, 0xCE, 0x7C, 0x0E, 0x00]),
    ('R', [0xFC, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0xE6, 0x00]),
    ('S', [0x7C, 0xC6, 0x60, 0x38, 0x0C, 0xC6, 0x7C, 0x00]),
    ('T', [0x7E, 0x7E, 0x5A, 0x18, 0x18, 0x18, 0x3C, 0x00]),
    ('U', [0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00]),
    ('V', [0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00]),
    ('W', [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00]),
    ('X', [0xC6, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0xC6, 0x00]),
    ('Y', [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x3C, 0x00]),
    ('Z', [0xFE, 0xC6, 0x8C, 0x18, 0x32, 0x66, 0xFE, 0x00]),

    ('[', [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00]),
    ('\\', [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00]),
    (']', [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00]),
    ('^', [0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00]),
    ('_', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF]),
    ('`', [0x30, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),

    ('a', [0x00, 0x00, 0x78, 0x0C, 0x7C, 0xCC, 0x76, 0x00]),
    ('b', [0xE0, 0x60, 0x6C, 0x76, 0x66, 0x66, 0xDC, 0x00]),
    ('c', [0x00, 0x00, 0x7C, 0xC6, 0xC0, 0xC6, 0x7C, 0x00]),
    ('d', [0x1C, 0x0C, 0x6C, 0xDC, 0xCC, 0xCC, 0x76, 0x00]),
    ('e', [0x00, 0x00, 0x7C, 0xC6, 0xFE, 0xC0, 0x7C, 0x00]),
    ('f', [0x3C, 0x66, 0x60, 0xF8, 0x60, 0x60, 0xF0, 0x00]),
    ('g', [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0xF8]),
    ('h', [0xE0, 0x60, 0x6C, 0x76, 0x66, 0x66, 0xE6, 0x00]),
    ('i', [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00]),
    ('j', [0x06, 0x00, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C]),
    ('k', [0xE0, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0xE6, 0x00]),
    ('l', [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00]),
    ('m', [0x00, 0x00, 0xCC, 0xFE, 0xFE, 0xD6, 0xC6, 0x00]),
    ('n', [0x00, 0x00, 0xDC, 0x66, 0x66, 0x66, 0x66, 0x00]),
    ('o', [0x00, 0x00, 0x7C, 0xC6, 0xC6, 0xC6, 0x7C, 0x00]),
    ('p', [0x00, 0x00, 0xDC, 0x66, 0x66, 0x7C, 0x60, 0xF0]),
    ('q', [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0x1E]),
    ('r', [0x00, 0x00, 0xDC, 0x76, 0x66, 0x60, 0xF0, 0x00]),
    ('s', [0x00, 0x00, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x00]),
    ('t', [0x30, 0x30, 0xFC, 0x30, 0x30, 0x36, 0x1C, 0x00]),
    ('u', [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00]),
    ('v', [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x00]),
    ('w', [0x00, 0x00, 0xC6, 0xD6, 0xFE, 0xFE, 0x6C, 0x00]),
    ('x', [0x00, 0x00, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0x00]),
    ('y', [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0x7C, 0x0C, 0xF8]),
    ('z', [0x00, 0x00, 0xFE, 0x4C, 0x18, 0x32, 0xFE, 0x00]),

    ('{', [0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00]),
    ('|', [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00]),
    ('}', [0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00]),
    ('~', [0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
];

/// Write an uncompressed 24-bit BMP file. BMPs are bottom-up: the last
/// row of `bgr_pixels` is written first.
pub fn write_bmp(
    path: &Path,
    width: u32,
    height: u32,
    bgr_pixels: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    if bgr_pixels.len() != (width * height * 3) as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "BMP pixel buffer size mismatch",
        ));
    }
    let row_size = ((width * 3 + 3) / 4) * 4;
    let pad_per_row = row_size - width * 3;
    let pixel_array_size = row_size * height;
    let file_size = 54u32 + pixel_array_size;

    let mut f = std::fs::File::create(path)?;

    // BITMAPFILEHEADER (14 bytes).
    f.write_all(b"BM")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(&[0u8; 4])?; // reserved
    f.write_all(&54u32.to_le_bytes())?; // pixel data offset

    // BITMAPINFOHEADER (40 bytes).
    f.write_all(&40u32.to_le_bytes())?;
    f.write_all(&width.to_le_bytes())?;
    f.write_all(&height.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&24u16.to_le_bytes())?;
    f.write_all(&[0u8; 4])?; // BI_RGB
    f.write_all(&pixel_array_size.to_le_bytes())?;
    f.write_all(&[0u8; 16])?; // ppm + colors used + colors important

    let pad = vec![0u8; pad_per_row as usize];
    for y in (0..height).rev() {
        let row_start = (y * width * 3) as usize;
        let row_end = row_start + (width * 3) as usize;
        f.write_all(&bgr_pixels[row_start..row_end])?;
        if pad_per_row > 0 {
            f.write_all(&pad)?;
        }
    }
    Ok(())
}

/// Convert a BGRA framebuffer to BGR (drop alpha) for BMP encoding.
pub fn bgra_to_bgr(bgra: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for chunk in bgra.chunks_exact(4) {
        out.push(chunk[0]); // B
        out.push(chunk[1]); // G
        out.push(chunk[2]); // R
    }
    out
}

/// Save a framebuffer to a BMP file. Returns the number of bytes written.
pub fn save_bmp(fb: &Framebuffer, path: impl AsRef<Path>) -> std::io::Result<u64> {
    let bgr = bgra_to_bgr(fb.bytes(), fb.width(), fb.height());
    write_bmp(path.as_ref(), fb.width(), fb.height(), &bgr)?;
    Ok(std::fs::metadata(path.as_ref())?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_sets_all_pixels() {
        let mut fb = Framebuffer::new(4, 2);
        fb.clear(Color::rgb(255, 0, 0));
        for px in fb.bytes().chunks_exact(4) {
            assert_eq!(px, &[0, 0, 255, 255]);
        }
    }

    #[test]
    fn put_clips_out_of_bounds() {
        let mut fb = Framebuffer::new(2, 2);
        fb.put(-1, 0, Color::rgb(255, 0, 0));
        fb.put(5, 5, Color::rgb(255, 0, 0));
        for byte in fb.bytes() {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn line_draws_diagonal() {
        let mut fb = Framebuffer::new(8, 8);
        fb.line(0, 0, 7, 7, Color::rgb(255, 0, 0));
        // Diagonal pixels should be set.
        for i in 0..8 {
            let off = ((i * 8) + i) * 4;
            assert_eq!(fb.bytes()[off + 2], 255, "pixel ({i},{i}) not set");
        }
    }

    #[test]
    fn triangle_fills_interior() {
        let mut fb = Framebuffer::new(16, 16);
        fb.triangle_filled(2, 2, 13, 2, 7, 13, Color::rgb(255, 0, 0));
        // Center should be filled.
        let center_off = (7 * 16 + 7) * 4;
        assert_eq!(fb.bytes()[center_off + 2], 255);
    }

    #[test]
    fn glyph_renders_into_framebuffer() {
        let mut fb = Framebuffer::new(8, 8);
        fb.glyph(0, 0, 'A', Color::rgb(255, 0, 0));
        let any_set = fb.bytes().chunks_exact(4).any(|p| p[2] == 255);
        assert!(any_set);
    }

    #[test]
    fn unknown_glyph_renders_block() {
        assert_eq!(glyph_bytes('Ω'), UNKNOWN);
    }

    #[test]
    fn bmp_writes_valid_header() {
        let mut fb = Framebuffer::new(4, 4);
        fb.clear(Color::rgb(255, 0, 0));
        let path = std::env::temp_dir().join(format!(
            "ignis_bmp_test_{}.bmp",
            std::process::id()
        ));
        save_bmp(&fb, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..2], b"BM");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn arrowhead_draws_triangle() {
        let mut fb = Framebuffer::new(32, 32);
        fb.arrowhead(20, 16, 1.0, 0.0, 8, Color::rgb(255, 0, 0));
        // Tip pixel must be set.
        let tip_off = (16 * 32 + 20) * 4;
        assert_eq!(fb.bytes()[tip_off + 2], 255);
    }
}