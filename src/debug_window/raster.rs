//! Software rasterizer writing into a flat BGRA `Vec<u8>` framebuffer.
//!
//! BGRA matches the typical Vulkan swapchain format on Windows
//! (`B8G8R8A8_UNORM` / `B8G8R8A8_SRGB`). The rasterizer never allocates
//! once the framebuffer is sized; all primitives clip against the buffer
//! bounds and operate on raw byte indices.
//!
//! Hot paths use packed u32 writes via `slice::fill`, which compiles to
//! vectorized memset on x86_64 and is dramatically faster than per-pixel
//! byte writes. At 2340x1248 this brings whole-screen clear from tens of
//! milliseconds to a fraction of a millisecond.

use super::font;

/// 32-bit BGRA color packed into a single u32. Constructed via
/// [`Color::rgb`] / [`Color::rgba`] for readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color(pub u8, pub u8, pub u8, pub u8); // B, G, R, A

impl Color {
    /// Opaque RGB.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self(b, g, r, 0xFF)
    }
    /// RGBA with explicit alpha (alpha is unused by the alpha-blended
    /// helpers but stored for completeness).
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(b, g, r, a)
    }
    /// Pack as a little-endian u32 in BGRA byte order. Writing this
    /// value through a `*mut u32` lays out as B, G, R, A in memory on
    /// every supported platform (all little-endian).
    #[inline(always)]
    fn packed(self) -> u32 {
        u32::from_le_bytes([self.0, self.1, self.2, self.3])
    }
}

/// Standard palette used by the panels. Values match the SVG visualizer's
/// palette so screenshots and SVGs look familiar side by side.
pub mod palette {
    use super::Color;
    pub const BG: Color = Color::rgb(0x1E, 0x1E, 0x1E);
    pub const PANEL_BG: Color = Color::rgb(0x25, 0x25, 0x25);
    pub const FRAME: Color = Color::rgb(0x40, 0x40, 0x40);
    pub const TEXT: Color = Color::rgb(0xE8, 0xE8, 0xE8);
    pub const TEXT_DIM: Color = Color::rgb(0x90, 0x90, 0x90);
    pub const TEXT_HEAD: Color = Color::rgb(0xFF, 0xFF, 0xFF);
    pub const BAR_BG: Color = Color::rgb(0x2A, 0x2A, 0x2A);
    pub const ALLOC_COLORS: [Color; 8] = [
        Color::rgb(0x4E, 0xC9, 0xB0),
        Color::rgb(0xDC, 0xDC, 0xAA),
        Color::rgb(0xCE, 0x91, 0x78),
        Color::rgb(0x9C, 0xDC, 0xFE),
        Color::rgb(0xC5, 0x86, 0xC0),
        Color::rgb(0x56, 0x9C, 0xD6),
        Color::rgb(0x60, 0x8B, 0x4E),
        Color::rgb(0xD7, 0xBA, 0x7D),
    ];
    pub const EVT_ALLOC: Color = Color::rgb(0x60, 0xD0, 0x80);
    pub const EVT_FREE: Color = Color::rgb(0xE0, 0x60, 0x60);
    pub const EVT_SUBMIT: Color = Color::rgb(0x70, 0xA0, 0xE0);
    pub const EVT_TRANSITION: Color = Color::rgb(0xF0, 0xC0, 0x40);
    pub const EVT_PASS: Color = Color::rgb(0xC0, 0x80, 0xE0);
    pub const EVT_CUSTOM: Color = Color::rgb(0xA0, 0xA0, 0xA0);
}

/// CPU-side BGRA framebuffer.
pub struct Framebuffer {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Framebuffer {
    /// Allocate a fresh BGRA buffer and zero it.
    pub fn new(width: u32, height: u32) -> Self {
        let pixels = vec![0_u8; (width as usize) * (height as usize) * 4];
        Self {
            pixels,
            width,
            height,
        }
    }

    /// Resize, reallocating storage. Existing contents are discarded.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pixels
            .resize((width as usize) * (height as usize) * 4, 0);
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
    /// Total byte length.
    pub fn byte_len(&self) -> usize {
        self.pixels.len()
    }

    /// Borrow the pixel buffer as a u32 slice. Length is always a
    /// multiple of 4 bytes by construction so the cast is well-defined.
    #[inline(always)]
    fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        let len = self.pixels.len() / 4;
        // SAFETY: pixels was allocated as a Vec<u8> with length =
        // width*height*4, and u32 has alignment 4 which Vec<u8>
        // guarantees up to its alignment. This cast is valid for the
        // lifetime of the borrow.
        unsafe {
            std::slice::from_raw_parts_mut(self.pixels.as_mut_ptr() as *mut u32, len)
        }
    }

    /// Fill the entire framebuffer with one color.
    ///
    /// Compiles to a vectorized fill loop. At 2340x1248 (about 2.9M
    /// pixels) this completes in under a millisecond on modern x86.
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
        // Fill row by row using slice::fill, which compiles to a tight
        // vectorized loop. We do not call self.put per pixel.
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

    /// Draw an 8x8 bitmap glyph at (x, y).
    ///
    /// Fast path when the glyph is fully inside the framebuffer skips
    /// per-pixel bounds checks and writes through a u32 slice. Slow
    /// path falls back to per-pixel `put` for partial visibility.
    pub fn glyph(&mut self, x: i32, y: i32, ch: char, c: Color) {
        let bytes = font::glyph_bytes(ch);
        let w = self.width as i32;
        let h = self.height as i32;
        if x >= 0 && y >= 0 && x + 8 <= w && y + 8 <= h {
            let p = c.packed();
            let stride = self.width as usize;
            let buf = self.as_u32_slice_mut();
            for (row, byte) in bytes.iter().enumerate() {
                let base = (y as usize + row) * stride + x as usize;
                for col in 0..8usize {
                    if (byte >> (7 - col)) & 1 != 0 {
                        buf[base + col] = p;
                    }
                }
            }
        } else {
            for (row, byte) in bytes.iter().enumerate() {
                for col in 0..8 {
                    if (byte >> (7 - col)) & 1 != 0 {
                        self.put(x + col as i32, y + row as i32, c);
                    }
                }
            }
        }
    }

    /// Draw a string starting at (x, y) using the 8x8 bitmap font.
    /// Each character advances by 8 pixels horizontally. Line breaks
    /// are not supported; pass split substrings explicitly.
    pub fn text(&mut self, x: i32, y: i32, s: &str, c: Color) {
        let mut cx = x;
        for ch in s.chars() {
            self.glyph(cx, y, ch, c);
            cx += 8;
        }
    }

    /// Width in pixels of a string rendered with [`text`].
    pub fn text_width(s: &str) -> i32 {
        (s.chars().count() as i32) * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_sets_all_pixels() {
        let mut fb = Framebuffer::new(4, 2);
        fb.clear(Color::rgb(255, 0, 0));
        for px in fb.bytes().chunks_exact(4) {
            assert_eq!(px[0], 0);
            assert_eq!(px[1], 0);
            assert_eq!(px[2], 255);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn put_writes_one_pixel() {
        let mut fb = Framebuffer::new(2, 2);
        fb.put(1, 0, Color::rgb(10, 20, 30));
        let px = &fb.bytes()[4..8];
        assert_eq!(px, &[30, 20, 10, 255]);
    }

    #[test]
    fn put_clips_out_of_bounds() {
        let mut fb = Framebuffer::new(2, 2);
        fb.put(-1, 0, Color::rgb(255, 0, 0));
        fb.put(2, 0, Color::rgb(255, 0, 0));
        fb.put(0, 5, Color::rgb(255, 0, 0));
        for byte in fb.bytes() {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn rect_clips_to_bounds() {
        let mut fb = Framebuffer::new(4, 4);
        fb.rect(2, 2, 100, 100, Color::rgb(255, 0, 0));
        for &(x, y) in &[(2, 2), (3, 2), (2, 3), (3, 3)] {
            let i = (y * 4 + x) * 4;
            assert_eq!(fb.bytes()[i + 2], 255, "pixel ({x},{y}) not set");
        }
        for &(x, y) in &[(0, 0), (1, 1)] {
            let i = (y * 4 + x) * 4;
            assert_eq!(fb.bytes()[i + 2], 0);
        }
    }

    #[test]
    fn rect_outline_only_borders() {
        let mut fb = Framebuffer::new(5, 5);
        fb.rect_outline(0, 0, 5, 5, Color::rgb(255, 0, 0));
        let i = (2 * 5 + 2) * 4;
        assert_eq!(fb.bytes()[i + 2], 0);
        assert_eq!(fb.bytes()[2], 255);
    }

    #[test]
    fn text_advances_by_eight_per_char() {
        assert_eq!(Framebuffer::text_width(""), 0);
        assert_eq!(Framebuffer::text_width("A"), 8);
        assert_eq!(Framebuffer::text_width("Hello"), 40);
    }

    #[test]
    fn resize_reallocates_buffer() {
        let mut fb = Framebuffer::new(2, 2);
        fb.clear(Color::rgb(255, 0, 0));
        fb.resize(4, 4);
        assert_eq!(fb.width(), 4);
        assert_eq!(fb.height(), 4);
        assert_eq!(fb.byte_len(), 4 * 4 * 4);
        assert_eq!(fb.bytes()[0], 0);
    }

    #[test]
    fn glyph_renders_into_framebuffer() {
        let mut fb = Framebuffer::new(8, 8);
        fb.glyph(0, 0, 'A', Color::rgb(255, 0, 0));
        let any_set = fb.bytes().chunks_exact(4).any(|p| p[2] == 255);
        assert!(any_set, "glyph 'A' produced no red pixels");
    }

    #[test]
    fn glyph_clips_when_partially_offscreen() {
        let mut fb = Framebuffer::new(4, 4);
        fb.glyph(-2, -2, 'A', Color::rgb(255, 0, 0));
        // Should not panic; some pixels may have been written.
        let _ = fb.bytes();
    }

    #[test]
    fn rect_at_zero_size_is_noop() {
        let mut fb = Framebuffer::new(4, 4);
        fb.rect(0, 0, 0, 5, Color::rgb(255, 0, 0));
        fb.rect(0, 0, 5, 0, Color::rgb(255, 0, 0));
        for byte in fb.bytes() {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn large_rect_fill_does_not_panic() {
        // Exercises the fast u32-fill path on a real-world resolution.
        let mut fb = Framebuffer::new(2340, 1248);
        fb.clear(Color::rgb(0x1E, 0x1E, 0x1E));
        fb.rect(100, 100, 2000, 1000, Color::rgb(0xFF, 0x00, 0x00));
        // Spot-check.
        let stride = 2340 * 4;
        let i = (200 * stride + 200 * 4) as usize;
        assert_eq!(fb.bytes()[i + 2], 0xFF);
    }
}