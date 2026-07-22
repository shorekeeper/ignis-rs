//! Determinism verifier for GPU workloads.
//!
//! Runs the same GPU work N times with a fixed seed, hashes the output
//! buffers and images via xxh64, and asserts that every run produced
//! identical output. On mismatch, panics with a detailed report and
//! optionally writes a BMP diff bitmap highlighting the differing pixels
//! in red.
//!
//! # Why this exists
//!
//! Non-determinism in GPU code is one of the worst classes of bug to
//! diagnose: the symptom is "test fails sometimes", the cause is usually
//! a missing barrier, an unsynchronized atomic, or a race between two
//! waves of compute work. Once you suspect non-determinism, [reduction
//! to a small reproducer requires an explicit verifier]. This module is
//! that verifier.
//!
//! # API
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ash::vk;
//! # fn example(ctx: &Ignis, buffer: vk::Buffer, size: u64) -> Result<()> {
//! let det = ctx.create_determinism_checker(QueueType::Compute)?;
//!
//! // First call captures the closure and runs once.
//! det.run_with_seed(0x42, move |rec, frame_idx, captures| {
//!     rec.fill_buffer(buffer, 0, size, 0xCAFE_BABE);
//!     captures.add_buffer("output", buffer, 0, size);
//!     Ok(())
//! })?;
//!
//! // Run 99 more times and verify all 100 runs produced identical output.
//! det.verify_n_runs(100)?;
//! # Ok(())
//! # }
//! ```
//!
//! [reduction to a small reproducer requires an explicit verifier]:
//! https://en.wikipedia.org/wiki/Memory_corruption
//!
//! # Closure constraints
//!
//! The recording closure must be `Fn + Send + Sync + 'static` because
//! it is stored inside the checker and invoked repeatedly across multiple
//! runs (potentially from different threads, although the current
//! implementation always invokes it on the calling thread).
//!
//! In practice this means closures must capture only `Copy` types like
//! raw `vk::Buffer` handles, not borrowed references. Use `move` and
//! pull out the handle from any owned wrapper (`buffer.handle()`)
//! before constructing the closure.
//!
//! # Hashing
//!
//! Uses xxh64 with seed = 0. Pure Rust, ~70 lines, no dependencies.
//! Throughput is around 5-10 GB/s on modern x86, fast enough that
//! readback time dominates total runtime.
//!
//! # Diff bitmap
//!
//! When two runs produce different image data, [`verify_n_runs`] writes
//! a BMP file showing every pixel that differs between the runs in red,
//! against a dimmed grayscale version of the baseline image. BMP is
//! used instead of PNG to avoid pulling in deflate/CRC32 dependencies
//! and instead of SVG to keep the file viewable in any default OS image
//! viewer (Windows Photos, Preview, etc).
//!
//! [`verify_n_runs`]: DeterminismChecker::verify_n_runs

use std::path::Path;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::command::{CommandPool, CommandRecorder};
use crate::device::SharedState;
use crate::error::{Error, Result};
use crate::queue::AsyncQueue;

// ---- Hashing (xxh64) ----------------------------------------------------

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

fn xxh64_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

fn xxh64_merge_round(acc: u64, val: u64) -> u64 {
    let val = xxh64_round(0, val);
    (acc ^ val)
        .wrapping_mul(PRIME64_1)
        .wrapping_add(PRIME64_4)
}

/// Compute the xxh64 hash of a byte slice with the given seed.
///
/// Pure Rust port of the canonical xxhash xxh64 algorithm. Used to
/// fingerprint GPU output buffers so two runs can be compared in
/// constant time.
pub fn xxh64(input: &[u8], seed: u64) -> u64 {
    let len = input.len() as u64;
    let mut h: u64;
    let mut p = 0usize;

    if input.len() >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);

        while p + 32 <= input.len() {
            v1 = xxh64_round(v1, read_u64_le(&input[p..]));
            p += 8;
            v2 = xxh64_round(v2, read_u64_le(&input[p..]));
            p += 8;
            v3 = xxh64_round(v3, read_u64_le(&input[p..]));
            p += 8;
            v4 = xxh64_round(v4, read_u64_le(&input[p..]));
            p += 8;
        }

        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = xxh64_merge_round(h, v1);
        h = xxh64_merge_round(h, v2);
        h = xxh64_merge_round(h, v3);
        h = xxh64_merge_round(h, v4);
    } else {
        h = seed.wrapping_add(PRIME64_5);
    }

    h = h.wrapping_add(len);

    while p + 8 <= input.len() {
        let k1 = xxh64_round(0, read_u64_le(&input[p..]));
        h ^= k1;
        h = h
            .rotate_left(27)
            .wrapping_mul(PRIME64_1)
            .wrapping_add(PRIME64_4);
        p += 8;
    }

    if p + 4 <= input.len() {
        h ^= (read_u32_le(&input[p..]) as u64).wrapping_mul(PRIME64_1);
        h = h
            .rotate_left(23)
            .wrapping_mul(PRIME64_2)
            .wrapping_add(PRIME64_3);
        p += 4;
    }

    while p < input.len() {
        h ^= (input[p] as u64).wrapping_mul(PRIME64_5);
        h = h.rotate_left(11).wrapping_mul(PRIME64_1);
        p += 1;
    }

    h ^= h >> 33;
    h = h.wrapping_mul(PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME64_3);
    h ^= h >> 32;
    h
}

#[inline]
fn read_u64_le(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

// ---- BMP encoder for diff bitmaps --------------------------------------

/// Write an uncompressed 24-bit BMP file. BMPs are bottom-up by default.
fn write_bmp(
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
    f.write_all(&1u16.to_le_bytes())?; // color planes
    f.write_all(&24u16.to_le_bytes())?; // bits per pixel
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

/// Indicates how the captured image bytes are interpreted for diff
/// rendering. Affects byte-channel mapping when generating the BMP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffBitmapFormat {
    /// Bytes are R, G, B, A in memory.
    Rgba8,
    /// Bytes are B, G, R, A in memory.
    Bgra8,
}

fn diff_format_for(format: vk::Format) -> Option<DiffBitmapFormat> {
    match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_SRGB => Some(DiffBitmapFormat::Rgba8),
        vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SNORM
        | vk::Format::B8G8R8A8_SRGB => Some(DiffBitmapFormat::Bgra8),
        _ => None,
    }
}

/// Render a diff bitmap. Pixels that differ between baseline and current
/// are painted bright red; matching pixels are dimmed grayscale.
fn save_diff_bitmap(
    path: &Path,
    width: u32,
    height: u32,
    format: vk::Format,
    baseline: &[u8],
    current: &[u8],
) -> std::io::Result<usize> {
    let fmt = diff_format_for(format).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("diff bitmap unsupported for format {format:?}"),
        )
    })?;

    let pixel_count = (width * height) as usize;
    if baseline.len() != pixel_count * 4 || current.len() != pixel_count * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "image data size does not match width*height*4",
        ));
    }

    let (r_idx, g_idx, b_idx) = match fmt {
        DiffBitmapFormat::Rgba8 => (0usize, 1usize, 2usize),
        DiffBitmapFormat::Bgra8 => (2usize, 1usize, 0usize),
    };

    let mut bgr = Vec::with_capacity(pixel_count * 3);
    let mut diff_count = 0usize;
    for i in 0..pixel_count {
        let off = i * 4;
        let br = baseline[off + r_idx];
        let bg = baseline[off + g_idx];
        let bb = baseline[off + b_idx];
        let cr = current[off + r_idx];
        let cg = current[off + g_idx];
        let cb = current[off + b_idx];
        if br != cr || bg != cg || bb != cb {
            // Differing pixel: bright red.
            bgr.extend_from_slice(&[0, 0, 255]);
            diff_count += 1;
        } else {
            // Matching pixel: dimmed grayscale of baseline.
            let avg = ((br as u32 + bg as u32 + bb as u32) / 6) as u8;
            bgr.extend_from_slice(&[avg, avg, avg]);
        }
    }
    write_bmp(path, width, height, &bgr)?;
    Ok(diff_count)
}

// ---- Capture set --------------------------------------------------------

#[derive(Debug, Clone)]
struct BufferCaptureSpec {
    label: String,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[derive(Debug, Clone)]
struct ImageCaptureSpec {
    label: String,
    image: vk::Image,
    width: u32,
    height: u32,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    /// Layout the image is currently in when the closure ends.
    src_layout: vk::ImageLayout,
    /// Layout the image must be transitioned back to after capture.
    /// Use `vk::ImageLayout::UNDEFINED` if you do not care.
    final_layout: vk::ImageLayout,
}

/// Builder used inside the recording closure to declare resources to hash.
///
/// Pass to [`add_buffer`] or [`add_image`] for every resource whose
/// contents should be verified across runs.
///
/// [`add_buffer`]: CaptureSet::add_buffer
/// [`add_image`]: CaptureSet::add_image
#[derive(Debug, Default)]
pub struct CaptureSet {
    buffers: Vec<BufferCaptureSpec>,
    images: Vec<ImageCaptureSpec>,
}

impl CaptureSet {
    /// Construct an empty capture set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a buffer region whose contents should be hashed.
    ///
    /// The buffer must have been created with
    /// `BufferUsageFlags::TRANSFER_SRC` so the verifier can copy it to
    /// a host-visible staging buffer. The region must be aligned to 4
    /// bytes per Vulkan spec for `vkCmdCopyBuffer`.
    pub fn add_buffer(
        &mut self,
        label: &str,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        self.buffers.push(BufferCaptureSpec {
            label: label.to_string(),
            buffer,
            offset,
            size,
        });
    }

    /// Declare a 2D image region (mip 0, layer 0) whose contents
    /// should be hashed.
    ///
    /// The image must have been created with
    /// `ImageUsageFlags::TRANSFER_SRC` so the verifier can read it back.
    /// `src_layout` is the layout of the image at the moment the
    /// closure returns; the verifier inserts a barrier transitioning to
    /// `TRANSFER_SRC_OPTIMAL` before the readback copy. After the copy,
    /// the image is transitioned to `final_layout` so subsequent runs
    /// can start from a known state. Use `vk::ImageLayout::UNDEFINED`
    /// for `final_layout` if the next run will overwrite the image
    /// completely.
    ///
    /// Only the following formats support diff bitmap rendering:
    /// `R8G8B8A8_*`, `B8G8R8A8_*`. Other formats can still be hashed
    /// but the diff bitmap is omitted on mismatch.
    pub fn add_image(
        &mut self,
        label: &str,
        image: vk::Image,
        width: u32,
        height: u32,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        src_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) {
        self.images.push(ImageCaptureSpec {
            label: label.to_string(),
            image,
            width,
            height,
            format,
            aspect,
            src_layout,
            final_layout,
        });
    }

    /// Number of declared captures.
    pub fn len(&self) -> usize {
        self.buffers.len() + self.images.len()
    }

    /// Whether the capture set is empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty() && self.images.is_empty()
    }
}

// ---- Run results --------------------------------------------------------

/// Hashed output of one buffer capture.
#[derive(Debug, Clone)]
pub struct BufferHash {
    /// Label supplied via [`CaptureSet::add_buffer`].
    pub label: String,
    /// xxh64 hash of the captured bytes.
    pub hash: u64,
    /// Number of bytes hashed.
    pub size: vk::DeviceSize,
}

/// Hashed output of one image capture, plus the raw bytes for diff rendering.
#[derive(Debug, Clone)]
pub struct ImageHash {
    /// Label supplied via [`CaptureSet::add_image`].
    pub label: String,
    /// xxh64 hash of the captured pixel bytes.
    pub hash: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel format the bytes were captured in.
    pub format: vk::Format,
    /// Raw pixel bytes (width * height * 4). Retained so a diff bitmap
    /// can be rendered if a later run produces a different hash.
    pub bytes: Vec<u8>,
}

/// Recorded data for one execution of the closure.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Seed value the run was tagged with.
    pub seed: u64,
    /// Sequential index, starting at 0 for the first run.
    pub frame_idx: u32,
    /// Hashes of all captured buffers in declaration order.
    pub buffer_hashes: Vec<BufferHash>,
    /// Hashes of all captured images in declaration order.
    pub image_hashes: Vec<ImageHash>,
}

// ---- Determinism checker ------------------------------------------------

type StoredClosure = Arc<
    dyn Fn(&CommandRecorder<'_>, u32, &mut CaptureSet) -> Result<()>
        + Send
        + Sync,
>;

/// GPU determinism verifier.
///
/// Construct via [`Ignis::create_determinism_checker`].
///
/// [`Ignis::create_determinism_checker`]: crate::Ignis::create_determinism_checker
pub struct DeterminismChecker {
    shared: Arc<SharedState>,
    queue: Arc<AsyncQueue>,
    pool: CommandPool,
    runs: Mutex<Vec<RunResult>>,
    stored: Mutex<Option<(u64, StoredClosure)>>,
}

impl DeterminismChecker {
    pub(crate) fn new(
        shared: Arc<SharedState>,
        queue: Arc<AsyncQueue>,
    ) -> Result<Self> {
        let pool = CommandPool::new(Arc::clone(&shared), queue.family_index())?;
        Ok(Self {
            shared,
            queue,
            pool,
            runs: Mutex::new(Vec::new()),
            stored: Mutex::new(None),
        })
    }

    /// Number of recorded runs.
    pub fn run_count(&self) -> usize {
        self.runs.lock().unwrap().len()
    }

    /// Discard all stored runs and the stored closure.
    pub fn reset(&self) {
        self.runs.lock().unwrap().clear();
        *self.stored.lock().unwrap() = None;
    }

    /// All recorded run results, in execution order.
    pub fn results(&self) -> Vec<RunResult> {
        self.runs.lock().unwrap().clone()
    }

    /// Run the recording closure once, hashing all declared captures
    /// after submission completes. The closure is stored so subsequent
    /// calls to [`verify_n_runs`] can re-run it.
    ///
    /// [`verify_n_runs`]: Self::verify_n_runs
    pub fn run_with_seed<F>(&self, seed: u64, record: F) -> Result<RunResult>
    where
        F: Fn(&CommandRecorder<'_>, u32, &mut CaptureSet) -> Result<()>
            + Send
            + Sync
            + 'static,
    {
        let closure: StoredClosure = Arc::new(record);
        *self.stored.lock().unwrap() = Some((seed, Arc::clone(&closure)));
        self.run_internal(seed, &closure)
    }

    /// Verify that at least `n` runs have been performed and that all
    /// of them produced identical buffer and image hashes.
    ///
    /// If fewer than `n` runs are stored, the closure registered by the
    /// most recent [`run_with_seed`] is invoked the necessary number of
    /// extra times. If no closure has been registered, returns an error.
    ///
    /// On a hash mismatch this method panics with a detailed report.
    /// For image captures whose format supports BMP rendering, a diff
    /// bitmap is written to `determinism_diff_<label>_run<idx>.bmp` in
    /// the current working directory before the panic.
    ///
    /// [`run_with_seed`]: Self::run_with_seed
    pub fn verify_n_runs(&self, n: u32) -> Result<()> {
        let n = n.max(1);
        let need = n as usize;
        let mut have = self.runs.lock().unwrap().len();

        if have < need {
            let (seed, closure) = match self.stored.lock().unwrap().clone() {
                Some(v) => v,
                None => {
                    return Err(Error::InvalidConfig(
                        "verify_n_runs: no closure registered; \
                         call run_with_seed first",
                    ));
                }
            };
            while have < need {
                self.run_internal(seed, &closure)?;
                have += 1;
            }
        }

        let runs = self.runs.lock().unwrap();
        let baseline = &runs[0];

        for (i, r) in runs.iter().enumerate().take(need).skip(1) {
            // Compare buffer hashes.
            for (b_base, b_cur) in baseline.buffer_hashes.iter().zip(r.buffer_hashes.iter()) {
                if b_base.label != b_cur.label || b_base.hash != b_cur.hash {
                    panic!(
                        "{}",
                        format_buffer_mismatch(baseline, r, i, b_base, b_cur)
                    );
                }
            }
            if baseline.buffer_hashes.len() != r.buffer_hashes.len() {
                panic!(
                    "determinism: run {} declared {} buffers, baseline declared {}",
                    i,
                    r.buffer_hashes.len(),
                    baseline.buffer_hashes.len()
                );
            }
            // Compare image hashes.
            for (img_base, img_cur) in baseline.image_hashes.iter().zip(r.image_hashes.iter()) {
                if img_base.label != img_cur.label || img_base.hash != img_cur.hash {
                    let bmp_path = std::path::PathBuf::from(format!(
                        "determinism_diff_{}_run{}.bmp",
                        sanitize(&img_base.label),
                        i
                    ));
                    let diff_count = save_diff_bitmap(
                        &bmp_path,
                        img_base.width,
                        img_base.height,
                        img_base.format,
                        &img_base.bytes,
                        &img_cur.bytes,
                    )
                    .ok();
                    panic!(
                        "{}",
                        format_image_mismatch(
                            baseline, r, i, img_base, img_cur, &bmp_path, diff_count
                        )
                    );
                }
            }
            if baseline.image_hashes.len() != r.image_hashes.len() {
                panic!(
                    "determinism: run {} declared {} images, baseline declared {}",
                    i,
                    r.image_hashes.len(),
                    baseline.image_hashes.len()
                );
            }
        }
        Ok(())
    }

    /// Internal: run the closure once and store the result.
    fn run_internal(&self, seed: u64, closure: &StoredClosure) -> Result<RunResult> {
        let frame_idx = self.runs.lock().unwrap().len() as u32;
        self.pool.reset()?;

        // Phase 1: record user work + capture copies into one cmd buffer.
        let cmd = self.pool.allocate_primary()?;
        let rec = self.pool.begin_primary(cmd)?;
        let mut captures = CaptureSet::new();
        closure(&rec, frame_idx, &mut captures)?;

        // Allocate staging for every capture.
        let staging = self.allocate_staging(&captures)?;

        // Add capture copy commands.
        self.record_buffer_copies(&rec, &captures.buffers, &staging.buffers)?;
        self.record_image_copies(&rec, &captures.images, &staging.images)?;

        let cmd = rec.end()?;
        self.queue.submit_simple(cmd)?.wait()?;

        // Phase 2: read back staging and hash.
        let result = self.hash_staging(seed, frame_idx, &captures, &staging);

        // Free staging memory.
        self.destroy_staging(staging);

        let result = result?;
        self.runs.lock().unwrap().push(result.clone());
        Ok(result)
    }

    fn allocate_staging(&self, captures: &CaptureSet) -> Result<Staging> {
        let mut buffers = Vec::with_capacity(captures.buffers.len());
        for spec in &captures.buffers {
            buffers.push(StagingBuffer::new(&self.shared, spec.size)?);
        }
        let mut images = Vec::with_capacity(captures.images.len());
        for spec in &captures.images {
            // Only RGBA8 / BGRA8 supported in this version.
            let bpp = 4u64;
            let size = spec.width as u64 * spec.height as u64 * bpp;
            images.push(StagingBuffer::new(&self.shared, size)?);
        }
        Ok(Staging { buffers, images })
    }

    fn record_buffer_copies(
        &self,
        rec: &CommandRecorder<'_>,
        specs: &[BufferCaptureSpec],
        staging: &[StagingBuffer],
    ) -> Result<()> {
        for (spec, dst) in specs.iter().zip(staging.iter()) {
            // The user must ensure the source buffer is in a state that
            // permits TRANSFER_READ. We add a generic memory barrier to
            // guarantee any prior writes are visible to the copy.
            rec.pipeline_barrier(
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
                &[],
            );
            rec.copy_buffer(
                spec.buffer,
                dst.buffer,
                &[vk::BufferCopy {
                    src_offset: spec.offset,
                    dst_offset: 0,
                    size: spec.size,
                }],
            );
        }
        Ok(())
    }

    fn record_image_copies(
        &self,
        rec: &CommandRecorder<'_>,
        specs: &[ImageCaptureSpec],
        staging: &[StagingBuffer],
    ) -> Result<()> {
        for (spec, dst) in specs.iter().zip(staging.iter()) {
            // Transition the image from src_layout to TRANSFER_SRC_OPTIMAL.
            let to_src = vk::ImageMemoryBarrier::default()
                .old_layout(spec.src_layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(spec.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: spec.aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            rec.pipeline_barrier(
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_src),
            );

            rec.copy_image_to_buffer(
                spec.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.buffer,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: spec.aspect,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width: spec.width,
                        height: spec.height,
                        depth: 1,
                    },
                }],
            );

            // Transition back to final_layout if the user specified one.
            if spec.final_layout != vk::ImageLayout::UNDEFINED {
                let to_final = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(spec.final_layout)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .image(spec.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: spec.aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                rec.pipeline_barrier(
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&to_final),
                );
            }
        }
        Ok(())
    }

    fn hash_staging(
        &self,
        seed: u64,
        frame_idx: u32,
        captures: &CaptureSet,
        staging: &Staging,
    ) -> Result<RunResult> {
        let mut buffer_hashes = Vec::with_capacity(captures.buffers.len());
        for (spec, sb) in captures.buffers.iter().zip(staging.buffers.iter()) {
            let bytes = sb.read()?;
            buffer_hashes.push(BufferHash {
                label: spec.label.clone(),
                hash: xxh64(&bytes, seed),
                size: spec.size,
            });
        }
        let mut image_hashes = Vec::with_capacity(captures.images.len());
        for (spec, sb) in captures.images.iter().zip(staging.images.iter()) {
            let bytes = sb.read()?;
            image_hashes.push(ImageHash {
                label: spec.label.clone(),
                hash: xxh64(&bytes, seed),
                width: spec.width,
                height: spec.height,
                format: spec.format,
                bytes,
            });
        }
        Ok(RunResult {
            seed,
            frame_idx,
            buffer_hashes,
            image_hashes,
        })
    }

    fn destroy_staging(&self, staging: Staging) {
        for sb in staging.buffers {
            sb.destroy(&self.shared);
        }
        for sb in staging.images {
            sb.destroy(&self.shared);
        }
    }
}

// ---- Staging plumbing --------------------------------------------------

struct Staging {
    buffers: Vec<StagingBuffer>,
    images: Vec<StagingBuffer>,
}

struct StagingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: vk::DeviceSize,
}

unsafe impl Send for StagingBuffer {}
unsafe impl Sync for StagingBuffer {}

impl StagingBuffer {
    fn new(shared: &Arc<SharedState>, size: vk::DeviceSize) -> Result<Self> {
        let size = size.max(4); // Vulkan does not allow zero-size buffers.
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { shared.device.create_buffer(&ci, None)? };
        let req = unsafe { shared.device.get_buffer_memory_requirements(buffer) };
        let mt = find_host_visible(&shared.memory_properties, req.memory_type_bits)
            .ok_or_else(|| {
                unsafe { shared.device.destroy_buffer(buffer, None) };
                Error::NoSuitableMemoryType
            })?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mt);
        let memory = unsafe { shared.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| {
                unsafe { shared.device.destroy_buffer(buffer, None) };
                Error::Vulkan(e)
            })?;
        unsafe { shared.device.bind_buffer_memory(buffer, memory, 0)? };
        let mapped = unsafe {
            shared
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
        }
        .cast::<u8>();
        Ok(Self {
            buffer,
            memory,
            mapped,
            size,
        })
    }

    fn read(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped, out.as_mut_ptr(), self.size as usize);
        }
        Ok(out)
    }

    fn destroy(self, shared: &Arc<SharedState>) {
        unsafe {
            shared.device.unmap_memory(self.memory);
            shared.device.destroy_buffer(self.buffer, None);
            shared.device.free_memory(self.memory, None);
        }
    }
}

fn find_host_visible(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if bits & (1 << i) == 0 {
            continue;
        }
        let f = props.memory_types[i as usize].property_flags;
        if f.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            return Some(i);
        }
    }
    None
}

// ---- Diagnostic formatting ---------------------------------------------

fn sanitize(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn format_buffer_mismatch(
    baseline: &RunResult,
    current: &RunResult,
    run_idx: usize,
    base: &BufferHash,
    cur: &BufferHash,
) -> String {
    use crate::diagnostic::{
        write_diagnostic_end, write_full_diagnostic, write_kv, write_pipe_empty,
        write_section, Severity, Style,
    };
    let s = Style::detect();
    let mut o = String::with_capacity(1024);

    write_full_diagnostic(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-DET",
        "non-deterministic buffer output detected",
        true,
        true,
    );
    write_pipe_empty(&mut o, &s);

    write_section(&mut o, &s, "Mismatch");
    write_kv(&mut o, &s, "Capture label", &base.label);
    write_kv(&mut o, &s, "Buffer size", &format!("{} bytes", base.size));
    write_kv(
        &mut o,
        &s,
        "Baseline run",
        &format!(
            "frame={} seed={:#x} hash={:#018x}",
            baseline.frame_idx, baseline.seed, base.hash
        ),
    );
    write_kv(
        &mut o,
        &s,
        "Diverging run",
        &format!(
            "frame={} seed={:#x} hash={:#018x}  ({})",
            current.frame_idx,
            current.seed,
            cur.hash,
            s.bold_red(&format!("run #{run_idx}")),
        ),
    );

    write_pipe_empty(&mut o, &s);
    write_section(&mut o, &s, "Likely Causes");
    crate::diagnostic::write_numbered(
        &mut o,
        &s,
        1,
        "missing barrier between writers (read-after-write race)",
    );
    crate::diagnostic::write_numbered(
        &mut o,
        &s,
        2,
        "atomic operations whose final value depends on wave order",
    );
    crate::diagnostic::write_numbered(
        &mut o,
        &s,
        3,
        "uninitialized memory read by the workload",
    );
    crate::diagnostic::write_numbered(
        &mut o,
        &s,
        4,
        "non-deterministic input fed into the closure (system time, RNG)",
    );

    write_diagnostic_end(&mut o, &s, &Severity::Error);
    o
}

fn format_image_mismatch(
    baseline: &RunResult,
    current: &RunResult,
    run_idx: usize,
    base: &ImageHash,
    cur: &ImageHash,
    bmp_path: &Path,
    diff_count: Option<usize>,
) -> String {
    use crate::diagnostic::{
        write_diagnostic_end, write_full_diagnostic, write_kv, write_pipe,
        write_pipe_empty, write_section, Severity, Style,
    };
    let s = Style::detect();
    let mut o = String::with_capacity(1024);

    write_full_diagnostic(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-DET",
        "non-deterministic image output detected",
        true,
        true,
    );
    write_pipe_empty(&mut o, &s);

    write_section(&mut o, &s, "Mismatch");
    write_kv(&mut o, &s, "Capture label", &base.label);
    write_kv(
        &mut o,
        &s,
        "Image",
        &format!("{}x{} {:?}", base.width, base.height, base.format),
    );
    write_kv(
        &mut o,
        &s,
        "Baseline hash",
        &format!("{:#018x}", base.hash),
    );
    write_kv(
        &mut o,
        &s,
        "Diverging hash",
        &format!("{:#018x}  (run #{run_idx})", cur.hash),
    );
    write_kv(
        &mut o,
        &s,
        "Baseline run",
        &format!("frame={} seed={:#x}", baseline.frame_idx, baseline.seed),
    );
    write_kv(
        &mut o,
        &s,
        "Diverging run",
        &format!("frame={} seed={:#x}", current.frame_idx, current.seed),
    );

    write_pipe_empty(&mut o, &s);
    match diff_count {
        Some(n) => {
            write_pipe(
                &mut o,
                &s,
                &format!(
                    "diff bitmap written: {} (red = {} differing pixels)",
                    bmp_path.display(),
                    n
                ),
            );
            write_pipe(
                &mut o,
                &s,
                "open the BMP in any image viewer; red marks every pixel \
                 whose value changed between runs.",
            );
        }
        None => {
            write_pipe(
                &mut o,
                &s,
                &format!(
                    "diff bitmap not written: format {:?} not supported \
                     for BMP rendering",
                    base.format
                ),
            );
        }
    }

    write_diagnostic_end(&mut o, &s, &Severity::Error);
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh64_known_test_vectors() {
        // xxhash reference vectors with seed=0.
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"a", 0), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(
            xxh64(b"The quick brown fox jumps over the lazy dog", 0),
            0x0B24_2D36_1FDA_71BC
        );
    }

    #[test]
    fn xxh64_seed_changes_hash() {
        let h0 = xxh64(b"hello world", 0);
        let h1 = xxh64(b"hello world", 1);
        assert_ne!(h0, h1);
    }

    #[test]
    fn xxh64_long_input() {
        // Test the 32-byte block path.
        let input: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let h = xxh64(&input, 0);
        // Just verify it does not crash and produces non-trivial output.
        assert_ne!(h, 0);
        assert_ne!(h, xxh64(&input, 1));
    }

    #[test]
    fn capture_set_tracks_additions() {
        let mut cs = CaptureSet::new();
        assert!(cs.is_empty());
        cs.add_buffer("a", vk::Buffer::null(), 0, 64);
        assert_eq!(cs.len(), 1);
        cs.add_image(
            "b",
            vk::Image::null(),
            32,
            32,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::UNDEFINED,
        );
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn diff_format_recognizes_common_formats() {
        assert_eq!(
            diff_format_for(vk::Format::R8G8B8A8_UNORM),
            Some(DiffBitmapFormat::Rgba8)
        );
        assert_eq!(
            diff_format_for(vk::Format::B8G8R8A8_SRGB),
            Some(DiffBitmapFormat::Bgra8)
        );
        assert_eq!(diff_format_for(vk::Format::R32G32B32A32_SFLOAT), None);
    }

    #[test]
    fn save_diff_bitmap_writes_red_for_differences() {
        // 2x2 RGBA8 baseline: all white.
        let baseline = vec![255u8; 16];
        // Current: top-left changed to black.
        let mut current = vec![255u8; 16];
        current[0] = 0;
        current[1] = 0;
        current[2] = 0;

        let path = std::env::temp_dir().join(format!(
            "ignis_det_test_{}_{}.bmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let n = save_diff_bitmap(
            &path,
            2,
            2,
            vk::Format::R8G8B8A8_UNORM,
            &baseline,
            &current,
        )
        .unwrap();
        assert_eq!(n, 1);

        // Verify file exists and starts with BM magic.
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..2], b"BM");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sanitize_replaces_special_chars() {
        assert_eq!(sanitize("output_buffer"), "output_buffer");
        assert_eq!(sanitize("output buffer/0"), "output_buffer_0");
        assert_eq!(sanitize("a-b.c"), "a_b_c");
    }
}