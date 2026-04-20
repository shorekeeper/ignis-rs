//! Frame graph: declarative pass ordering with automatic barrier computation.
//!
//! Users declare passes together with their resource reads and writes.
//! The graph performs topological sort, issues minimum barriers between
//! passes, and invokes each pass's execute closure in order.
//!
//! # Supported now
//!
//! - Image and buffer resource declaration (transient or imported)
//! - Read/write access lists per pass with `ImageUsageContext` /
//!   `BufferUsageContext`
//! - Dependency analysis via resource aliasing
//! - Automatic barrier emission using the existing `ResourceTracker`
//! - Topological sort using Kahn's algorithm
//!
//! # Not yet
//!
//! - Transient memory aliasing (two non-overlapping attachments sharing
//!   device memory). The descriptor layer exists; aliasing is a future
//!   extension.
//! - Async compute scheduling across queues (all passes currently run
//!   on the queue passed to `execute`).
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ignis::framegraph::*;
//! # use ash::vk;
//! # fn example(ignis: &Ignis, queue: &AsyncQueue, pool: &CommandPool) -> Result<()> {
//! let mut fg = FrameGraph::new();
//!
//! let gbuffer = fg.declare_image("gbuffer", ImageDesc {
//!     width: 1920, height: 1080,
//!     format: vk::Format::R16G16B16A16_SFLOAT,
//!     usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
//! });
//! let depth = fg.declare_image("depth", ImageDesc::depth(1920, 1080, vk::Format::D32_SFLOAT));
//!
//! fg.add_pass("geometry", |pass| {
//!     pass.writes_image(gbuffer, ImageUsageContext::ColorAttachment);
//!     pass.writes_image(depth, ImageUsageContext::DepthStencilAttachment);
//!     pass.execute(Box::new(|_rec, _resolver| {
//!         // record draw calls
//!     }));
//! });
//!
//! fg.add_pass("lighting", |pass| {
//!     pass.reads_image(gbuffer, ImageUsageContext::FragmentShaderRead);
//!     pass.reads_image(depth, ImageUsageContext::FragmentShaderRead);
//!     pass.execute(Box::new(|_rec, _resolver| {
//!         // record lighting draw
//!     }));
//! });
//!
//! let compiled = fg.compile(ignis)?;
//! compiled.execute(ignis, pool, queue)?;
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ash::vk;

use crate::command::CommandRecorder;
use crate::device::SharedState;
use crate::error::{Error, Result};
use crate::memory::allocator::Allocator;
use crate::memory::resources::{Buffer, BufferInfo, Image, ImageInfo, MemoryLocation};

#[cfg(feature = "tracking")]
use crate::tracking::tracker::{BufferUsageContext, ImageUsageContext, ResourceTracker};

/// Opaque handle identifying a declared image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(u32);

/// Opaque handle identifying a declared buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u32);

/// Descriptor for a transient image to be allocated by the graph.
#[derive(Debug, Clone)]
pub struct ImageDesc {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Pixel format.
    pub format: vk::Format,
    /// Combined usage flags. Must include every way the image will be used.
    pub usage: vk::ImageUsageFlags,
}

impl ImageDesc {
    /// Shorthand for a color target.
    pub fn color(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        }
    }

    /// Shorthand for a depth target.
    pub fn depth(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        }
    }
}

/// Descriptor for a transient buffer to be allocated by the graph.
#[derive(Debug, Clone)]
pub struct BufferDesc {
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Memory location hint.
    pub location: MemoryLocation,
}

enum ImageSource {
    Transient(ImageDesc),
    Imported {
        handle: vk::Image,
        initial_layout: vk::ImageLayout,
        mip_levels: u32,
        array_layers: u32,
        aspect: vk::ImageAspectFlags,
    },
}

enum BufferSource {
    Transient(BufferDesc),
    Imported(vk::Buffer),
}

struct ImageDecl {
    name: String,
    source: ImageSource,
}

struct BufferDecl {
    name: String,
    source: BufferSource,
}

#[cfg(feature = "tracking")]
#[derive(Clone, Copy)]
enum ImageAccessMode {
    Read(ImageUsageContext),
    Write(ImageUsageContext),
}

#[cfg(feature = "tracking")]
#[derive(Clone, Copy)]
enum BufferAccessMode {
    Read(BufferUsageContext),
    Write(BufferUsageContext),
}

#[cfg(feature = "tracking")]
struct PassAccess {
    image: Vec<(ImageHandle, ImageAccessMode)>,
    buffer: Vec<(BufferHandle, BufferAccessMode)>,
}

#[cfg(not(feature = "tracking"))]
struct PassAccess {
    _phantom: (),
}

/// Callback invoked by the graph when a pass executes.
///
/// Receives the command recorder and a `Resolver` for mapping handles to
/// actual Vulkan resources.
pub type PassExecute = Box<dyn FnOnce(&CommandRecorder<'_>, &Resolver) + Send>;

struct PassDecl {
    name: String,
    access: PassAccess,
    execute: Option<PassExecute>,
}

/// Builder used inside `FrameGraph::add_pass`.
pub struct PassBuilder<'a> {
    pass: &'a mut PassDecl,
}

impl<'a> PassBuilder<'a> {
    /// Declare that the pass reads an image with the given usage.
    #[cfg(feature = "tracking")]
    pub fn reads_image(&mut self, handle: ImageHandle, usage: ImageUsageContext) -> &mut Self {
        self.pass
            .access
            .image
            .push((handle, ImageAccessMode::Read(usage)));
        self
    }

    /// Declare that the pass writes an image with the given usage.
    #[cfg(feature = "tracking")]
    pub fn writes_image(&mut self, handle: ImageHandle, usage: ImageUsageContext) -> &mut Self {
        self.pass
            .access
            .image
            .push((handle, ImageAccessMode::Write(usage)));
        self
    }

    /// Declare that the pass reads a buffer with the given usage.
    #[cfg(feature = "tracking")]
    pub fn reads_buffer(&mut self, handle: BufferHandle, usage: BufferUsageContext) -> &mut Self {
        self.pass
            .access
            .buffer
            .push((handle, BufferAccessMode::Read(usage)));
        self
    }

    /// Declare that the pass writes a buffer with the given usage.
    #[cfg(feature = "tracking")]
    pub fn writes_buffer(&mut self, handle: BufferHandle, usage: BufferUsageContext) -> &mut Self {
        self.pass
            .access
            .buffer
            .push((handle, BufferAccessMode::Write(usage)));
        self
    }

    /// Install the execute closure.
    pub fn execute(&mut self, cb: PassExecute) -> &mut Self {
        self.pass.execute = Some(cb);
        self
    }
}

/// The graph under construction.
pub struct FrameGraph {
    images: Vec<ImageDecl>,
    buffers: Vec<BufferDecl>,
    passes: Vec<PassDecl>,
}

impl FrameGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            buffers: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// Declare a transient image managed by the graph.
    pub fn declare_image(&mut self, name: impl Into<String>, desc: ImageDesc) -> ImageHandle {
        let h = ImageHandle(self.images.len() as u32);
        self.images.push(ImageDecl {
            name: name.into(),
            source: ImageSource::Transient(desc),
        });
        h
    }

    /// Import an externally owned image. The graph will not free it.
    pub fn import_image(
        &mut self,
        name: impl Into<String>,
        image: vk::Image,
        initial_layout: vk::ImageLayout,
        mip_levels: u32,
        array_layers: u32,
        aspect: vk::ImageAspectFlags,
    ) -> ImageHandle {
        let h = ImageHandle(self.images.len() as u32);
        self.images.push(ImageDecl {
            name: name.into(),
            source: ImageSource::Imported {
                handle: image,
                initial_layout,
                mip_levels,
                array_layers,
                aspect,
            },
        });
        h
    }

    /// Declare a transient buffer.
    pub fn declare_buffer(&mut self, name: impl Into<String>, desc: BufferDesc) -> BufferHandle {
        let h = BufferHandle(self.buffers.len() as u32);
        self.buffers.push(BufferDecl {
            name: name.into(),
            source: BufferSource::Transient(desc),
        });
        h
    }

    /// Import an externally owned buffer.
    pub fn import_buffer(&mut self, name: impl Into<String>, buffer: vk::Buffer) -> BufferHandle {
        let h = BufferHandle(self.buffers.len() as u32);
        self.buffers.push(BufferDecl {
            name: name.into(),
            source: BufferSource::Imported(buffer),
        });
        h
    }

    /// Register a pass with the graph.
    pub fn add_pass<F>(&mut self, name: impl Into<String>, build: F) -> &mut Self
    where
        F: FnOnce(&mut PassBuilder<'_>),
    {
        let mut pass = PassDecl {
            name: name.into(),
            #[cfg(feature = "tracking")]
            access: PassAccess {
                image: Vec::new(),
                buffer: Vec::new(),
            },
            #[cfg(not(feature = "tracking"))]
            access: PassAccess { _phantom: () },
            execute: None,
        };
        let mut builder = PassBuilder { pass: &mut pass };
        build(&mut builder);
        self.passes.push(pass);
        self
    }

    /// Compile the graph into an executable form.
    ///
    /// Performs topological sort and allocates transient resources. After
    /// this call, the graph is consumed and cannot be modified further.
    pub fn compile(self, ignis: &crate::Ignis) -> Result<CompiledFrameGraph> {
        let shared = ignis.shared_state().clone();
        let allocator = ignis.create_block_allocator();

        // Allocate transient resources.
        let mut realized_images: Vec<RealizedImage> = Vec::with_capacity(self.images.len());
        for decl in &self.images {
            let realized = match &decl.source {
                ImageSource::Transient(desc) => {
                    let info = ImageInfo::texture_2d(
                        desc.width,
                        desc.height,
                        desc.format,
                        desc.usage,
                    );
                    let aspect = crate::format::format_aspect_mask(desc.format);
                    let image = Image::new(shared.clone(), allocator.clone(), &info)?;
                    RealizedImage {
                        handle: image.handle(),
                        owned: Some(image),
                        aspect,
                        mip_levels: 1,
                        array_layers: 1,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                    }
                }
                ImageSource::Imported {
                    handle,
                    initial_layout,
                    mip_levels,
                    array_layers,
                    aspect,
                } => RealizedImage {
                    handle: *handle,
                    owned: None,
                    aspect: *aspect,
                    mip_levels: *mip_levels,
                    array_layers: *array_layers,
                    initial_layout: *initial_layout,
                },
            };
            realized_images.push(realized);
        }

        let mut realized_buffers: Vec<RealizedBuffer> = Vec::with_capacity(self.buffers.len());
        for decl in &self.buffers {
            let realized = match &decl.source {
                BufferSource::Transient(desc) => {
                    let info = BufferInfo {
                        size: desc.size,
                        usage: desc.usage,
                        location: desc.location,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                    };
                    let buffer = Buffer::new(shared.clone(), allocator.clone(), &info)?;
                    RealizedBuffer {
                        handle: buffer.handle(),
                        owned: Some(buffer),
                    }
                }
                BufferSource::Imported(h) => RealizedBuffer {
                    handle: *h,
                    owned: None,
                },
            };
            realized_buffers.push(realized);
        }

        // Topological sort.
        let order = topo_sort(&self.passes)?;

        Ok(CompiledFrameGraph {
            shared,
            _allocator: allocator,
            images: realized_images,
            buffers: realized_buffers,
            image_names: self.images.iter().map(|d| d.name.clone()).collect(),
            buffer_names: self.buffers.iter().map(|d| d.name.clone()).collect(),
            passes: self.passes,
            order,
        })
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

struct RealizedImage {
    handle: vk::Image,
    #[allow(dead_code)]
    owned: Option<Image>,
    aspect: vk::ImageAspectFlags,
    mip_levels: u32,
    array_layers: u32,
    initial_layout: vk::ImageLayout,
}

struct RealizedBuffer {
    handle: vk::Buffer,
    #[allow(dead_code)]
    owned: Option<Buffer>,
}

/// Resolves handles to concrete Vulkan objects during pass execution.
pub struct Resolver<'a> {
    images: &'a [RealizedImage],
    buffers: &'a [RealizedBuffer],
}

impl<'a> Resolver<'a> {
    /// Get the `VkImage` handle for a declared image.
    pub fn image(&self, handle: ImageHandle) -> vk::Image {
        self.images[handle.0 as usize].handle
    }

    /// Get the `VkBuffer` handle for a declared buffer.
    pub fn buffer(&self, handle: BufferHandle) -> vk::Buffer {
        self.buffers[handle.0 as usize].handle
    }
}

/// A compiled, executable frame graph.
pub struct CompiledFrameGraph {
    shared: Arc<SharedState>,
    _allocator: Arc<dyn Allocator>,
    images: Vec<RealizedImage>,
    buffers: Vec<RealizedBuffer>,
    image_names: Vec<String>,
    buffer_names: Vec<String>,
    passes: Vec<PassDecl>,
    order: Vec<usize>,
}

impl CompiledFrameGraph {
    /// Execute the graph on the given queue, recording all passes into
    /// a freshly allocated command buffer from `pool` and submitting it.
    ///
    /// Blocks until GPU work completes (suitable for single-shot or
    /// test usage). For per-frame usage, use
    /// [`record`](Self::record) instead and integrate with your own
    /// FrameSync loop.
    pub fn execute(
        mut self,
        _ignis: &crate::Ignis,
        pool: &crate::CommandPool,
        queue: &crate::AsyncQueue,
    ) -> Result<()> {
        let cmd = pool.allocate_primary()?;
        let rec = pool.begin_primary(cmd)?;
        self.record_into(&rec);
        let cmd = rec.end()?;
        queue.submit_simple(cmd)?.wait()?;
        Ok(())
    }

    /// Record all passes into an already-begun command buffer.
    /// Does not submit or wait.
    pub fn record(mut self, rec: &CommandRecorder<'_>) {
        self.record_into(rec);
    }

    fn record_into(&mut self, rec: &CommandRecorder<'_>) {
        #[cfg(feature = "tracking")]
        {
            self.record_with_tracking(rec);
        }
        #[cfg(not(feature = "tracking"))]
        {
            self.record_without_tracking(rec);
        }
    }

    #[cfg(feature = "tracking")]
    fn record_with_tracking(&mut self, rec: &CommandRecorder<'_>) {
        let mut tracker = ResourceTracker::new();

        // Register every image with its initial state.
        for (i, img) in self.images.iter().enumerate() {
            tracker.track_image(
                img.handle,
                img.initial_layout,
                img.mip_levels,
                img.array_layers,
                img.aspect,
            );
            let _ = i;
        }
        for buf in &self.buffers {
            tracker.track_buffer(buf.handle);
        }

        let order = std::mem::take(&mut self.order);
        let mut passes = std::mem::take(&mut self.passes);

        for pass_idx in order {
            let pass = &passes[pass_idx];

            // Emit barriers for every access declared by this pass.
            let mut image_transitions = Vec::new();
            let mut buffer_transitions = Vec::new();
            for (h, mode) in &pass.access.image {
                let usage = match mode {
                    ImageAccessMode::Read(u) | ImageAccessMode::Write(u) => *u,
                };
                let img_handle = self.images[h.0 as usize].handle;
                if let Some(t) = tracker.transition_image(img_handle, usage) {
                    image_transitions.push(t);
                }
            }
            for (h, mode) in &pass.access.buffer {
                let usage = match mode {
                    BufferAccessMode::Read(u) | BufferAccessMode::Write(u) => *u,
                };
                let buf_handle = self.buffers[h.0 as usize].handle;
                if let Some(t) = tracker.transition_buffer(buf_handle, usage) {
                    buffer_transitions.push(t);
                }
            }
            rec.apply_image_transitions(&image_transitions);
            rec.apply_buffer_transitions(&buffer_transitions);

            // Execute the pass.
            let resolver = Resolver {
                images: &self.images,
                buffers: &self.buffers,
            };
            if let Some(cb) = passes[pass_idx].execute.take_if_available() {
                cb(rec, &resolver);
            }
        }

        let _ = &self.shared;
    }

    #[cfg(not(feature = "tracking"))]
    fn record_without_tracking(&mut self, rec: &CommandRecorder<'_>) {
        // Without the tracking feature, we cannot emit automatic barriers.
        // We still honor topological order so users can rely on it.
        let order = std::mem::take(&mut self.order);
        let mut passes = std::mem::take(&mut self.passes);
        for pass_idx in order {
            let resolver = Resolver {
                images: &self.images,
                buffers: &self.buffers,
            };
            if let Some(cb) = passes[pass_idx].execute.take_if_available() {
                cb(rec, &resolver);
            }
        }
    }

    /// Debug dump of the compiled plan.
    pub fn dump_plan(&self) -> String {
        let mut o = String::with_capacity(1024);
        o.push_str("FrameGraph plan:\n");
        for (i, pass_idx) in self.order.iter().enumerate() {
            let pass = &self.passes[*pass_idx];
            o.push_str(&format!("  {i:>2}. {}\n", pass.name));
            #[cfg(feature = "tracking")]
            {
                for (h, mode) in &pass.access.image {
                    let name = &self.image_names[h.0 as usize];
                    let tag = match mode {
                        ImageAccessMode::Read(_) => "R",
                        ImageAccessMode::Write(_) => "W",
                    };
                    o.push_str(&format!("       [{tag}] image  {name}\n"));
                }
                for (h, mode) in &pass.access.buffer {
                    let name = &self.buffer_names[h.0 as usize];
                    let tag = match mode {
                        BufferAccessMode::Read(_) => "R",
                        BufferAccessMode::Write(_) => "W",
                    };
                    o.push_str(&format!("       [{tag}] buffer {name}\n"));
                }
            }
        }
        let _ = &self.image_names;
        let _ = &self.buffer_names;
        o
    }
}

// Small helper trait used above: PassDecl.execute is an Option but we
// cannot move out of it through an index in the common case without
// take(). This helper keeps the call sites clean.
trait TakeIf {
    fn take_if_available(&mut self) -> Option<PassExecute>;
}

impl TakeIf for Option<PassExecute> {
    fn take_if_available(&mut self) -> Option<PassExecute> {
        self.take()
    }
}

/// Kahn's algorithm. Produces a topological ordering of pass indices.
///
/// Edge rule: each resource's readers depend on every writer of that
/// resource. When multiple passes write the same resource, those writers
/// are serialized in registration order. Self-references (a pass that
/// both reads and writes the same resource) are ignored in the reader
/// step to avoid spurious self-loops.
fn topo_sort(passes: &[PassDecl]) -> Result<Vec<usize>> {
    let n = passes.len();
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indegree = vec![0u32; n];

    #[cfg(feature = "tracking")]
    {
        // Pass 1: collect every writer of every resource.
        let mut image_writers: HashMap<ImageHandle, Vec<usize>> = HashMap::new();
        let mut buffer_writers: HashMap<BufferHandle, Vec<usize>> = HashMap::new();

        for (i, pass) in passes.iter().enumerate() {
            for (h, mode) in &pass.access.image {
                if matches!(mode, ImageAccessMode::Write(_)) {
                    image_writers.entry(*h).or_default().push(i);
                }
            }
            for (h, mode) in &pass.access.buffer {
                if matches!(mode, BufferAccessMode::Write(_)) {
                    buffer_writers.entry(*h).or_default().push(i);
                }
            }
        }

        let mut added: HashSet<(usize, usize)> = HashSet::new();

        // Pass 2a: serialize writers-of-the-same-resource in the order
        // they were registered. This keeps write-after-write deterministic
        // when multiple passes stomp on the same resource.
        for writers in image_writers.values() {
            for w in writers.windows(2) {
                let (from, to) = (w[0], w[1]);
                if added.insert((from, to)) {
                    edges[from].push(to);
                    indegree[to] += 1;
                }
            }
        }
        for writers in buffer_writers.values() {
            for w in writers.windows(2) {
                let (from, to) = (w[0], w[1]);
                if added.insert((from, to)) {
                    edges[from].push(to);
                    indegree[to] += 1;
                }
            }
        }

        // Pass 2b: every reader depends on every writer of the resource.
        // A pass that both reads and writes the same resource is skipped
        // in the reader step to avoid creating a self-loop; the write is
        // already covered by write-write serialization above.
        for (i, pass) in passes.iter().enumerate() {
            for (h, mode) in &pass.access.image {
                if matches!(mode, ImageAccessMode::Read(_)) {
                    if let Some(writers) = image_writers.get(h) {
                        for &w in writers {
                            if w != i && added.insert((w, i)) {
                                edges[w].push(i);
                                indegree[i] += 1;
                            }
                        }
                    }
                }
            }
            for (h, mode) in &pass.access.buffer {
                if matches!(mode, BufferAccessMode::Read(_)) {
                    if let Some(writers) = buffer_writers.get(h) {
                        for &w in writers {
                            if w != i && added.insert((w, i)) {
                                edges[w].push(i);
                                indegree[i] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Kahn's: process zero-indegree nodes in declaration order.
    let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut out = Vec::with_capacity(n);
    while let Some(node) = ready.first().copied() {
        ready.remove(0);
        out.push(node);
        let successors = std::mem::take(&mut edges[node]);
        for s in successors {
            indegree[s] -= 1;
            if indegree[s] == 0 {
                ready.push(s);
            }
        }
    }

    if out.len() != n {
        return Err(Error::InvalidConfig(
            "framegraph: dependency cycle detected",
        ));
    }
    Ok(out)
}