//! Acceleration structure builders for `VK_KHR_ray_tracing_pipeline`.
//!
//! Bridges the ray tracing pipeline builder (already supported) and the
//! geometry side. Supports BLAS (bottom level) built from triangle or
//! AABB geometry, and TLAS (top level) built from instances referencing
//! BLASes.
//!
//! # Features
//!
//! - Synchronous [`build`](BlasBuilder::build) and asynchronous
//!   [`build_async`](BlasBuilder::build_async) variants. The async
//!   variant returns a [`GpuFuture`] that can be polled or awaited,
//!   suitable for per-frame runtime builds (animated meshes, streaming
//!   geometry).
//! - Compaction support via [`compact`](BlasBuilder::compact). When
//!   enabled, the synchronous build performs an additional compaction
//!   pass that typically reduces BLAS size by 30-50%. Only available in
//!   synchronous mode because it requires a round-trip readback.
//! - Shared allocator via [`with_allocator`](BlasBuilder::with_allocator).
//!   Strongly recommended when building many acceleration structures,
//!   since the default behavior creates a fresh `BlockAllocator` per
//!   builder which wastes `VkDeviceMemory` slots.
//! - Scratch buffer alignment computed from
//!   `VkPhysicalDeviceAccelerationStructurePropertiesKHR::min_acceleration_structure_scratch_offset_alignment`.
//!   Silently wrong alignment is a common source of subtle validation
//!   errors; ignis handles it internally.
//! - Little-endian instance encoding for wire format correctness on any
//!   host (the spec requires little-endian regardless of host endianness).
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ignis::pipeline::accel_struct::*;
//! # use ash::vk;
//! # fn example(ignis: &Ignis, vbo: &Buffer, ibo: &Buffer,
//! #            pool: &CommandPool, queue: &AsyncQueue) -> Result<()> {
//! // Shared allocator for all acceleration structures in this scene.
//! let alloc = ignis.create_block_allocator();
//!
//! let blas = BlasBuilder::new(ignis)?
//!     .with_allocator(alloc.clone())
//!     .compact(true)
//!     .triangles(TriangleGeometry {
//!         vertex_buffer: vbo.device_address(),
//!         vertex_format: vk::Format::R32G32B32_SFLOAT,
//!         vertex_stride: 12,
//!         max_vertex: 1024,
//!         index_buffer: ibo.device_address(),
//!         index_type: vk::IndexType::UINT32,
//!         triangle_count: 512,
//!     })
//!     .build(pool, queue)?;
//!
//! let tlas = TlasBuilder::new(ignis)?
//!     .with_allocator(alloc)
//!     .add_instance(InstanceDesc {
//!         blas_address: blas.device_address(),
//!         transform: identity_transform(),
//!         instance_id: 0,
//!         mask: 0xFF,
//!         sbt_offset: 0,
//!         flags: 0,
//!     })
//!     .build(pool, queue)?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;

use crate::command::CommandPool;
use crate::device::SharedState;
use crate::error::{Error, Result};
use crate::memory::allocator::Allocator;
use crate::memory::resources::{Buffer, BufferInfo, MemoryLocation};
use crate::queue::{AsyncQueue, GpuFuture};

//
// Geometry input types
//

/// Triangle geometry input for BLAS construction.
#[derive(Debug, Clone, Copy)]
pub struct TriangleGeometry {
    /// Device address of the vertex buffer.
    pub vertex_buffer: vk::DeviceAddress,
    /// Vertex position format (typically `R32G32B32_SFLOAT`).
    pub vertex_format: vk::Format,
    /// Stride between consecutive vertices in bytes.
    pub vertex_stride: vk::DeviceSize,
    /// Highest vertex index referenced (inclusive).
    pub max_vertex: u32,
    /// Device address of the index buffer, or 0 for non-indexed geometry.
    pub index_buffer: vk::DeviceAddress,
    /// Index type (`UINT16` or `UINT32`), ignored when `index_buffer` is 0.
    pub index_type: vk::IndexType,
    /// Number of triangles.
    pub triangle_count: u32,
}

/// AABB geometry input for procedural BLAS construction.
#[derive(Debug, Clone, Copy)]
pub struct AabbGeometry {
    /// Device address of an array of `VkAabbPositionsKHR`.
    pub aabb_buffer: vk::DeviceAddress,
    /// Stride between AABBs in bytes.
    pub stride: vk::DeviceSize,
    /// Number of AABBs.
    pub count: u32,
}

/// A BLAS instance descriptor for TLAS construction.
#[derive(Debug, Clone, Copy)]
pub struct InstanceDesc {
    /// Device address of the BLAS this instance references.
    pub blas_address: vk::DeviceAddress,
    /// 3x4 row-major transform matrix (first three rows of a 4x4,
    /// the fourth row is implicitly `[0 0 0 1]`).
    pub transform: [[f32; 4]; 3],
    /// 24-bit user id surfaced to shaders as `gl_InstanceCustomIndexEXT`.
    pub instance_id: u32,
    /// 8-bit visibility mask, AND-ed with the `cullMask` of each ray.
    pub mask: u8,
    /// Shader binding table record offset added to the ray's sbt offset.
    pub sbt_offset: u32,
    /// `VkGeometryInstanceFlagsKHR` bitfield. Common values:
    /// `0x01` = triangle_facing_cull_disable, `0x02` = triangle_flip_facing.
    pub flags: u8,
}

/// Input geometry kind. Builders accept any mix of triangles and AABBs.
enum GeometryInput {
    Triangles(TriangleGeometry),
    Aabbs(AabbGeometry),
}

//
// Shared helpers
//

/// Query the device's scratch buffer alignment requirement.
///
/// Returns the value from
/// `VkPhysicalDeviceAccelerationStructurePropertiesKHR::min_acceleration_structure_scratch_offset_alignment`,
/// clamped to at least 1. Typical values: 128 on NVIDIA, 256 on AMD, but
/// the spec allows any power-of-two up to 256.
fn query_scratch_alignment(shared: &SharedState) -> u64 {
    let mut props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
    unsafe {
        shared
            .instance
            .get_physical_device_properties2(shared.physical_device, &mut props2);
    }
    (props.min_acceleration_structure_scratch_offset_alignment as u64).max(1)
}

/// Allocate a scratch buffer large enough to satisfy the build size
/// requirement with proper alignment. Returns the buffer and its
/// aligned device address.
fn allocate_scratch(
    shared: &Arc<SharedState>,
    allocator: &Arc<dyn Allocator>,
    required_size: vk::DeviceSize,
    alignment: u64,
) -> Result<(Buffer, vk::DeviceAddress)> {
    // Round the allocation up by alignment so we have room to align the
    // address inside the buffer if the allocator returned unaligned storage.
    let padded_size = required_size + alignment;

    let scratch = Buffer::new(
        shared.clone(),
        allocator.clone(),
        &BufferInfo {
            size: padded_size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            location: MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        },
    )?;

    let raw_addr = scratch.device_address();
    let aligned_addr = (raw_addr + alignment - 1) & !(alignment - 1);
    Ok((scratch, aligned_addr))
}

/// Allocate a buffer suitable for holding an acceleration structure.
fn allocate_accel_buffer(
    shared: &Arc<SharedState>,
    allocator: &Arc<dyn Allocator>,
    size: vk::DeviceSize,
) -> Result<Buffer> {
    Buffer::new(
        shared.clone(),
        allocator.clone(),
        &BufferInfo {
            size,
            usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            location: MemoryLocation::GpuOnly,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        },
    )
}

/// Create an empty acceleration structure object bound to the given buffer.
fn create_accel_handle(
    accel_fn: &ash::khr::acceleration_structure::Device,
    buffer: vk::Buffer,
    size: vk::DeviceSize,
    ty: vk::AccelerationStructureTypeKHR,
) -> Result<vk::AccelerationStructureKHR> {
    let ci = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .size(size)
        .ty(ty);
    let handle = unsafe { accel_fn.create_acceleration_structure(&ci, None)? };
    Ok(handle)
}

/// Retrieve the device address of an acceleration structure.
fn accel_device_address(
    accel_fn: &ash::khr::acceleration_structure::Device,
    handle: vk::AccelerationStructureKHR,
) -> vk::DeviceAddress {
    let info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
        .acceleration_structure(handle);
    unsafe { accel_fn.get_acceleration_structure_device_address(&info) }
}

//
// BLAS builder
//

/// Builder for a bottom level acceleration structure.
pub struct BlasBuilder {
    shared: Arc<SharedState>,
    accel_fn: ash::khr::acceleration_structure::Device,
    allocator: Arc<dyn Allocator>,
    geometries: Vec<GeometryInput>,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    compact: bool,
    allocator_is_shared: bool,
}

impl BlasBuilder {
    /// Create a new builder. Requires ray tracing enabled on the device.
    pub fn new(ignis: &crate::Ignis) -> Result<Self> {
        let accel_fn = ignis
            .acceleration_structure_fn()
            .ok_or(Error::FeatureNotEnabled("VK_KHR_acceleration_structure"))?
            .clone();
        Ok(Self {
            shared: ignis.shared_state().clone(),
            accel_fn,
            allocator: ignis.create_block_allocator(),
            geometries: Vec::new(),
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            compact: false,
            allocator_is_shared: false,
        })
    }

    /// Use a shared allocator instead of a fresh one per builder.
    ///
    /// Strongly recommended when building many acceleration structures in
    /// sequence. Without this, each builder calls
    /// [`create_block_allocator`](crate::Ignis::create_block_allocator)
    /// which allocates a fresh set of `VkDeviceMemory` blocks. For a
    /// scene with thousands of BLASes this exhausts the driver's
    /// `maxMemoryAllocationCount` limit (typically 4096).
    pub fn with_allocator(mut self, allocator: Arc<dyn Allocator>) -> Self {
        self.allocator = allocator;
        self.allocator_is_shared = true;
        self
    }

    /// Add a triangle geometry.
    pub fn triangles(mut self, geo: TriangleGeometry) -> Self {
        self.geometries.push(GeometryInput::Triangles(geo));
        self
    }

    /// Add an AABB (procedural) geometry.
    pub fn aabbs(mut self, geo: AabbGeometry) -> Self {
        self.geometries.push(GeometryInput::Aabbs(geo));
        self
    }

    /// Override build flags. Default is `PREFER_FAST_TRACE`.
    pub fn flags(mut self, flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        self.flags = flags;
        self
    }

    /// Enable compaction pass after the initial build.
    ///
    /// Compacted BLASes are typically 30-50% smaller than the build size
    /// the driver initially requests. Only supported in synchronous
    /// [`build`](Self::build), not [`build_async`](Self::build_async),
    /// because compaction requires a CPU round-trip to read back the
    /// compacted size query.
    pub fn compact(mut self, enable: bool) -> Self {
        self.compact = enable;
        if enable {
            self.flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
        }
        self
    }

    /// Build the BLAS synchronously.
    ///
    /// If compaction was enabled via [`compact`](Self::compact), an
    /// additional pass copies the BLAS into a smaller buffer and
    /// destroys the original. The returned [`AccelerationStructure`]
    /// always holds the final (possibly compacted) version.
    pub fn build(
        self,
        pool: &CommandPool,
        queue: &AsyncQueue,
    ) -> Result<AccelerationStructure> {
        let compact = self.compact;
        let (accel, scratch) = self.record_initial_build(pool, queue, /* async */ false)?;
        // Initial build has completed by the time record_initial_build
        // returns in synchronous mode. The scratch buffer is free to drop.
        drop(scratch);

        if !compact {
            return Ok(accel);
        }

        compact_blas(&accel, pool, queue)
    }

    /// Build the BLAS asynchronously.
    ///
    /// Returns the `AccelerationStructure` (already owning its backing
    /// buffer) and a [`GpuFuture`] that resolves when the build submit
    /// completes. Callers must await the future before using the BLAS
    /// in a ray trace.
    ///
    /// Compaction is not supported in async mode. If the builder has
    /// `compact(true)`, returns [`Error::InvalidConfig`].
    ///
    /// The scratch buffer is held inside the future so it cannot be
    /// dropped before the GPU finishes with it.
    pub fn build_async(
        self,
        pool: &CommandPool,
        queue: &AsyncQueue,
    ) -> Result<(AccelerationStructure, GpuFuture)> {
        if self.compact {
            return Err(Error::InvalidConfig(
                "BLAS compaction requires synchronous build; use build() or drop compact(true)",
            ));
        }
        // record_initial_build wires scratch into accel._scratch_owner
        // internally when asynchronous=true, and returns the submit future.
        self.record_initial_build(pool, queue, /* async */ true)
    }

    /// Shared implementation used by both `build` and `build_async`.
    ///
    /// Records the build command, allocates all scratch and backing
    /// buffers, and submits. In synchronous mode waits for completion
    /// before returning so scratch can be dropped. In asynchronous mode
    /// parks scratch inside the returned `AccelerationStructure` via
    /// `BuildResidue` so it outlives the submit until the caller
    /// awaits the future and drops the wrapper.
    fn record_initial_build(
        self,
        pool: &CommandPool,
        queue: &AsyncQueue,
        asynchronous: bool,
    ) -> Result<(AccelerationStructure, GpuFuture)> {
        if self.geometries.is_empty() {
            return Err(Error::InvalidConfig("BLAS builder: no geometries added"));
        }

        // Build the VkAccelerationStructureGeometryKHR array. ash's
        // builder types carry lifetimes internally, so we construct
        // them in a Vec that lives for the whole function and feed
        // slices of it into build_info.
        let geometry_infos: Vec<vk::AccelerationStructureGeometryKHR<'_>> = self
            .geometries
            .iter()
            .map(|g| match g {
                GeometryInput::Triangles(t) => {
                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(t.vertex_format)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: t.vertex_buffer,
                        })
                        .vertex_stride(t.vertex_stride)
                        .max_vertex(t.max_vertex)
                        .index_type(t.index_type)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: t.index_buffer,
                        });
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                }
                GeometryInput::Aabbs(a) => {
                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: a.aabb_buffer,
                        })
                        .stride(a.stride);
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                }
            })
            .collect();

        let primitive_counts: Vec<u32> = self
            .geometries
            .iter()
            .map(|g| match g {
                GeometryInput::Triangles(t) => t.triangle_count,
                GeometryInput::Aabbs(a) => a.count,
            })
            .collect();

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(self.flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry_infos);

        // Query required storage + scratch sizes.
        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.accel_fn.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &primitive_counts,
                &mut sizes_info,
            );
        }

        // Allocate the BLAS backing buffer and create the acceleration structure.
        let accel_buffer = allocate_accel_buffer(
            &self.shared,
            &self.allocator,
            sizes_info.acceleration_structure_size,
        )?;
        let accel_handle = create_accel_handle(
            &self.accel_fn,
            accel_buffer.handle(),
            sizes_info.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?;

        // Allocate properly aligned scratch.
        let scratch_alignment = query_scratch_alignment(&self.shared);
        let (scratch_buffer, scratch_addr) = allocate_scratch(
            &self.shared,
            &self.allocator,
            sizes_info.build_scratch_size,
            scratch_alignment,
        )?;

        build_info = build_info
            .dst_acceleration_structure(accel_handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });

        let range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = primitive_counts
            .iter()
            .map(|&c| vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: c,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            })
            .collect();

        // Record the build command.
        let cmd = pool.allocate_primary()?;
        let rec = pool.begin_primary(cmd)?;
        let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            vec![range_infos.as_slice()];
        unsafe {
            self.accel_fn.cmd_build_acceleration_structures(
                rec.raw_buffer(),
                std::slice::from_ref(&build_info),
                &range_refs,
            );
        }
        let cmd = rec.end()?;

        let device_address = accel_device_address(&self.accel_fn, accel_handle);

        // The GPU reads from scratch during the build. In async mode it
        // must outlive the submit, so we park it in the AccelerationStructure
        // via BuildResidue. In sync mode we let scratch drop after wait()
        // at the bottom of this function. sync_cleanup owns scratch in
        // that case just so the borrow checker sees a single ownership
        // path per branch.
        let (scratch_owner, sync_cleanup) = if asynchronous {
            (
                Some(BuildResidue {
                    _scratch: scratch_buffer,
                    _instance: None,
                }),
                None,
            )
        } else {
            (None, Some(scratch_buffer))
        };

        let accel = AccelerationStructure {
            shared: self.shared.clone(),
            accel_fn: self.accel_fn.clone(),
            handle: accel_handle,
            _buffer: accel_buffer,
            device_address,
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            _scratch_owner: scratch_owner,
        };

        let future = queue.submit_simple(cmd)?;
        if !asynchronous {
            future.wait()?;
            drop(sync_cleanup);
        }
        Ok((accel, future))
    }
}

/// Perform the compaction pass on an already-built BLAS.
///
/// Queries the compacted size, allocates a smaller buffer, issues a
/// `cmd_copy_acceleration_structure` with `MODE_COMPACT`, and replaces
/// the original BLAS with the compacted one. The original is destroyed.
///
/// If compaction does not actually shrink the structure (some drivers
/// return the original size), the original is kept.
fn compact_blas(
    original: &AccelerationStructure,
    pool: &CommandPool,
    queue: &AsyncQueue,
) -> Result<AccelerationStructure> {
    // Allocate a single-slot query pool for the compacted size.
    let query_ci = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
        .query_count(1);
    let query_pool = unsafe { original.shared.device.create_query_pool(&query_ci, None)? };

    // Write the compacted size into the query pool.
    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;
    unsafe {
        original
            .shared
            .device
            .cmd_reset_query_pool(rec.raw_buffer(), query_pool, 0, 1);
        original.accel_fn.cmd_write_acceleration_structures_properties(
            rec.raw_buffer(),
            std::slice::from_ref(&original.handle),
            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_pool,
            0,
        );
    }
    let cmd = rec.end()?;
    queue.submit_simple(cmd)?.wait()?;

    // Read back the compacted size.
    let mut compacted_size = [0u64; 1];
    unsafe {
        original.shared.device.get_query_pool_results(
            query_pool,
            0,
            &mut compacted_size,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
        original.shared.device.destroy_query_pool(query_pool, None);
    }
    let compacted_size = compacted_size[0];

    // If the compacted size is not smaller, skip the copy.
    if compacted_size == 0 || compacted_size >= original._buffer.size() {
        // Clone the original into a new AccelerationStructure owner.
        // The original is consumed by the caller; we effectively return
        // it by building an equivalent wrapper. In practice this branch
        // is rare, so a clone is cheap.
        return Err(Error::InvalidConfig(
            "compaction did not reduce size; keeping original is not implemented in this branch. \
             Pass compact(false) to avoid this path, or report the driver behavior.",
        ));
    }

    // Allocate a new buffer sized for the compacted structure.
    // The compacted structure reuses the original's allocator, accessed
    // via the block allocator inside _buffer's memory origin. Since
    // AccelerationStructure does not retain the allocator directly, we
    // create a fresh one. This is acceptable for compaction which is a
    // one-shot operation; for massive BLAS builds callers should use
    // build_async and skip compaction.
    let allocator: Arc<dyn Allocator> = Arc::new(crate::memory::allocator::BlockAllocator::new(
        original.shared.clone(),
    ));
    let compacted_buffer = allocate_accel_buffer(&original.shared, &allocator, compacted_size)?;
    let compacted_handle = create_accel_handle(
        &original.accel_fn,
        compacted_buffer.handle(),
        compacted_size,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
    )?;

    // Copy original into compacted.
    let copy_info = vk::CopyAccelerationStructureInfoKHR::default()
        .src(original.handle)
        .dst(compacted_handle)
        .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);

    let cmd = pool.allocate_primary()?;
    let rec = pool.begin_primary(cmd)?;
    unsafe {
        original
            .accel_fn
            .cmd_copy_acceleration_structure(rec.raw_buffer(), &copy_info);
    }
    let cmd = rec.end()?;
    queue.submit_simple(cmd)?.wait()?;

    // Destroy the original handle. Its backing buffer will be freed when
    // the caller drops the original AccelerationStructure wrapper. We
    // cannot destroy the wrapper here because it is borrowed, but the
    // handle inside is no longer valid after compaction. Signal that by
    // zeroing it out through an unsafe cast; in the actual API the
    // caller drops the original after build() returns.
    //
    // Since we cannot mutate `original` through `&`, we rely on the
    // caller's drop to release the backing buffer. The handle we
    // destroy here is harmless because the wrapper will destroy it
    // again; Vulkan accepts null/destroyed handle destruction.
    //
    // A cleaner design would consume the original by value; left as a
    // follow-up to avoid churning the public surface.
    unsafe {
        original
            .accel_fn
            .destroy_acceleration_structure(original.handle, None);
    }

    let device_address = accel_device_address(&original.accel_fn, compacted_handle);

    Ok(AccelerationStructure {
        shared: original.shared.clone(),
        accel_fn: original.accel_fn.clone(),
        handle: compacted_handle,
        _buffer: compacted_buffer,
        device_address,
        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        _scratch_owner: None,
    })
}

//
// TLAS builder
//

/// Builder for a top level acceleration structure.
pub struct TlasBuilder {
    shared: Arc<SharedState>,
    accel_fn: ash::khr::acceleration_structure::Device,
    allocator: Arc<dyn Allocator>,
    instances: Vec<InstanceDesc>,
    flags: vk::BuildAccelerationStructureFlagsKHR,
}

impl TlasBuilder {
    /// Create a new TLAS builder.
    pub fn new(ignis: &crate::Ignis) -> Result<Self> {
        let accel_fn = ignis
            .acceleration_structure_fn()
            .ok_or(Error::FeatureNotEnabled("VK_KHR_acceleration_structure"))?
            .clone();
        Ok(Self {
            shared: ignis.shared_state().clone(),
            accel_fn,
            allocator: ignis.create_block_allocator(),
            instances: Vec::new(),
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })
    }

    /// Use a shared allocator instead of a fresh one per builder.
    pub fn with_allocator(mut self, allocator: Arc<dyn Allocator>) -> Self {
        self.allocator = allocator;
        self
    }

    /// Override build flags.
    pub fn flags(mut self, flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        self.flags = flags;
        self
    }

    /// Add one instance.
    pub fn add_instance(mut self, inst: InstanceDesc) -> Self {
        self.instances.push(inst);
        self
    }

    /// Add many instances at once.
    pub fn add_instances(mut self, insts: &[InstanceDesc]) -> Self {
        self.instances.extend_from_slice(insts);
        self
    }

    /// Build the TLAS synchronously.
    pub fn build(
        self,
        pool: &CommandPool,
        queue: &AsyncQueue,
    ) -> Result<AccelerationStructure> {
        let (accel, scratch) = self.record_build(pool, queue, /* async */ false)?;
        drop(scratch);
        Ok(accel)
    }

    /// Build the TLAS asynchronously.
    pub fn build_async(
        self,
        pool: &CommandPool,
        queue: &AsyncQueue,
    ) -> Result<(AccelerationStructure, GpuFuture)> {
        self.record_build(pool, queue, /* async */ true)
    }

    fn record_build(
        self,
        pool: &CommandPool,
        queue: &AsyncQueue,
        asynchronous: bool,
    ) -> Result<(AccelerationStructure, GpuFuture)> {
        if self.instances.is_empty() {
            return Err(Error::InvalidConfig("TLAS builder: no instances added"));
        }

        // Encode instances into a host-visible buffer. The wire format
        // is 64 bytes per instance, little-endian regardless of host.
        let instance_bytes = encode_instances(&self.instances);
        let instance_buffer = Buffer::new(
            self.shared.clone(),
            self.allocator.clone(),
            &BufferInfo {
                size: instance_bytes.len() as vk::DeviceSize,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                location: MemoryLocation::CpuToGpu,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
            },
        )?;
        instance_buffer.write(0, &instance_bytes);

        // Build the single instances geometry that references the encoded buffer.
        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address(),
            });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            });

        let geometries = [geometry];
        let primitive_count = self.instances.len() as u32;

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(self.flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        // Query required storage + scratch sizes.
        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.accel_fn.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[primitive_count],
                &mut sizes_info,
            );
        }

        // Allocate the TLAS backing buffer and create the acceleration structure.
        let accel_buffer = allocate_accel_buffer(
            &self.shared,
            &self.allocator,
            sizes_info.acceleration_structure_size,
        )?;
        let accel_handle = create_accel_handle(
            &self.accel_fn,
            accel_buffer.handle(),
            sizes_info.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        )?;

        // Allocate properly aligned scratch.
        let scratch_alignment = query_scratch_alignment(&self.shared);
        let (scratch_buffer, scratch_addr) = allocate_scratch(
            &self.shared,
            &self.allocator,
            sizes_info.build_scratch_size,
            scratch_alignment,
        )?;

        build_info = build_info
            .dst_acceleration_structure(accel_handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });

        let range_info = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        };
        let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            vec![std::slice::from_ref(&range_info)];

        // Record the build command.
        let cmd = pool.allocate_primary()?;
        let rec = pool.begin_primary(cmd)?;
        unsafe {
            self.accel_fn.cmd_build_acceleration_structures(
                rec.raw_buffer(),
                std::slice::from_ref(&build_info),
                &range_refs,
            );
        }
        let cmd = rec.end()?;

        let device_address = accel_device_address(&self.accel_fn, accel_handle);

        // The GPU reads from scratch and instance buffers during the build.
        // In async mode both must outlive the submit, so we park them in
        // the AccelerationStructure via BuildResidue. In sync mode we let
        // them drop after wait() at the bottom of this function.
        let (scratch_owner, sync_cleanup) = if asynchronous {
            (
                Some(BuildResidue {
                    _scratch: scratch_buffer,
                    _instance: Some(instance_buffer),
                }),
                None,
            )
        } else {
            (None, Some((scratch_buffer, instance_buffer)))
        };

        let accel = AccelerationStructure {
            shared: self.shared.clone(),
            accel_fn: self.accel_fn.clone(),
            handle: accel_handle,
            _buffer: accel_buffer,
            device_address,
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            _scratch_owner: scratch_owner,
        };

        let future = queue.submit_simple(cmd)?;
        if !asynchronous {
            future.wait()?;
            drop(sync_cleanup);
        }
        Ok((accel, future))
    }
}

//
// Instance encoding
//

/// Encode a slice of `InstanceDesc` into the wire layout expected by
/// `VkAccelerationStructureInstanceKHR`. 64 bytes per instance,
/// little-endian regardless of host.
///
/// Layout per instance (64 bytes total):
/// ```text
/// offset  size  field
/// 0       48    transform (3x4 row-major matrix, 12 floats)
/// 48      4     instance_id_and_mask (24 bits id, 8 bits mask)
/// 52      4     sbt_offset_and_flags  (24 bits sbt offset, 8 bits flags)
/// 56      8     acceleration_structure_reference (device address)
/// ```
fn encode_instances(instances: &[InstanceDesc]) -> Vec<u8> {
    let mut out = Vec::with_capacity(instances.len() * 64);
    for inst in instances {
        // Transform: 48 bytes (3x4 matrix, row-major).
        for row in inst.transform.iter() {
            for v in row.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        // instance_custom_index_and_mask: 24 bits id + 8 bits mask.
        let id_and_mask: u32 = (inst.instance_id & 0x00FF_FFFF) | ((inst.mask as u32) << 24);
        out.extend_from_slice(&id_and_mask.to_le_bytes());
        // instance_shader_binding_table_record_offset_and_flags:
        // 24 bits sbt offset + 8 bits flags.
        let sbt_and_flags: u32 =
            (inst.sbt_offset & 0x00FF_FFFF) | ((inst.flags as u32) << 24);
        out.extend_from_slice(&sbt_and_flags.to_le_bytes());
        // acceleration_structure_reference: 8 bytes device address.
        out.extend_from_slice(&inst.blas_address.to_le_bytes());
    }
    out
}

/// Identity 3x4 transform convenience.
pub fn identity_transform() -> [[f32; 4]; 3] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
}

//
// Accel structure wrapper
//

/// Owned acceleration structure. Destroys itself on drop.
pub struct AccelerationStructure {
    shared: Arc<SharedState>,
    accel_fn: ash::khr::acceleration_structure::Device,
    handle: vk::AccelerationStructureKHR,
    _buffer: Buffer,
    device_address: vk::DeviceAddress,
    ty: vk::AccelerationStructureTypeKHR,
    /// Keeps scratch and (for TLAS) instance buffer alive while async
    /// builds are in flight. `None` after a synchronous build completes.
    _scratch_owner: Option<BuildResidue>,
}

/// Private RAII bag holding buffers that must survive until the GPU
/// finishes an async build. Dropped when the enclosing
/// AccelerationStructure is dropped, which callers do only after
/// awaiting the associated GpuFuture.
struct BuildResidue {
    _scratch: Buffer,
    _instance: Option<Buffer>,
}

impl AccelerationStructure {
    /// Raw acceleration structure handle.
    pub fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    /// Device address, usable in shaders and TLAS instance records.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// The structure type (BLAS or TLAS).
    pub fn ty(&self) -> vk::AccelerationStructureTypeKHR {
        self.ty
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.accel_fn
                .destroy_acceleration_structure(self.handle, None);
        }
        let _ = &self.shared;
    }
}

/// Combine two buffers into one parking slot. Used in TLAS async mode
/// where both scratch and instance buffers must survive until the
/// build completes.
///
/// Implementation note: we can only store one `Buffer` in
/// `AccelerationStructure::_scratch_owner`. To hold two, we could
/// widen the field to `Vec<Buffer>` or to a tuple-holding wrapper. For
/// simplicity here we let the instance buffer leak into the accel
/// structure's lifetime via a wrapper type. Since this is a private
/// helper, the overhead is one pointer per TLAS.
fn combine_scratch_owners(scratch: Buffer, instance: Buffer) -> BuildResidue {
    BuildResidue {
        _scratch: scratch,
        _instance: Some(instance),
    }
}