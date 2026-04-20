//! Pipeline statistics queries via `VK_QUERY_TYPE_PIPELINE_STATISTICS`.
//!
//! Answers "why is this pass slow" questions that the GPU timer cannot:
//! how many vertices were shaded, how many primitives were clipped, how
//! many fragment invocations happened (before vs after a mip-skip
//! optimization), how many compute invocations fired.

use std::sync::Arc;

use ash::vk;

use crate::command::CommandRecorder;
use crate::device::SharedState;
use crate::error::Result;

/// Which statistics counters to enable on a pool.
///
/// Maps directly to `VkQueryPipelineStatisticFlags`. Passed as a u32
/// bitfield because we do not want to introduce a bitflags dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStats(pub u32);

impl PipelineStats {
    /// Input assembly vertices read.
    pub const INPUT_ASSEMBLY_VERTICES: Self = Self(0x0000_0001);
    /// Input assembly primitives produced.
    pub const INPUT_ASSEMBLY_PRIMITIVES: Self = Self(0x0000_0002);
    /// Vertex shader invocations.
    pub const VERTEX_INVOCATIONS: Self = Self(0x0000_0004);
    /// Geometry shader invocations.
    pub const GEOMETRY_INVOCATIONS: Self = Self(0x0000_0008);
    /// Geometry shader primitives produced.
    pub const GEOMETRY_PRIMITIVES: Self = Self(0x0000_0010);
    /// Primitives reaching the clipping stage.
    pub const CLIPPING_INVOCATIONS: Self = Self(0x0000_0020);
    /// Primitives output by clipping.
    pub const CLIPPING_PRIMITIVES: Self = Self(0x0000_0040);
    /// Fragment shader invocations.
    pub const FRAGMENT_INVOCATIONS: Self = Self(0x0000_0080);
    /// Tessellation control shader patches processed.
    pub const TESS_CONTROL_PATCHES: Self = Self(0x0000_0100);
    /// Tessellation evaluation shader invocations.
    pub const TESS_EVALUATION_INVOCATIONS: Self = Self(0x0000_0200);
    /// Compute shader invocations.
    pub const COMPUTE_INVOCATIONS: Self = Self(0x0000_0400);

    /// Union with another set of flags.
    pub const fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Count how many bits are set; determines result slot count.
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterate through the individual enabled bits, returning tuples of
    /// `(flag, name)` for the enabled counters in canonical order.
    fn enabled_iter(self) -> impl Iterator<Item = (Self, &'static str)> {
        const ALL: &[(PipelineStats, &str)] = &[
            (PipelineStats::INPUT_ASSEMBLY_VERTICES, "ia_vertices"),
            (PipelineStats::INPUT_ASSEMBLY_PRIMITIVES, "ia_primitives"),
            (PipelineStats::VERTEX_INVOCATIONS, "vs_invocations"),
            (PipelineStats::GEOMETRY_INVOCATIONS, "gs_invocations"),
            (PipelineStats::GEOMETRY_PRIMITIVES, "gs_primitives"),
            (PipelineStats::CLIPPING_INVOCATIONS, "clip_invocations"),
            (PipelineStats::CLIPPING_PRIMITIVES, "clip_primitives"),
            (PipelineStats::FRAGMENT_INVOCATIONS, "fs_invocations"),
            (PipelineStats::TESS_CONTROL_PATCHES, "tcs_patches"),
            (
                PipelineStats::TESS_EVALUATION_INVOCATIONS,
                "tes_invocations",
            ),
            (PipelineStats::COMPUTE_INVOCATIONS, "cs_invocations"),
        ];
        let bits = self.0;
        ALL.iter().copied().filter(move |(f, _)| (bits & f.0) != 0)
    }
}

impl std::ops::BitOr for PipelineStats {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.or(rhs)
    }
}

/// Handle returned by `begin` and passed to `end`.
#[derive(Debug, Clone, Copy)]
pub struct PipelineStatsScope {
    query_index: u32,
    //label_index: usize,
}

/// Results of one scope after readback.
#[derive(Debug, Clone)]
pub struct PipelineStatsResult {
    /// Human-readable label supplied at `begin`.
    pub label: String,
    /// Named counters; only the bits enabled on the pool are present.
    pub counters: Vec<(&'static str, u64)>,
}

impl PipelineStatsResult {
    /// Get a counter by its short name, if present.
    pub fn get(&self, name: &str) -> Option<u64> {
        self.counters
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
    }

    /// Convenience accessor for fragment invocations.
    pub fn fragment_invocations(&self) -> Option<u64> {
        self.get("fs_invocations")
    }

    /// Convenience accessor for vertex invocations.
    pub fn vertex_invocations(&self) -> Option<u64> {
        self.get("vs_invocations")
    }

    /// Convenience accessor for compute invocations.
    pub fn compute_invocations(&self) -> Option<u64> {
        self.get("cs_invocations")
    }
}

/// A `VkQueryPool` configured for pipeline statistics.
pub struct PipelineStatsPool {
    shared: Arc<SharedState>,
    handle: vk::QueryPool,
    enabled: PipelineStats,
    slot_count: u32,
    max_scopes: u32,
    next_query: u32,
    labels: Vec<String>,
}

impl PipelineStatsPool {
    /// Create a pool supporting up to `max_scopes` begin/end pairs.
    ///
    /// Each scope consumes one query slot. The total result buffer size is
    /// `max_scopes * enabled_counter_count * 8` bytes.
    pub fn new(shared: Arc<SharedState>, enabled: PipelineStats, max_scopes: u32) -> Result<Self> {
        let slot_count = enabled.count();
        let ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .query_count(max_scopes)
            .pipeline_statistics(vk::QueryPipelineStatisticFlags::from_raw(enabled.0));
        let handle = unsafe { shared.device.create_query_pool(&ci, None)? };
        Ok(Self {
            shared,
            handle,
            enabled,
            slot_count,
            max_scopes,
            next_query: 0,
            labels: Vec::with_capacity(max_scopes as usize),
        })
    }

    /// Reset all queries. Must be called before the first `begin` of a
    /// recording session. Call from a command buffer, outside a render pass.
    pub fn reset(&mut self, rec: &CommandRecorder<'_>) {
        unsafe {
            self.shared.device.cmd_reset_query_pool(
                rec.raw_buffer(),
                self.handle,
                0,
                self.max_scopes,
            );
        }
        self.next_query = 0;
        self.labels.clear();
    }

    /// Begin a stats scope. The returned handle must be passed to `end`.
    pub fn begin(&mut self, rec: &CommandRecorder<'_>, label: &str) -> PipelineStatsScope {
        let query_index = self.next_query;
        self.next_query += 1;
        self.labels.push(label.to_string());
        unsafe {
            rec.raw_device().cmd_begin_query(
                rec.raw_buffer(),
                self.handle,
                query_index,
                vk::QueryControlFlags::empty(),
            );
        }
        PipelineStatsScope { query_index }
    }

    /// End a stats scope.
    pub fn end(&self, rec: &CommandRecorder<'_>, scope: PipelineStatsScope) {
        unsafe {
            self.shared
                .device
                .cmd_end_query(rec.raw_buffer(), self.handle, scope.query_index);
        }
        // label_index is retained in labels[] for readback.
        //let _ = scope.label_index;
    }

    /// Block and read results for all completed scopes.
    pub fn readback(&self) -> Result<Vec<PipelineStatsResult>> {
        if self.next_query == 0 {
            return Ok(Vec::new());
        }
        let total_u64s = (self.next_query as usize) * (self.slot_count as usize);
        let mut raw = vec![0u64; total_u64s];
        unsafe {
            self.shared.device.get_query_pool_results(
                self.handle,
                0,
                &mut raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let names: Vec<&'static str> = self.enabled.enabled_iter().map(|(_, n)| n).collect();
        let mut out = Vec::with_capacity(self.next_query as usize);
        for i in 0..(self.next_query as usize) {
            let base = i * (self.slot_count as usize);
            let counters = names
                .iter()
                .enumerate()
                .map(|(j, name)| (*name, raw[base + j]))
                .collect();
            out.push(PipelineStatsResult {
                label: self.labels[i].clone(),
                counters,
            });
        }
        Ok(out)
    }
}

impl Drop for PipelineStatsPool {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_query_pool(self.handle, None);
        }
    }
}
