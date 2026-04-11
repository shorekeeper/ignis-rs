//! GPU timestamp profiler.
//!
//! [`GpuProfiler`] manages `VkQueryPool` timestamp queries and produces
//! hierarchical timing reports. Insert scopes into command buffers
//! and read back results after GPU execution.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ignis::debug::profiler::*; use ash::vk;
//! # fn example(ignis: &Ignis, pool: &CommandPool,
//! #            queue: &AsyncQueue) -> Result<()> {
//! let mut profiler = GpuProfiler::new(ignis.shared_state(), 128)?;
//!
//! let cmd = pool.allocate_primary()?;
//! let rec = pool.begin_primary(cmd)?;
//! profiler.reset(&rec);
//! let scope = profiler.begin_scope(&rec, "geometry_pass");
//! // ... draw commands ...
//! profiler.end_scope(&rec, scope);
//! let cmd = rec.end()?;
//! queue.submit_simple(cmd)?.wait()?;
//!
//! for result in profiler.readback()? {
//!     println!("{}: {:.3}ms", result.label, result.elapsed_ms);
//! }
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::Result;
use crate::command::CommandRecorder;

/// An active profiling scope handle.
#[derive(Debug, Clone, Copy)]
pub struct ScopeHandle {
    begin_query: u32,
    end_query: u32,
    label_index: usize,
}

/// A timing result from a completed scope.
#[derive(Debug, Clone)]
pub struct ScopeResult {
    /// Human-readable label.
    pub label: String,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: f64,
    /// Elapsed time in nanoseconds.
    pub elapsed_ns: u64,
    /// Begin timestamp (raw ticks).
    pub begin_tick: u64,
    /// End timestamp (raw ticks).
    pub end_tick: u64,
}

/// GPU timestamp query profiler.
pub struct GpuProfiler {
    shared: Arc<SharedState>,
    query_pool: vk::QueryPool,
    max_queries: u32,
    next_query: u32,
    timestamp_period: f64,
    labels: Vec<String>,
    scopes: Vec<(u32, u32, usize)>, // (begin, end, label_idx)
}

impl GpuProfiler {
    /// Create a profiler with the given maximum number of timestamp queries.
    ///
    /// Each scope uses 2 queries. `max_queries` should be at least
    /// `2 * max_scopes`.
    pub fn new(shared: &Arc<SharedState>, max_queries: u32) -> Result<Self> {
        let ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(max_queries);
        let query_pool = unsafe { shared.device.create_query_pool(&ci, None)? };
        let period = shared.device_properties.limits.timestamp_period as f64;
        Ok(Self {
            shared: Arc::clone(shared),
            query_pool,
            max_queries,
            next_query: 0,
            timestamp_period: period,
            labels: Vec::new(),
            scopes: Vec::new(),
        })
    }

    /// Reset the query pool. Must be called before recording new scopes.
    pub fn reset(&mut self, rec: &CommandRecorder<'_>) {
        unsafe {
            rec.device
                .cmd_reset_query_pool(rec.buffer, self.query_pool, 0, self.max_queries);
        }
        self.next_query = 0;
        self.labels.clear();
        self.scopes.clear();
    }

    /// Begin a profiling scope. Returns a handle to pass to [`end_scope`](Self::end_scope).
    pub fn begin_scope(&mut self, rec: &CommandRecorder<'_>, label: &str) -> ScopeHandle {
        let begin_query = self.next_query;
        self.next_query += 1;
        let end_query = self.next_query;
        self.next_query += 1;
        let label_index = self.labels.len();
        self.labels.push(label.to_string());

        unsafe {
            rec.device.cmd_write_timestamp(
                rec.buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                begin_query,
            );
        }

        ScopeHandle {
            begin_query,
            end_query,
            label_index,
        }
    }

    /// End a profiling scope.
    pub fn end_scope(&mut self, rec: &CommandRecorder<'_>, handle: ScopeHandle) {
        unsafe {
            rec.device.cmd_write_timestamp(
                rec.buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                handle.end_query,
            );
        }
        self.scopes.push((
            handle.begin_query,
            handle.end_query,
            handle.label_index,
        ));
    }

    /// Read back timing results after GPU execution has completed.
    pub fn readback(&self) -> Result<Vec<ScopeResult>> {
        if self.next_query == 0 {
            return Ok(Vec::new());
        }
        let mut timestamps = vec![0u64; self.next_query as usize];
        unsafe {
            self.shared.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let results = self
            .scopes
            .iter()
            .map(|&(begin, end, label_idx)| {
                let begin_tick = timestamps[begin as usize];
                let end_tick = timestamps[end as usize];
                let delta = end_tick.saturating_sub(begin_tick);
                let elapsed_ns = (delta as f64 * self.timestamp_period) as u64;
                let elapsed_ms = elapsed_ns as f64 / 1_000_000.0;
                ScopeResult {
                    label: self.labels[label_idx].clone(),
                    elapsed_ms,
                    elapsed_ns,
                    begin_tick,
                    end_tick,
                }
            })
            .collect();
        Ok(results)
    }
}

impl Drop for GpuProfiler {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_query_pool(self.query_pool, None);
        }
    }
}