//! GPU memory budget monitor.
//!
//! Queries `VK_EXT_memory_budget` (when available) to track per-heap
//! memory usage against driver-reported budgets. Emits warnings as
//! usage approaches configurable thresholds.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::diagnostic::{self, Severity, Style};

/// Thresholds at which warnings are emitted (fractions, 0.0 - 1.0).
#[derive(Debug, Clone)]
pub struct BudgetThresholds {
    /// First warning level (default 0.80).
    pub warn: f64,
    /// Second warning level (default 0.90).
    pub high: f64,
    /// Critical level (default 0.95).
    pub critical: f64,
}

impl Default for BudgetThresholds {
    fn default() -> Self {
        Self {
            warn: 0.80,
            high: 0.90,
            critical: 0.95,
        }
    }
}

/// Per-heap status.
#[derive(Debug, Clone)]
pub struct HeapStatus {
    /// Heap index.
    pub heap_index: u32,
    /// Heap size in bytes.
    pub heap_size: u64,
    /// Memory property flags for memory types on this heap.
    pub flags: vk::MemoryHeapFlags,
    /// Budget reported by the driver (0 if extension unavailable).
    pub budget: u64,
    /// Usage reported by the driver (0 if extension unavailable).
    pub usage: u64,
    /// Usage as a fraction of budget (0.0 - 1.0).
    pub usage_fraction: f64,
}

/// Point-in-time budget snapshot.
#[derive(Debug, Clone)]
pub struct BudgetSnapshot {
    /// Per-heap status.
    pub heaps: Vec<HeapStatus>,
    /// Whether VK_EXT_memory_budget was available.
    pub has_budget_extension: bool,
}

/// Monitors GPU memory consumption against driver budgets.
pub struct BudgetMonitor {
    shared: Arc<SharedState>,
    thresholds: BudgetThresholds,
    has_budget_ext: bool,
}

impl BudgetMonitor {
    /// Create a new budget monitor.
    ///
    /// Automatically detects whether `VK_EXT_memory_budget` is available
    /// by checking device extensions. If unavailable, polling still works
    /// but reports heap sizes from properties instead of live budgets.
    pub fn new(shared: Arc<SharedState>, thresholds: BudgetThresholds) -> Self {
        // Heuristic: try to query and see if budget values are non-zero.
        let has_ext = Self::probe_budget(&shared);

        Self {
            shared,
            thresholds,
            has_budget_ext: has_ext,
        }
    }

    fn probe_budget(shared: &SharedState) -> bool {
        let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 =
            vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);
        unsafe {
            shared
                .instance
                .get_physical_device_memory_properties2(shared.physical_device, &mut props2);
        }
        // If any budget is non-zero, the extension is working.
        budget_props.heap_budget.iter().any(|&b| b > 0)
    }

    /// Poll current memory budget and usage.
    pub fn poll(&self) -> BudgetSnapshot {
        let mem_props = &self.shared.memory_properties;

        let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 =
            vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);
        unsafe {
            self.shared
                .instance
                .get_physical_device_memory_properties2(self.shared.physical_device, &mut props2);
        }

        let mut heaps = Vec::with_capacity(mem_props.memory_heap_count as usize);
        for i in 0..mem_props.memory_heap_count {
            let idx = i as usize;
            let heap = mem_props.memory_heaps[idx];
            let budget = budget_props.heap_budget[idx];
            let usage = budget_props.heap_usage[idx];

            let effective_budget = if budget > 0 { budget } else { heap.size };
            let fraction = if effective_budget > 0 {
                usage as f64 / effective_budget as f64
            } else {
                0.0
            };

            heaps.push(HeapStatus {
                heap_index: i,
                heap_size: heap.size,
                flags: heap.flags,
                budget: effective_budget,
                usage,
                usage_fraction: fraction,
            });
        }

        BudgetSnapshot {
            heaps,
            has_budget_extension: self.has_budget_ext,
        }
    }

    /// Check budget and return a warning report if any threshold is exceeded.
    pub fn check(&self) -> Option<String> {
        let snapshot = self.poll();
        let mut warnings: Vec<(&HeapStatus, &str)> = Vec::new();

        for heap in &snapshot.heaps {
            if heap.usage_fraction >= self.thresholds.critical {
                warnings.push((heap, "CRITICAL"));
            } else if heap.usage_fraction >= self.thresholds.high {
                warnings.push((heap, "HIGH"));
            } else if heap.usage_fraction >= self.thresholds.warn {
                warnings.push((heap, "WARN"));
            }
        }

        if warnings.is_empty() {
            return None;
        }

        Some(format_budget_report(&snapshot, &warnings, &self.thresholds))
    }
}

fn format_budget_report(
    snapshot: &BudgetSnapshot,
    warnings: &[(&HeapStatus, &str)],
    thresholds: &BudgetThresholds,
) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(1024);

    let worst = warnings
        .iter()
        .map(|(h, _)| h.usage_fraction)
        .fold(0.0_f64, f64::max);

    let sev = if worst >= thresholds.critical {
        Severity::Error
    } else {
        Severity::Warning
    };

    diagnostic::write_header(
        &mut o,
        &s,
        &sev,
        "IGN-M001",
        &format!(
            "GPU memory budget threshold exceeded ({:.0}%)",
            worst * 100.0
        ),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    for (heap, level) in warnings {
        let level_str = match *level {
            "CRITICAL" => s.bold_red(level),
            "HIGH" => s.bold_yellow(level),
            _ => s.yellow(level),
        };

        let device_local = if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            " DEVICE_LOCAL"
        } else {
            ""
        };

        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!("heap {}: {level_str}{device_local}", heap.heap_index,),
        );

        let budget_mb = heap.budget as f64 / (1024.0 * 1024.0);
        let usage_mb = heap.usage as f64 / (1024.0 * 1024.0);

        let bar = render_bar(heap.usage_fraction, 40, &s);
        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "  {bar} {:.0}/{:.0} MiB ({:.1}%)",
                usage_mb,
                budget_mb,
                heap.usage_fraction * 100.0,
            ),
        );
        diagnostic::write_pipe_empty(&mut o, &s);
    }

    if !snapshot.has_budget_extension {
        diagnostic::write_warn(
            &mut o,
            &s,
            "VK_EXT_memory_budget not available, using heap size as budget",
        );
    }

    diagnostic::write_note(
        &mut o,
        &s,
        &format!(
            "thresholds: warn={:.0}% high={:.0}% critical={:.0}%",
            thresholds.warn * 100.0,
            thresholds.high * 100.0,
            thresholds.critical * 100.0,
        ),
    );
    diagnostic::write_help(
        &mut o,
        &s,
        "reduce texture resolution, enable streaming, or\nrelease unused resources to lower memory pressure",
    );

    o
}

fn render_bar(fraction: f64, width: usize, s: &Style) -> String {
    let filled = (fraction * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;

    let fill_str: String = std::iter::repeat('#').take(filled).collect();
    let empty_str: String = std::iter::repeat('-').take(empty).collect();

    let colored_fill = if fraction >= 0.95 {
        s.bold_red(&fill_str)
    } else if fraction >= 0.80 {
        s.yellow(&fill_str)
    } else {
        s.green(&fill_str)
    };

    format!("[{colored_fill}{}]", s.dim(&empty_str))
}
