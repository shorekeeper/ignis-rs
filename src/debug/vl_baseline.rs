//! Validation Layer baseline capture and diff for CI-grade regression detection.
//!
//! [`VlBaseline`] is a deterministic snapshot of every VUID emitted during
//! a single process run, grouped by VUID identifier with occurrence
//! counts. Two snapshots can be diffed to detect:
//!
//! - **New VUIDs**: present in the current run but not in the baseline.
//!   These are regressions: a code change introduced a new validation
//!   warning that did not exist before.
//! - **Removed VUIDs**: present in the baseline but not in the current
//!   run. These are improvements: a fix eliminated a previously-emitted
//!   warning. The diff reports them so progress is visible.
//! - **Frequency changes**: the VUID is in both, but the count differs.
//!   Increases are regressions; decreases are improvements.
//!
//! # CI integration
//!
//! ```text
//! # In CI:
//! cargo run --example my_smoke_test --features full
//! # Then verify against checked-in baseline:
//! ```
//!
//! ```rust,ignore
//! let diff = ctx.diff_vl_baseline("ci/vl_baseline.vl")?;
//! if diff.has_regressions() {
//!     eprintln!("{diff}");
//!     std::process::exit(1);
//! }
//! ```
//!
//! When the smoke test introduces a new VUID, the diff prints it and the
//! CI build fails. Engineers review the new VUID, fix it, and either the
//! diff goes away or the baseline is updated intentionally.
//!
//! # File format
//!
//! Tab-separated values with a single header line. One row per unique
//! VUID, sorted alphabetically for determinism (so `git diff` of the
//! baseline file shows only meaningful changes).
//!
//! ```text
//! # IGNIS-VL-BASELINE v1
//! # vuid	severity	category	function	count
//! VUID-vkCmdCopyBuffer-size-00115	Error	MemoryBinding	vkCmdCopyBuffer	5
//! VUID-vkBeginCommandBuffer-commandBuffer-00049	Error	Other	vkBeginCommandBuffer	1
//! ```
//!
//! Plain text means the file diffs cleanly in `git`, can be edited by
//! hand to whitelist known-good warnings, and works with any text tool.
//!
//! # Hooking into the VL pipeline
//!
//! The collector is fed automatically from the validation layer
//! messenger callback in [`debug::validation`](super::validation), one
//! call site per parsed diagnostic. It runs before any user-configurable
//! filtering or dedup, so suppressing a VUID via [`ignis_vl!`] does NOT
//! remove it from the baseline. The baseline reflects raw layer output.
//!
//! [`ignis_vl!`]: crate::ignis_vl

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use super::validation_forensic::{DiagnosticCategory, LayerSeverity, ValidationDiagnostic};

/// Format version stamped into the baseline file. Bumped if the
/// serialization layout changes incompatibly.
pub const BASELINE_FORMAT_VERSION: u32 = 1;

/// One row of a baseline: a unique VUID with its severity, category,
/// function context, and total occurrence count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineEntry {
    /// Full VUID identifier, e.g. `"VUID-vkCmdCopyBuffer-size-00115"`.
    pub vuid: String,
    /// Severity label as written on disk: `"Error"`, `"Warning"`, or `"Info"`.
    pub severity: String,
    /// Category label as written on disk (matches
    /// [`DiagnosticCategory`] variant names).
    pub category: String,
    /// Vulkan function that triggered the diagnostic.
    pub function: String,
    /// Total number of times this VUID was emitted.
    pub count: u64,
}

/// A complete deterministic snapshot of all VUIDs seen so far, ordered
/// alphabetically by VUID for stable output.
#[derive(Debug, Clone)]
pub struct VlBaseline {
    /// File format version.
    pub version: u32,
    /// Entries keyed by VUID.
    pub entries: BTreeMap<String, BaselineEntry>,
}

impl VlBaseline {
    /// Total number of VUID occurrences across all entries.
    pub fn total_count(&self) -> u64 {
        self.entries.values().map(|e| e.count).sum()
    }

    /// Number of distinct VUIDs.
    pub fn unique_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of error-severity entries.
    pub fn error_count(&self) -> u64 {
        self.entries
            .values()
            .filter(|e| e.severity == "Error")
            .map(|e| e.count)
            .sum()
    }

    /// Number of warning-severity entries.
    pub fn warning_count(&self) -> u64 {
        self.entries
            .values()
            .filter(|e| e.severity == "Warning")
            .map(|e| e.count)
            .sum()
    }

    /// Save the baseline to disk in TSV format. Atomic via temp file
    /// rename so an in-progress write cannot corrupt the existing file.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp_path = std::path::PathBuf::from(tmp);

        let body = self.serialize();
        std::fs::write(&tmp_path, body)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Load a baseline from disk.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::deserialize(&s).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    /// Render the baseline as the on-disk TSV string.
    pub fn serialize(&self) -> String {
        let mut o = String::with_capacity(self.entries.len() * 80 + 64);
        o.push_str(&format!(
            "# IGNIS-VL-BASELINE v{}\n",
            self.version
        ));
        o.push_str("# vuid\tseverity\tcategory\tfunction\tcount\n");
        for entry in self.entries.values() {
            o.push_str(&entry.vuid);
            o.push('\t');
            o.push_str(&entry.severity);
            o.push('\t');
            o.push_str(&entry.category);
            o.push('\t');
            o.push_str(&entry.function);
            o.push('\t');
            o.push_str(&entry.count.to_string());
            o.push('\n');
        }
        o
    }

    /// Parse a TSV-formatted baseline string.
    pub fn deserialize(text: &str) -> Result<Self, String> {
        let mut version = BASELINE_FORMAT_VERSION;
        let mut entries: BTreeMap<String, BaselineEntry> = BTreeMap::new();

        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            // Header version comment.
            if let Some(rest) = line.strip_prefix("# IGNIS-VL-BASELINE v") {
                version = rest
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| format!("line {}: invalid version", line_no + 1))?;
                continue;
            }
            // Other comments are ignored.
            if line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 5 {
                return Err(format!(
                    "line {}: expected 5 tab-separated fields, got {}",
                    line_no + 1,
                    parts.len()
                ));
            }
            let count = parts[4]
                .parse::<u64>()
                .map_err(|_| format!("line {}: count is not a u64", line_no + 1))?;

            let entry = BaselineEntry {
                vuid: parts[0].to_string(),
                severity: parts[1].to_string(),
                category: parts[2].to_string(),
                function: parts[3].to_string(),
                count,
            };
            entries.insert(entry.vuid.clone(), entry);
        }

        Ok(Self { version, entries })
    }

    /// Compute the difference between this baseline (treated as the
    /// reference) and `current` (the new run).
    ///
    /// Regressions are characterized by either a new VUID in `current`
    /// or an increased count for an existing one.
    pub fn diff(&self, current: &VlBaseline) -> VlDiffReport {
        let mut new_vuids = Vec::new();
        let mut removed_vuids = Vec::new();
        let mut frequency_changes = Vec::new();

        for (vuid, cur_entry) in &current.entries {
            match self.entries.get(vuid) {
                None => new_vuids.push(cur_entry.clone()),
                Some(base_entry) if base_entry.count != cur_entry.count => {
                    frequency_changes.push(FreqChange {
                        vuid: vuid.clone(),
                        severity: cur_entry.severity.clone(),
                        category: cur_entry.category.clone(),
                        function: cur_entry.function.clone(),
                        baseline_count: base_entry.count,
                        current_count: cur_entry.count,
                    });
                }
                Some(_) => {}
            }
        }

        for (vuid, base_entry) in &self.entries {
            if !current.entries.contains_key(vuid) {
                removed_vuids.push(base_entry.clone());
            }
        }

        // Sort outputs deterministically.
        new_vuids.sort_by(|a, b| a.vuid.cmp(&b.vuid));
        removed_vuids.sort_by(|a, b| a.vuid.cmp(&b.vuid));
        frequency_changes.sort_by(|a, b| a.vuid.cmp(&b.vuid));

        VlDiffReport {
            new_vuids,
            removed_vuids,
            frequency_changes,
            baseline_total: self.total_count(),
            current_total: current.total_count(),
            baseline_unique: self.unique_count(),
            current_unique: current.unique_count(),
        }
    }
}

/// One frequency-change entry: a VUID present in both baseline and
/// current, but with a different count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreqChange {
    /// The VUID identifier.
    pub vuid: String,
    /// Severity label.
    pub severity: String,
    /// Category label.
    pub category: String,
    /// Function name.
    pub function: String,
    /// Count in the baseline file.
    pub baseline_count: u64,
    /// Count observed in the current run.
    pub current_count: u64,
}

impl FreqChange {
    /// Signed delta: positive means the current run emitted MORE of this
    /// VUID than the baseline (regression); negative means fewer
    /// (improvement).
    pub fn delta(&self) -> i64 {
        self.current_count as i64 - self.baseline_count as i64
    }
}

/// Result of comparing a baseline to a current run.
#[derive(Debug, Clone)]
pub struct VlDiffReport {
    /// VUIDs present in current but not in baseline.
    pub new_vuids: Vec<BaselineEntry>,
    /// VUIDs present in baseline but not in current.
    pub removed_vuids: Vec<BaselineEntry>,
    /// VUIDs with different counts.
    pub frequency_changes: Vec<FreqChange>,
    /// Total VUID emissions in the baseline.
    pub baseline_total: u64,
    /// Total VUID emissions in the current run.
    pub current_total: u64,
    /// Number of distinct VUIDs in the baseline.
    pub baseline_unique: usize,
    /// Number of distinct VUIDs in the current run.
    pub current_unique: usize,
}

impl VlDiffReport {
    /// True if the diff contains any signal that should fail CI:
    /// a new VUID, or an existing VUID with an increased count.
    pub fn has_regressions(&self) -> bool {
        !self.new_vuids.is_empty()
            || self.frequency_changes.iter().any(|c| c.delta() > 0)
    }

    /// True if the diff contains improvements: removed VUIDs or
    /// decreased counts.
    pub fn has_improvements(&self) -> bool {
        !self.removed_vuids.is_empty()
            || self.frequency_changes.iter().any(|c| c.delta() < 0)
    }

    /// True if baseline and current are identical (same VUIDs, same counts).
    pub fn is_identical(&self) -> bool {
        self.new_vuids.is_empty()
            && self.removed_vuids.is_empty()
            && self.frequency_changes.is_empty()
    }

    /// Number of distinct regression entries (new VUIDs +
    /// frequency increases).
    pub fn regression_count(&self) -> usize {
        self.new_vuids.len()
            + self
                .frequency_changes
                .iter()
                .filter(|c| c.delta() > 0)
                .count()
    }
}

impl fmt::Display for VlDiffReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::diagnostic::{
            write_diagnostic_end, write_header, write_kv, write_pipe, write_pipe_empty,
            write_pipe_raw, write_section, Severity, Style,
        };

        let s = Style::detect();
        let mut o = String::with_capacity(2048);

        let sev = if self.has_regressions() {
            Severity::Error
        } else if self.has_improvements() {
            Severity::Info
        } else {
            Severity::Info
        };

        let title = if self.is_identical() {
            "VL baseline diff: no changes".to_string()
        } else if self.has_regressions() {
            format!(
                "VL baseline diff: {} regression(s)",
                self.regression_count()
            )
        } else {
            "VL baseline diff: improvements only".to_string()
        };

        write_header(&mut o, &s, &sev, "IGN-VLB", &title);
        write_pipe_empty(&mut o, &s);

        write_kv(
            &mut o,
            &s,
            "Baseline",
            &format!(
                "{} unique VUID(s), {} total emission(s)",
                self.baseline_unique, self.baseline_total
            ),
        );
        write_kv(
            &mut o,
            &s,
            "Current",
            &format!(
                "{} unique VUID(s), {} total emission(s)",
                self.current_unique, self.current_total
            ),
        );

        if self.is_identical() {
            write_pipe_empty(&mut o, &s);
            write_pipe_raw(
                &mut o,
                &s,
                &s.bold_green("  ✓ no differences detected"),
            );
            write_diagnostic_end(&mut o, &s, &sev);
            return f.write_str(&o);
        }

        if !self.new_vuids.is_empty() {
            write_section(
                &mut o,
                &s,
                &format!("New VUIDs ({} regression)", self.new_vuids.len()),
            );
            for entry in &self.new_vuids {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  {} {} ×{} ({}) {}",
                        s.bold_red("+"),
                        s.bright_white(&entry.vuid),
                        entry.count,
                        s.dim(&entry.severity),
                        s.dim(&entry.function),
                    ),
                );
            }
        }

        let increases: Vec<&FreqChange> = self
            .frequency_changes
            .iter()
            .filter(|c| c.delta() > 0)
            .collect();
        if !increases.is_empty() {
            write_section(
                &mut o,
                &s,
                &format!(
                    "Frequency increases ({} regression)",
                    increases.len()
                ),
            );
            for c in &increases {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  {} {} {} -> {} ({}{})",
                        s.bold_red("↑"),
                        s.bright_white(&c.vuid),
                        c.baseline_count,
                        c.current_count,
                        s.bold_red("+"),
                        s.bold_red(&format!("{}", c.delta())),
                    ),
                );
            }
        }

        let decreases: Vec<&FreqChange> = self
            .frequency_changes
            .iter()
            .filter(|c| c.delta() < 0)
            .collect();
        if !decreases.is_empty() {
            write_section(
                &mut o,
                &s,
                &format!(
                    "Frequency decreases ({} improvement)",
                    decreases.len()
                ),
            );
            for c in &decreases {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  {} {} {} -> {} ({})",
                        s.bold_green("↓"),
                        s.bright_white(&c.vuid),
                        c.baseline_count,
                        c.current_count,
                        s.bold_green(&format!("{}", c.delta())),
                    ),
                );
            }
        }

        if !self.removed_vuids.is_empty() {
            write_section(
                &mut o,
                &s,
                &format!("Removed VUIDs ({} improvement)", self.removed_vuids.len()),
            );
            for entry in &self.removed_vuids {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  {} {} (was ×{})",
                        s.bold_green("-"),
                        s.bright_white(&entry.vuid),
                        entry.count,
                    ),
                );
            }
        }

        write_pipe_empty(&mut o, &s);
        if self.has_regressions() {
            write_pipe(
                &mut o,
                &s,
                "regressions detected: a code change introduced new validation issues.",
            );
            write_pipe(
                &mut o,
                &s,
                "either fix the new VUIDs or update the baseline if intentional.",
            );
        } else {
            write_pipe(
                &mut o,
                &s,
                "no regressions; consider updating the baseline to lock in the improvement.",
            );
        }

        write_diagnostic_end(&mut o, &s, &sev);
        f.write_str(&o)
    }
}

// ---- Internal collector -------------------------------------------------

/// Process-global state. Lazily initialized on first feed/snapshot call.
struct Collector {
    counts: BTreeMap<String, BaselineEntry>,
}

static COLLECTOR: Mutex<Option<Collector>> = Mutex::new(None);

fn ensure_collector() {
    let mut g = COLLECTOR.lock().unwrap();
    if g.is_none() {
        *g = Some(Collector {
            counts: BTreeMap::new(),
        });
    }
}

fn category_label(cat: DiagnosticCategory) -> &'static str {
    match cat {
        DiagnosticCategory::UsageFlagMismatch => "UsageFlagMismatch",
        DiagnosticCategory::LayoutTransition => "LayoutTransition",
        DiagnosticCategory::SynchronizationHazard => "SynchronizationHazard",
        DiagnosticCategory::DescriptorMismatch => "DescriptorMismatch",
        DiagnosticCategory::PipelineMismatch => "PipelineMismatch",
        DiagnosticCategory::ObjectLifetime => "ObjectLifetime",
        DiagnosticCategory::MemoryBinding => "MemoryBinding",
        DiagnosticCategory::QueueSubmission => "QueueSubmission",
        DiagnosticCategory::FeatureNotEnabled => "FeatureNotEnabled",
        DiagnosticCategory::Other => "Other",
    }
}

fn severity_label(sev: LayerSeverity) -> &'static str {
    match sev {
        LayerSeverity::Error => "Error",
        LayerSeverity::Warning => "Warning",
        LayerSeverity::Info => "Info",
    }
}

/// Feed one parsed validation diagnostic into the global collector.
///
/// Called by the validation layer messenger (in
/// [`debug::validation`](super::validation)) for every successfully
/// parsed VUID. Suppression and dedup do not affect this path.
pub(crate) fn feed(diag: &ValidationDiagnostic) {
    ensure_collector();
    let mut g = COLLECTOR.lock().unwrap();
    let c = g.as_mut().unwrap();
    let entry = c
        .counts
        .entry(diag.vuid.clone())
        .or_insert_with(|| BaselineEntry {
            vuid: diag.vuid.clone(),
            severity: severity_label(diag.severity).to_string(),
            category: category_label(diag.category).to_string(),
            function: diag.function.clone(),
            count: 0,
        });
    entry.count = entry.count.saturating_add(1);
}

/// Snapshot the current accumulated state of the collector.
pub fn snapshot() -> VlBaseline {
    ensure_collector();
    let g = COLLECTOR.lock().unwrap();
    let c = g.as_ref().unwrap();
    VlBaseline {
        version: BASELINE_FORMAT_VERSION,
        entries: c.counts.clone(),
    }
}

/// Reset the collector. The next snapshot will be empty until new
/// diagnostics arrive.
pub fn reset() {
    let mut g = COLLECTOR.lock().unwrap();
    if let Some(c) = g.as_mut() {
        c.counts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(vuid: &str, count: u64) -> BaselineEntry {
        BaselineEntry {
            vuid: vuid.to_string(),
            severity: "Error".to_string(),
            category: "Other".to_string(),
            function: "vkUnknown".to_string(),
            count,
        }
    }

    #[test]
    fn serialize_round_trip() {
        let mut entries = BTreeMap::new();
        entries.insert("VUID-A-001".to_string(), entry("VUID-A-001", 3));
        entries.insert("VUID-B-002".to_string(), entry("VUID-B-002", 1));
        let baseline = VlBaseline {
            version: 1,
            entries,
        };

        let s = baseline.serialize();
        assert!(s.starts_with("# IGNIS-VL-BASELINE v1\n"));
        let parsed = VlBaseline::deserialize(&s).unwrap();
        assert_eq!(parsed.entries.len(), 2);
        assert_eq!(parsed.entries["VUID-A-001"].count, 3);
        assert_eq!(parsed.entries["VUID-B-002"].count, 1);
    }

    #[test]
    fn deserialize_rejects_bad_format() {
        let bad = "not\ta\tvalid\tline\n";
        assert!(VlBaseline::deserialize(bad).is_err());
    }

    #[test]
    fn deserialize_handles_comments_and_blank_lines() {
        let s = "# IGNIS-VL-BASELINE v1\n\
                 # comment\n\
                 \n\
                 VUID-X-001\tError\tOther\tvkX\t5\n";
        let parsed = VlBaseline::deserialize(s).unwrap();
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries["VUID-X-001"].count, 5);
    }

    #[test]
    fn diff_detects_new_vuids() {
        let mut base_entries = BTreeMap::new();
        base_entries.insert("VUID-A".to_string(), entry("VUID-A", 1));
        let baseline = VlBaseline {
            version: 1,
            entries: base_entries,
        };

        let mut cur_entries = BTreeMap::new();
        cur_entries.insert("VUID-A".to_string(), entry("VUID-A", 1));
        cur_entries.insert("VUID-B".to_string(), entry("VUID-B", 2));
        let current = VlBaseline {
            version: 1,
            entries: cur_entries,
        };

        let diff = baseline.diff(&current);
        assert!(diff.has_regressions());
        assert_eq!(diff.new_vuids.len(), 1);
        assert_eq!(diff.new_vuids[0].vuid, "VUID-B");
        assert!(diff.frequency_changes.is_empty());
    }

    #[test]
    fn diff_detects_frequency_changes() {
        let mut be = BTreeMap::new();
        be.insert("VUID-A".to_string(), entry("VUID-A", 2));
        let baseline = VlBaseline {
            version: 1,
            entries: be,
        };

        let mut ce = BTreeMap::new();
        ce.insert("VUID-A".to_string(), entry("VUID-A", 5));
        let current = VlBaseline {
            version: 1,
            entries: ce,
        };

        let diff = baseline.diff(&current);
        assert!(diff.has_regressions());
        assert_eq!(diff.frequency_changes.len(), 1);
        assert_eq!(diff.frequency_changes[0].delta(), 3);
    }

    #[test]
    fn diff_detects_removed_vuids_as_improvement() {
        let mut be = BTreeMap::new();
        be.insert("VUID-A".to_string(), entry("VUID-A", 1));
        be.insert("VUID-B".to_string(), entry("VUID-B", 1));
        let baseline = VlBaseline {
            version: 1,
            entries: be,
        };

        let mut ce = BTreeMap::new();
        ce.insert("VUID-A".to_string(), entry("VUID-A", 1));
        let current = VlBaseline {
            version: 1,
            entries: ce,
        };

        let diff = baseline.diff(&current);
        assert!(!diff.has_regressions());
        assert!(diff.has_improvements());
        assert_eq!(diff.removed_vuids.len(), 1);
        assert_eq!(diff.removed_vuids[0].vuid, "VUID-B");
    }

    #[test]
    fn identical_baselines_have_no_diff() {
        let mut e = BTreeMap::new();
        e.insert("VUID-A".to_string(), entry("VUID-A", 7));
        let a = VlBaseline {
            version: 1,
            entries: e.clone(),
        };
        let b = VlBaseline { version: 1, entries: e };
        let diff = a.diff(&b);
        assert!(diff.is_identical());
        assert!(!diff.has_regressions());
        assert!(!diff.has_improvements());
    }

    #[test]
    fn collector_aggregates_across_feeds() {
        // Reset to ensure test isolation when running with other tests.
        reset();

        let diag = ValidationDiagnostic {
            vuid: "VUID-TEST-001".to_string(),
            vuid_suffix: "001".to_string(),
            function: "vkTest".to_string(),
            parameter: None,
            objects: Vec::new(),
            values: Vec::new(),
            raw_body: String::new(),
            category: DiagnosticCategory::Other,
            severity: LayerSeverity::Error,
            knowledge: None,
            submit_backtrace: None,
        };

        feed(&diag);
        feed(&diag);
        feed(&diag);

        let snap = snapshot();
        let entry = snap.entries.get("VUID-TEST-001").expect("entry present");
        assert_eq!(entry.count, 3);
        assert_eq!(entry.severity, "Error");

        reset();
        let snap2 = snapshot();
        assert_eq!(snap2.entries.len(), 0);
    }
}