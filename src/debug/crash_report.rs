//! Automatic post-mortem report generation on `VK_ERROR_DEVICE_LOST`.
//!
//! Collects every debug signal the crate has produced so far (journal,
//! breadcrumbs, descriptor audit, environment context) and writes them
//! to a single Markdown file that can be attached to a bug report.
//!
//! This is pure orchestration: every data source already exists and is
//! already formatted by its own module. This module just bundles them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::fmt::Write;
use std::io::Write as OtherWrite;

use ash::vk;

use crate::diagnostic;

#[cfg(feature = "debug-tools")]
use super::descriptor_audit::DescriptorAuditor;
#[cfg(feature = "debug-tools")]
use super::hang_detector::BreadcrumbBuffer;
#[cfg(feature = "debug-tools")]
use super::journal::SubmissionJournal;

/// Registered action on device lost.
pub type DeviceLostHandler = Box<dyn Fn(&CrashReport) + Send + Sync>;

/// Data gathered for a crash report.
pub struct CrashReport {
    /// Timestamp when the crash was detected, formatted as ISO-8601.
    pub timestamp: String,
    /// The Vulkan error code that triggered the report.
    pub error: vk::Result,
    /// Human-readable name of the error.
    pub error_name: &'static str,
    /// Full markdown body, already rendered.
    pub body: String,
}

impl CrashReport {
    /// Save the report to disk as a Markdown file.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let final_path = path.as_ref();
        // Write to a sibling temp file, flush, then rename. rename is
        // atomic on both NTFS and ext4 on the same filesystem, so if the
        // process crashes during write the original (if any) is preserved
        // and no partial crash report appears on disk.
        let mut tmp_name = final_path
            .file_name()
            .map(|n| n.to_owned())
            .unwrap_or_default();
        tmp_name.push(".partial");
        let tmp_path = final_path.with_file_name(tmp_name);

        {
            let mut f = std::fs::File::create(&tmp_path)?;
            f.write_all(self.body.as_bytes())?;
            f.sync_all()?;
        }

        std::fs::rename(&tmp_path, final_path)?;
        Ok(())
    }

    /// Default output path based on timestamp.
    pub fn default_path(&self) -> PathBuf {
        PathBuf::from(format!("crash_report_{}.md", self.timestamp.replace(':', "-")))
    }
}

/// Coordinates crash reporting by pulling data from every registered source.
///
/// Create once at startup, register data sources with the `attach_*` methods,
/// then call `generate` on device lost.
pub struct CrashReporter {
    #[cfg(feature = "debug-tools")]
    journal: Mutex<Option<Arc<SubmissionJournal>>>,
    #[cfg(feature = "debug-tools")]
    breadcrumbs: Mutex<Vec<Arc<BreadcrumbBuffer>>>,
    #[cfg(feature = "debug-tools")]
    descriptor_auditor: Mutex<Option<Arc<Mutex<DescriptorAuditor>>>>,
    handler: Mutex<Option<DeviceLostHandler>>,
    extra_sections: Mutex<Vec<(String, String)>>,
}

impl CrashReporter {
    /// Create an empty reporter.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "debug-tools")]
            journal: Mutex::new(None),
            #[cfg(feature = "debug-tools")]
            breadcrumbs: Mutex::new(Vec::new()),
            #[cfg(feature = "debug-tools")]
            descriptor_auditor: Mutex::new(None),
            handler: Mutex::new(None),
            extra_sections: Mutex::new(Vec::new()),
        }
    }

    /// Register a submission journal whose contents will be dumped on crash.
    #[cfg(feature = "debug-tools")]
    pub fn attach_journal(&self, journal: Arc<SubmissionJournal>) {
        *self.journal.lock().unwrap() = Some(journal);
    }

    /// Register a breadcrumb buffer whose trail will be included on crash.
    /// Multiple buffers may be attached (one per queue/submission context).
    #[cfg(feature = "debug-tools")]
    pub fn attach_breadcrumbs(&self, buffer: Arc<BreadcrumbBuffer>) {
        self.breadcrumbs.lock().unwrap().push(buffer);
    }

    /// Register a descriptor auditor for stale reference reporting.
    #[cfg(feature = "debug-tools")]
    pub fn attach_descriptor_auditor(&self, auditor: Arc<Mutex<DescriptorAuditor>>) {
        *self.descriptor_auditor.lock().unwrap() = Some(auditor);
    }

    /// Register a custom section that will be appended to every report.
    /// Useful for application-specific context (scene name, frame number,
    /// user inputs, etc).
    pub fn add_section(&self, title: impl Into<String>, body: impl Into<String>) {
        self.extra_sections
            .lock()
            .unwrap()
            .push((title.into(), body.into()));
    }

    /// Install a handler to be invoked when `trigger` is called.
    pub fn on_device_lost<F>(&self, handler: F)
    where
        F: Fn(&CrashReport) + Send + Sync + 'static,
    {
        *self.handler.lock().unwrap() = Some(Box::new(handler));
    }

    /// Build a report without invoking the handler. Useful for manual dumps.
    pub fn generate(&self, error: vk::Result) -> CrashReport {
        let timestamp = current_iso_timestamp();
        let error_name = diagnostic::vk_result_name(error);

        let mut body = String::with_capacity(16 * 1024);
        writeln!(body, "# Vulkan Crash Report").unwrap();
        writeln!(body).unwrap();
        writeln!(body, "- **Timestamp:** {timestamp}").unwrap();
        writeln!(body, "- **Error:** `{error_name}`").unwrap();
        writeln!(body, "- **Process:** {}", std::process::id()).unwrap();
        writeln!(body, "- **Uptime:** {}", diagnostic::app_uptime()).unwrap();
        writeln!(body).unwrap();

        // Environment section.
        writeln!(body, "## Environment").unwrap();
        writeln!(body).unwrap();
        writeln!(body, "```").unwrap();
        // Write env via a sink that strips ANSI for markdown readability.
        let s = crate::diagnostic::Style { on: false };
        let mut env_buf = String::new();
        crate::diagnostic::write_environment_block(&mut env_buf, &s);
        // Remove the pipe frame prefix for cleaner markdown.
        for line in env_buf.lines() {
            let trimmed = line.trim_start_matches(|c: char| c == ' ' || c == '|');
            writeln!(body, "{trimmed}").unwrap();
        }
        writeln!(body, "```").unwrap();
        writeln!(body).unwrap();

        // Submission journal.
        #[cfg(feature = "debug-tools")]
        {
            if let Some(j) = self.journal.lock().unwrap().as_ref() {
                writeln!(body, "## Submission Journal").unwrap();
                writeln!(body).unwrap();
                writeln!(body, "```").unwrap();
                // dump_with_error() already formats with pipes; strip them.
                let raw = j.dump_with_error(error);
                for line in raw.lines() {
                    let cleaned = strip_ansi_and_pipe(line);
                    writeln!(body, "{cleaned}").unwrap();
                }
                writeln!(body, "```").unwrap();
                writeln!(body).unwrap();
            }
        }

        // Breadcrumbs.
        #[cfg(feature = "debug-tools")]
        {
            let bcs = self.breadcrumbs.lock().unwrap();
            if !bcs.is_empty() {
                writeln!(body, "## Breadcrumb Trails").unwrap();
                writeln!(body).unwrap();
                for (i, bc) in bcs.iter().enumerate() {
                    writeln!(body, "### Trail {i}").unwrap();
                    writeln!(body).unwrap();
                    writeln!(body, "| ID | Label | Status |").unwrap();
                    writeln!(body, "|----|-------|--------|").unwrap();
                    for (crumb, done) in bc.readback() {
                        let status = if done { "completed" } else { "**PENDING**" };
                        writeln!(body, "| {} | {} | {} |", crumb.id, crumb.label, status).unwrap();
                    }
                    writeln!(body).unwrap();
                }
            }
        }

        // Descriptor auditor snapshot.
        #[cfg(feature = "debug-tools")]
        {
            if let Some(aud) = self.descriptor_auditor.lock().unwrap().as_ref() {
                let auditor = aud.lock().unwrap();
                let issues = auditor.audit_all();
                if !issues.is_empty() {
                    writeln!(body, "## Descriptor Stale References").unwrap();
                    writeln!(body).unwrap();
                    writeln!(body, "```").unwrap();
                    let raw = auditor.report(&issues);
                    for line in raw.lines() {
                        let cleaned = strip_ansi_and_pipe(line);
                        writeln!(body, "{cleaned}").unwrap();
                    }
                    writeln!(body, "```").unwrap();
                    writeln!(body).unwrap();
                }
            }
        }

        // Custom sections appended last so app-specific context
        // does not compete with the structural dumps.
        let extras = self.extra_sections.lock().unwrap();
        for (title, content) in extras.iter() {
            writeln!(body, "## {title}").unwrap();
            writeln!(body).unwrap();
            writeln!(body, "{content}").unwrap();
            writeln!(body).unwrap();
        }

        CrashReport {
            timestamp,
            error,
            error_name,
            body,
        }
    }

    /// Invoke the full reporting pipeline: build a report and deliver it
    /// to the registered handler. If no handler is registered, writes the
    /// report to `crash_report_{timestamp}.md` in the current directory.
    pub fn trigger(&self, error: vk::Result) -> CrashReport {
        let report = self.generate(error);
        match self.handler.lock().unwrap().as_ref() {
            Some(h) => h(&report),
            None => {
                let path = report.default_path();
                if let Err(e) = report.write_to_file(&path) {
                    // Writable CWD is not guaranteed in release builds.
                    // Fall back to the OS temp directory.
                    let fallback = std::env::temp_dir().join(path.file_name().unwrap_or_default());
                    match report.write_to_file(&fallback) {
                        Ok(()) => {
                            eprintln!(
                                "ignis: crash report written to {fallback:?} (CWD was not writable: {e})"
                            );
                        }
                        Err(e2) => {
                            eprintln!(
                                "ignis: failed to write crash report to {path:?} ({e}) or {fallback:?} ({e2})"
                            );
                        }
                    }
                } else {
                    eprintln!("ignis: crash report written to {path:?}");
                }
            }
        }
        report
    }
}

impl Default for CrashReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Return a simple ISO-8601 timestamp string without external crates.
///
/// Windows priority: uses `GetSystemTimeAsFileTime`. Linux falls back to
/// `clock_gettime`. macOS uses the same Linux path.
fn current_iso_timestamp() -> String {
    // We only need a unique timestamp, not accurate calendar time.
    // Epoch seconds plus microseconds from SystemTime is enough and
    // requires no platform-specific code.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let micros = now.subsec_micros();
    // Break into UTC date components manually.
    let (y, mo, d, h, mi, s) = unix_epoch_to_utc(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{micros:06}Z")
}

/// Convert Unix epoch seconds to UTC date components.
/// Algorithm: Howard Hinnant's civil_from_days.
fn unix_epoch_to_utc(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let days = (secs / 86400) as i64;
    let sod = (secs % 86400) as u32;
    let h = sod / 3600;
    let m = (sod % 3600) / 60;
    let s = sod % 60;

    let z = days + 719468;
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if mo <= 2 { y + 1 } else { y };
    (y as i32, mo, d, h, m, s)
}

/// Strip ANSI escape sequences and a leading pipe frame prefix from a line.
///
/// Only the first pipe character after leading whitespace is treated as
/// the frame. Any subsequent pipes (for example in a GPU name that
/// contains a literal `|`) are preserved.
fn strip_ansi_and_pipe(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_esc = false;
    for ch in line.chars() {
        if in_esc {
            if ch == 'm' {
                in_esc = false;
            }
            continue;
        }
        if ch == '\x1b' {
            in_esc = true;
            continue;
        }
        out.push(ch);
    }
    // Strip "   |" or "  |" prefix (ignis frame), leaving subsequent pipes alone.
    let after_ws = out.trim_start_matches(|c: char| c == ' ');
    let leading_ws_len = out.len() - after_ws.len();
    let body = if let Some(rest) = after_ws.strip_prefix("| ") {
        rest
    } else if let Some(rest) = after_ws.strip_prefix('|') {
        rest
    } else {
        &out[leading_ws_len..]
    };
    body.to_string()
}