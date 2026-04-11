//! Shared diagnostic formatting primitives for all ignis debug modules.
//!
//! Provides ANSI terminal styling, structured report builders, and
//! helper functions used by every debugging subsystem. This module is
//! the single source of truth for all diagnostic output formatting
//! across ignis — every error, warning, and informational message
//! flows through these primitives to ensure visual consistency.
//!
//! # Visual Design
//!
//! Diagnostics use a consistent framed format with severity-colored
//! borders, bright-white content, timestamps, thread identity, process
//! ID, application uptime, and optional GPU environment context. This
//! ensures diagnostic output is visually distinct from normal
//! application output even in dense log streams.
//!
//! ```text
//!  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
//!  ▓▓                         CRITICAL ERROR DETECTED                           ▓▓
//!  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
//!
//!  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//!  🔴 error[IGN-H001]: front guard band corruption
//!  at 14:23:45.123 │ thread="main" │ pid=12345 │ uptime=3.21s
//!  spec: Vulkan §11.6 Resource Memory Association
//!    --> VkDeviceMemory(0xdc..dc) offset=448 size=128B
//!     |
//!     |  ── Environment ──────────────────────
//!     |  GPU: NVIDIA GeForce RTX 4090
//!     |  Driver: 546.33
//!     |  Vulkan API: 1.3.270
//!     |  OS: windows
//!     |  PID: 12345
//!     |  Uptime: 3.21s
//!     |  Memory heaps: 3
//!     |  Features: tracking, debug-tools, slab-allocator
//!     |
//!     |  ── Backtrace ────────────────────────
//!     |    0: ignis::debug::hardened::HardenedAllocator::free
//!     |    1: ignis::memory::resources::Buffer::drop
//!     |    2: my_app::renderer::cleanup
//!     |    3: my_app::main
//!     |
//!     |  ... content in bright white ...
//!     |
//!     = note: additional context
//!     = help: actionable suggestion
//!  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//! ```
//!
//! # Severity Levels
//!
//! - **Error** (red, 🔴): Corruption, use-after-free, invalid state,
//!   double-free, hang detection. The application has a bug that must
//!   be fixed. Errors include environment context and backtrace
//!   automatically.
//! - **Warning** (yellow, 🟡): Leaks, budget pressure, suboptimal
//!   barriers, quarantine re-verification failures. The application
//!   works but is degraded or at risk.
//! - **Info** (cyan, 🔵): Statistics, reports, non-actionable context.
//!   Informational output for diagnostics and performance analysis.
//!
//! # Diagnostic Codes
//!
//! Every diagnostic has a unique code (e.g., `IGN-H001`) that can be
//! searched in documentation or issue trackers. Codes are organized
//! by subsystem:
//!
//! | Prefix | Subsystem |
//! |--------|-----------|
//! | `IGN-H` | Hardened allocator (guard corruption, double-free) |
//! | `IGN-S` | Slab allocator and command state validator |
//! | `IGN-A` | Resource aliasing detector |
//! | `IGN-O` | Barrier optimizer |
//! | `IGN-D` | Descriptor set auditor |
//! | `IGN-P` | Pipeline compatibility checker |
//! | `IGN-T` | Thread safety auditor |
//! | `IGN-W` | Hang detector (watchdog) |
//! | `IGN-M` | Memory budget monitor |
//! | `IGN-L` | Object lifetime tracker |
//! | `IGN-J` | Submission journal |
//! | `IGN-Q` | Deletion queue |
//! | `IGN-SUM` | Session summary |
//!
//! # Session Tracking
//!
//! The module maintains global atomic counters of all emitted
//! diagnostics. When the `Ignis` context is dropped (or on demand
//! via [`session_summary`]), a summary report is produced showing
//! total errors, warnings, and per-code breakdown. Repeated
//! diagnostics are annotated with their occurrence count to help
//! identify the noisiest issues.
//!
//! # Color Support
//!
//! Respects the `NO_COLOR` environment variable (<https://no-color.org/>).
//! When `NO_COLOR` is set, all ANSI escape codes are suppressed and
//! raw text is emitted. Emoji indicators (🔴, 🟡, 🔵) are always
//! present regardless of color support for accessibility.
//!
//! # Vulkan Spec References
//!
//! Each diagnostic code can be mapped to a specific section of the
//! Vulkan specification via [`spec_reference`]. This is printed in
//! the diagnostic header so the user can immediately look up the
//! relevant rule.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Width of the diagnostic frame borders in visible characters.
const DIAG_WIDTH: usize = 76;

// ─────────────────────────────────────────────────────────────────────────────
// Terminal styling
// ─────────────────────────────────────────────────────────────────────────────

/// ANSI terminal style controller.
///
/// Respects the `NO_COLOR` environment variable (<https://no-color.org/>).
/// When disabled, all methods return the input string unchanged, ensuring
/// that log files and non-terminal outputs remain readable.
///
/// # Thread Safety
///
/// `Style::detect()` reads `NO_COLOR` from the environment on each call.
/// For hot paths, cache the result:
///
/// ```rust,ignore
/// let s = Style::detect();
/// // ... use `s` for all formatting in this scope ...
/// ```
pub(crate) struct Style {
    /// Whether ANSI color codes should be emitted.
    pub on: bool,
}

impl Style {
    /// Detect whether color output is enabled by checking the `NO_COLOR`
    /// environment variable. Returns a `Style` with `on = true` if
    /// `NO_COLOR` is not set, `on = false` otherwise.
    pub fn detect() -> Self {
        Self {
            on: std::env::var_os("NO_COLOR").is_none(),
        }
    }

    /// Apply an ANSI escape code to the given text. When color is
    /// disabled, returns the text unchanged.
    fn esc(&self, code: &str, text: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    // ── Bold + color ──────────────────────────────────────────────────────

    /// Bold red text. Used for error severity labels, corrupted byte
    /// values, and critical violation markers.
    pub fn bold_red(&self, t: &str) -> String {
        self.esc("1;31", t)
    }

    /// Bold yellow text. Used for warning severity labels and threshold
    /// markers in budget reports.
    pub fn bold_yellow(&self, t: &str) -> String {
        self.esc("1;33", t)
    }

    /// Bold green text. Used for "OK" status markers, expected values
    /// in hex diffs, and passing diagnostic checks.
    pub fn bold_green(&self, t: &str) -> String {
        self.esc("1;32", t)
    }

    /// Bold cyan text. Used for info severity labels, resource names
    /// in diagnostic output, and section headers.
    pub fn bold_cyan(&self, t: &str) -> String {
        self.esc("1;36", t)
    }

    /// Bold magenta text. Available for custom extensions and
    /// user-facing highlight needs.
    #[allow(dead_code)]
    pub fn bold_magenta(&self, t: &str) -> String {
        self.esc("1;35", t)
    }

    /// Bold text without a specific color. Used for structural emphasis
    /// such as diagnostic codes, key labels, and section titles.
    pub fn bold(&self, t: &str) -> String {
        self.esc("1", t)
    }

    // ── Regular color ─────────────────────────────────────────────────────

    /// Blue text. Used for pipe characters (`|`) in diagnostic frames
    /// and location arrows (`-->`).
    pub fn blue(&self, t: &str) -> String {
        self.esc("34", t)
    }

    /// Red text (non-bold). Used for hex dump actual values that differ
    /// from expected, and "actual:" labels.
    pub fn red(&self, t: &str) -> String {
        self.esc("31", t)
    }

    /// Green text (non-bold). Used for "expect:" labels in hex diffs
    /// and utilization bars below 50%.
    pub fn green(&self, t: &str) -> String {
        self.esc("32", t)
    }

    /// Yellow text (non-bold). Used for utilization bars between 50-80%
    /// and medium-severity indicators.
    pub fn yellow(&self, t: &str) -> String {
        self.esc("33", t)
    }

    /// Cyan text (non-bold). Available for custom extensions.
    #[allow(dead_code)]
    pub fn cyan(&self, t: &str) -> String {
        self.esc("36", t)
    }

    /// Magenta text (non-bold). Available for custom extensions.
    #[allow(dead_code)]
    pub fn magenta(&self, t: &str) -> String {
        self.esc("35", t)
    }

    // ── Bright / high-intensity ───────────────────────────────────────────

    /// Bright white (high-intensity). Used for diagnostic content to
    /// make it stand out against the default terminal foreground.
    /// This is the primary content color for all diagnostic messages.
    pub fn bright_white(&self, t: &str) -> String {
        self.esc("1;37", t)
    }

    /// Bright red (high-intensity, not bold). Used for high-visibility
    /// error indicators where bold-red would be too heavy.
    #[allow(dead_code)]
    pub fn bright_red(&self, t: &str) -> String {
        self.esc("91", t)
    }

    /// Bright yellow (high-intensity, not bold). Used for secondary
    /// warning indicators.
    #[allow(dead_code)]
    pub fn bright_yellow(&self, t: &str) -> String {
        self.esc("93", t)
    }

    /// Bright green (high-intensity, not bold). Used for emphasis on
    /// successful outcomes.
    #[allow(dead_code)]
    pub fn bright_green(&self, t: &str) -> String {
        self.esc("92", t)
    }

    /// Bright cyan (high-intensity, not bold). Used for emphasis on
    /// informational content.
    #[allow(dead_code)]
    pub fn bright_cyan(&self, t: &str) -> String {
        self.esc("96", t)
    }

    // ── Decorations ───────────────────────────────────────────────────────

    /// Dimmed text. Used for timestamps, metadata, separator lines,
    /// and secondary context that should not compete with the main
    /// diagnostic content for visual attention.
    pub fn dim(&self, t: &str) -> String {
        self.esc("2", t)
    }

    /// Underlined text. Used for source file locations and clickable
    /// references in terminal emulators that support it.
    pub fn underline(&self, t: &str) -> String {
        self.esc("4", t)
    }

    /// Italic text. Available for custom extensions and emphasis
    /// where bold would be too strong.
    #[allow(dead_code)]
    pub fn italic(&self, t: &str) -> String {
        self.esc("3", t)
    }

    /// Strikethrough text. Available for showing deprecated or
    /// replaced values in comparison displays.
    #[allow(dead_code)]
    pub fn strikethrough(&self, t: &str) -> String {
        self.esc("9", t)
    }

    // ── Background colors ─────────────────────────────────────────────────

    /// Red background with bold white text. Used for critical error
    /// severity badges that must be immediately visible in any
    /// terminal environment.
    #[allow(dead_code)]
    pub fn bg_red(&self, t: &str) -> String {
        self.esc("41;1;37", t)
    }

    /// Yellow background with bold black text. Used for warning
    /// severity badges.
    #[allow(dead_code)]
    pub fn bg_yellow(&self, t: &str) -> String {
        self.esc("43;1;30", t)
    }

    /// Green background with bold white text. Used for info severity
    /// badges and "all clear" indicators.
    #[allow(dead_code)]
    pub fn bg_green(&self, t: &str) -> String {
        self.esc("42;1;37", t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Severity
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level for a diagnostic.
///
/// Determines the color scheme, border style, emoji indicator, and
/// what automatic context is included (errors get environment blocks
/// and backtraces by default).
pub(crate) enum Severity {
    /// Corruption, use-after-free, invalid state. Must be fixed.
    Error,
    /// Leaks, budget pressure, suboptimal usage. Should be fixed.
    Warning,
    /// Statistics, reports, informational context. No action needed.
    Info,
}

impl Severity {
    /// Format the severity as a colored label: "error", "warning", or "info".
    pub fn label(&self, s: &Style) -> String {
        match self {
            Severity::Error => s.bold_red("error"),
            Severity::Warning => s.bold_yellow("warning"),
            Severity::Info => s.bold_cyan("info"),
        }
    }

    /// Return a Unicode emoji indicator for the severity level.
    /// Always present regardless of color support for accessibility.
    pub fn icon(&self) -> &'static str {
        match self {
            Severity::Error => "🔴",
            Severity::Warning => "🟡",
            Severity::Info => "🔵",
        }
    }

    /// Return the Unicode heavy horizontal line in the severity's color.
    /// Used for top and bottom borders of diagnostic frames.
    fn border(&self, s: &Style, width: usize) -> String {
        let line = "━".repeat(width);
        match self {
            Severity::Error => s.bold_red(&line),
            Severity::Warning => s.bold_yellow(&line),
            Severity::Info => s.bold_cyan(&line),
        }
    }

    /// Format a fixed-width severity badge with colored background.
    /// Suitable for summary tables where badges need to be aligned.
    pub fn badge(&self, s: &Style) -> String {
        match self {
            Severity::Error => s.bg_red("  ERROR  "),
            Severity::Warning => s.bg_yellow(" WARNING "),
            Severity::Info => s.bg_green("  INFO   "),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global diagnostic context
// ─────────────────────────────────────────────────────────────────────────────

/// Application start time, initialized on first diagnostic context creation.
static APP_START: OnceLock<Instant> = OnceLock::new();

/// Global device/environment context, populated once from `SharedState`.
static DIAG_CTX: OnceLock<DiagnosticContext> = OnceLock::new();

/// Global diagnostic emission counters, created on first diagnostic.
static COUNTERS: OnceLock<DiagnosticCounters> = OnceLock::new();

/// Snapshot of the GPU environment captured once during initialization.
///
/// Included in diagnostic output so that every error report contains
/// enough context to reproduce the issue without asking the user
/// "what GPU do you have?" or "which driver version?".
pub(crate) struct DiagnosticContext {
    /// Human-readable GPU device name (e.g., "NVIDIA GeForce RTX 4090").
    pub gpu_name: String,
    /// Formatted driver version string (e.g., "546.33.0").
    pub driver_version: String,
    /// Formatted Vulkan API version supported by the device (e.g., "1.3.270").
    pub api_version: String,
    /// Number of memory heaps available on the device.
    pub heap_count: u32,
    /// List of ignis feature flags that are enabled at compile time.
    pub features: Vec<&'static str>,
    /// Operating system identifier (e.g., "windows", "linux", "macos").
    pub os: String,
    /// Process ID at the time of context creation.
    pub pid: u32,
}

impl DiagnosticContext {
    /// Build the diagnostic context from the shared Vulkan state.
    ///
    /// Extracts GPU name, driver version, API version, and memory
    /// heap count from the physical device properties. Also captures
    /// the set of enabled ignis features at compile time.
    pub fn from_shared(shared: &crate::device::SharedState) -> Self {
        let props = &shared.device_properties;

        let gpu_name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
        }
        .to_string_lossy()
        .into_owned();

        let api_version = format!(
            "{}.{}.{}",
            ash::vk::api_version_major(props.api_version),
            ash::vk::api_version_minor(props.api_version),
            ash::vk::api_version_patch(props.api_version),
        );

        let driver_version = format!(
            "{}.{}.{}",
            ash::vk::api_version_major(props.driver_version),
            ash::vk::api_version_minor(props.driver_version),
            ash::vk::api_version_patch(props.driver_version),
        );

        let mut features = Vec::new();
        #[cfg(feature = "tracking")]
        features.push("tracking");
        #[cfg(feature = "debug-tools")]
        features.push("debug-tools");
        #[cfg(feature = "slab-allocator")]
        features.push("slab-allocator");
        #[cfg(feature = "descriptors")]
        features.push("descriptors");
        #[cfg(feature = "swapchain")]
        features.push("swapchain");
        #[cfg(feature = "interop")]
        features.push("interop");

        Self {
            gpu_name,
            driver_version,
            api_version,
            heap_count: shared.memory_properties.memory_heap_count,
            features,
            os: std::env::consts::OS.to_string(),
            pid: std::process::id(),
        }
    }
}

/// Initialize the global diagnostic context from shared Vulkan state.
///
/// Should be called once during `Ignis::managed` or `Ignis::external`
/// creation. Safe to call multiple times — only the first call takes
/// effect (subsequent calls are silently ignored).
///
/// Also initializes the application start time if not already set.
pub(crate) fn init_diagnostic_context(shared: &crate::device::SharedState) {
    let _ = APP_START.get_or_init(Instant::now);
    let _ = DIAG_CTX.get_or_init(|| DiagnosticContext::from_shared(shared));
}

/// Get formatted application uptime since the first diagnostic context
/// was initialized. Returns "unknown" if the context was never initialized.
pub(crate) fn app_uptime() -> String {
    APP_START
        .get()
        .map(|start| format_duration(start.elapsed()))
        .unwrap_or_else(|| "unknown".into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic session counters
// ─────────────────────────────────────────────────────────────────────────────

/// Global atomic counters for all diagnostic emissions during the process
/// lifetime. Used to produce the session summary at shutdown.
pub(crate) struct DiagnosticCounters {
    /// Total number of error-level diagnostics emitted.
    pub errors: AtomicU64,
    /// Total number of warning-level diagnostics emitted.
    pub warnings: AtomicU64,
    /// Total number of info-level diagnostics emitted.
    pub infos: AtomicU64,
    /// Per-code occurrence counts. Key is the diagnostic code string
    /// (e.g., "IGN-H001"), value is how many times it has been emitted.
    pub seen_codes: Mutex<HashMap<String, u64>>,
}

/// Get or initialize the global diagnostic counters.
pub(crate) fn counters() -> &'static DiagnosticCounters {
    COUNTERS.get_or_init(|| DiagnosticCounters {
        errors: AtomicU64::new(0),
        warnings: AtomicU64::new(0),
        infos: AtomicU64::new(0),
        seen_codes: Mutex::new(HashMap::new()),
    })
}

/// Record that a diagnostic with the given severity and code was emitted.
///
/// Increments the appropriate severity counter and the per-code occurrence
/// counter. Returns the occurrence count for this specific code (1 on first
/// emission, 2+ on subsequent emissions).
///
/// This is called automatically by [`write_header`] and
/// [`write_full_diagnostic`].
pub(crate) fn record_diagnostic(sev: &Severity, code: &str) -> u64 {
    let c = counters();
    match sev {
        Severity::Error => {
            c.errors.fetch_add(1, Ordering::Relaxed);
        }
        Severity::Warning => {
            c.warnings.fetch_add(1, Ordering::Relaxed);
        }
        Severity::Info => {
            c.infos.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut seen = c.seen_codes.lock().unwrap();
    let count = seen.entry(code.to_string()).or_insert(0);
    *count += 1;
    *count
}

/// Generate a final diagnostic session summary.
///
/// Produces a framed report showing total errors, warnings, and infos
/// emitted during the session, with a per-code breakdown sorted by
/// frequency (most frequent first).
///
/// Returns an empty string if no diagnostics were emitted.
///
/// # When to Call
///
/// - Automatically called by `Ignis::drop` if the counters feature is active.
/// - Can be called manually at any point for an interim report.
pub(crate) fn session_summary() -> String {
    let c = counters();
    let errors = c.errors.load(Ordering::Relaxed);
    let warnings = c.warnings.load(Ordering::Relaxed);
    let infos = c.infos.load(Ordering::Relaxed);
    let total = errors + warnings + infos;

    if total == 0 {
        return String::new();
    }

    let s = Style::detect();
    let mut o = String::with_capacity(2048);

    let sev = if errors > 0 {
        Severity::Error
    } else if warnings > 0 {
        Severity::Warning
    } else {
        Severity::Info
    };

    write_header(
        &mut o,
        &s,
        &sev,
        "IGN-SUM",
        &format!("diagnostic session summary: {} total emission(s)", total),
    );
    write_pipe_empty(&mut o, &s);

    // ── Severity totals with colored counts ──

    let err_display = if errors > 0 {
        format!(
            "{} {}",
            s.bold_red(&format!("{errors}")),
            s.bold_red("error(s)")
        )
    } else {
        s.green("0 errors").to_string()
    };

    let warn_display = if warnings > 0 {
        format!(
            "{} {}",
            s.bold_yellow(&format!("{warnings}")),
            s.bold_yellow("warning(s)")
        )
    } else {
        s.green("0 warnings").to_string()
    };

    let info_display = format!("{infos} info(s)");

    write_pipe_raw(
        &mut o,
        &s,
        &format!("  {err_display}  │  {warn_display}  │  {info_display}"),
    );
    write_pipe_empty(&mut o, &s);

    // ── Utilization bar showing error/warning/info ratio ──

    if total > 0 {
        let err_frac = errors as f64 / total as f64;
        let warn_frac = warnings as f64 / total as f64;

        let bar_width:usize = 50;
        let err_chars = (err_frac * bar_width as f64).round() as usize;
        let warn_chars = (warn_frac * bar_width as f64).round() as usize;
        let info_chars = bar_width.saturating_sub(err_chars).saturating_sub(warn_chars);

        let bar = format!(
            "[{}{}{}]",
            s.bold_red(&"█".repeat(err_chars)),
            s.bold_yellow(&"█".repeat(warn_chars)),
            s.bold_cyan(&"█".repeat(info_chars)),
        );
        write_pipe_raw(&mut o, &s, &format!("  {bar}"));
        write_pipe_raw(
            &mut o,
            &s,
            &s.dim(&format!(
                "   {}=error {}=warning {}=info",
                "█".repeat(2),
                "█".repeat(2),
                "█".repeat(2),
            )),
        );
        write_pipe_empty(&mut o, &s);
    }

    // ── Per-code breakdown ──

    let seen = c.seen_codes.lock().unwrap();
    if !seen.is_empty() {
        write_section(&mut o, &s, "Breakdown by Diagnostic Code");

        let mut codes: Vec<(&String, &u64)> = seen.iter().collect();
        codes.sort_by(|a, b| b.1.cmp(a.1)); // Most frequent first.

        for (code, count) in &codes {
            let severity_hint = code_to_severity_hint(code, &s);
            write_pipe_raw(
                &mut o,
                &s,
                &format!(
                    "  {severity_hint}  {:<14}  ×{}",
                    s.bright_white(code),
                    s.bold(&count.to_string()),
                ),
            );
        }
        write_pipe_empty(&mut o, &s);
    }

    // ── Session metadata ──

    write_separator(&mut o, &s);
    write_kv(&mut o, &s, "Session duration", &app_uptime());

    if let Some(ctx) = DIAG_CTX.get() {
        write_kv(&mut o, &s, "GPU", &ctx.gpu_name);
        write_kv(&mut o, &s, "Vulkan API", &ctx.api_version);
    }

    // ── Actionable advice if errors present ──

    if errors > 0 {
        write_pipe_empty(&mut o, &s);
        write_help(
            &mut o,
            &s,
            "fix all error-level diagnostics before shipping\n\
             error diagnostics indicate bugs that cause undefined behavior,\n\
             crashes, or visual corruption on end-user hardware",
        );
    }

    if warnings > 5 {
        write_warn(
            &mut o,
            &s,
            &format!(
                "{warnings} warnings emitted — consider reviewing the most frequent\n\
                 codes above and addressing root causes to reduce diagnostic noise"
            ),
        );
    }

    write_diagnostic_end(&mut o, &s, &sev);

    o
}

/// Map a diagnostic code prefix to a severity hint label for the
/// session summary breakdown table.
fn code_to_severity_hint(code: &str, s: &Style) -> String {
    // Explicit info/warning codes that would otherwise match error prefixes.
    match code {
        "IGN-H004" | "IGN-H005" => return s.bold_yellow("WARN "),
        "IGN-H006" => return s.bold_cyan("INFO "),
        "IGN-S012" => return s.bold_cyan("INFO "),
        "IGN-J002" => return s.bold_cyan("INFO "),
        "IGN-SUM"  => return s.bold_cyan("INFO "),
        _ => {}
    }

    // Prefix-based classification for remaining codes.
    if code.starts_with("IGN-H")       // H001, H002, H003
        || code.starts_with("IGN-S001") // cmd_state invalid command
        || code.starts_with("IGN-S002") // cmd_state missing binding
        || code.starts_with("IGN-S010") // slab double-free
        || code.starts_with("IGN-S011") // slab overflow
        || code.starts_with("IGN-A")    // aliasing
        || code.starts_with("IGN-D")    // descriptor audit
        || code.starts_with("IGN-P")    // pipeline audit
        || code.starts_with("IGN-T")    // thread audit
        || code.starts_with("IGN-W")    // hang detector
        || code.starts_with("IGN-J001") // journal error dump
    {
        s.bold_red(" ERR ")
    } else if code.starts_with("IGN-M")    // budget monitor
        || code.starts_with("IGN-O")       // barrier optimizer
        || code.starts_with("IGN-L")       // lifetime leaks
        || code.starts_with("IGN-Q")       // deletion queue
    {
        s.bold_yellow("WARN ")
    } else {
        s.bold_cyan("INFO ")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan spec references
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the relevant Vulkan specification section for a given
/// diagnostic code, if known.
///
/// These references are printed in the diagnostic header so the user
/// can immediately look up the rule that was violated without having
/// to search the spec manually.
pub(crate) fn spec_reference(code: &str) -> Option<&'static str> {
    match code {
        "IGN-H001" | "IGN-H002" => Some("§11.6 Resource Memory Association"),
        "IGN-H003" => Some("§11.6.13 Freeing Device Memory"),
        "IGN-H004" => Some("§11.6 Resource Memory Association (quarantine verification)"),
        "IGN-H005" => Some("§11.6 Device Memory Lifecycle"),
        "IGN-S001" => Some("§6.1 Command Buffer Lifecycle"),
        "IGN-S002" => Some("§10.5.1 Binding Pipeline Objects"),
        "IGN-S010" => Some("§11.6.13 Freeing Device Memory (double free)"),
        "IGN-S011" => Some("§11.6 Resource Memory Association (overflow)"),
        "IGN-S012" => None, // Stats report, no spec violation.
        "IGN-T001" => Some("§3.3.1 External Synchronization (Command Pools)"),
        "IGN-A001" => Some("§7.1.3 Pipeline Barriers / §7.6.1 Execution Dependencies"),
        "IGN-O001" => Some("§7.1 Synchronization / §7.6.2 Memory Dependencies"),
        "IGN-D001" => Some("§14.2.3 Descriptor Set Updates"),
        "IGN-P001" => Some("§14.2.2 Pipeline Layout Compatibility"),
        "IGN-W001" => Some("§5.2 Queue Submission / §5.5 Device Lost"),
        "IGN-M001" => Some("§11.6 VK_EXT_memory_budget"),
        "IGN-L001" => Some("§3.3.3 Object Lifetime"),
        "IGN-J001" => Some("§5.5 Lost Device"),
        "IGN-J002" => None, // Info dump, no spec violation.
        "IGN-Q001" => Some("§3.3.3 Object Lifetime (deferred destruction)"),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Context helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Get a wall-clock timestamp string formatted as `HH:MM:SS.mmm`.
///
/// Uses `SystemTime` for a human-readable timestamp. Not monotonic —
/// for duration measurements use [`Instant`].
fn wall_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let dur = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = dur.as_secs();
    let h = (total_secs / 3600) % 24;
    let m = (total_secs / 60) % 60;
    let sec = total_secs % 60;
    let ms = dur.subsec_millis();
    format!("{h:02}:{m:02}:{sec:02}.{ms:03}")
}

/// Get the current thread name, or `"<unnamed>"` if the thread has no name.
///
/// Used in diagnostic context lines to identify which thread produced
/// the diagnostic, which is critical for debugging multi-threaded
/// command recording issues.
pub(crate) fn current_thread_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core write primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Build a diagnostic header with top border, severity icon, code, message,
/// and automatic context (timestamp, thread, pid, uptime).
///
/// This is the opening of every diagnostic block. Pair with
/// [`write_diagnostic_end`] to close the frame.
///
/// Also records the diagnostic emission in the global counters via
/// [`record_diagnostic`].
pub(crate) fn write_header(o: &mut String, s: &Style, sev: &Severity, code: &str, msg: &str) {
    let _count = record_diagnostic(sev, code);

    // Top border in severity color.
    let _ = writeln!(o, "\n {}", sev.border(s, DIAG_WIDTH));

    // Header line: emoji + severity[CODE]: message in bright white.
    let _ = writeln!(
        o,
        " {} {}{}: {}",
        sev.icon(),
        sev.label(s),
        s.bold(&format!("[{code}]")),
        s.bright_white(msg)
    );

    // Context line: timestamp, thread, PID, uptime.
    let ts = wall_timestamp();
    let thread = current_thread_name();
    let pid = std::process::id();
    let uptime = app_uptime();
    let _ = writeln!(
        o,
        " {}",
        s.dim(&format!(
            "at {ts} │ thread=\"{thread}\" │ pid={pid} │ uptime={uptime}"
        ))
    );

    // Vulkan spec reference if available.
    if let Some(section) = spec_reference(code) {
        let _ = writeln!(o, " {}", s.dim(&format!("spec: Vulkan {section}")));
    }
}

/// Enhanced diagnostic header that includes environment context, repeat
/// tracking notice, backtrace, and Vulkan spec reference.
///
/// Use this instead of [`write_header`] when the diagnostic needs
/// maximum context (e.g., memory corruption, hang detection, device lost).
///
/// # Arguments
///
/// * `o` - Output string buffer to append to.
/// * `s` - Terminal style configuration.
/// * `sev` - Severity level (determines colors and automatic inclusions).
/// * `code` - Diagnostic code (e.g., "IGN-H001").
/// * `msg` - Human-readable summary message.
/// * `include_env` - Whether to include the GPU environment block.
///   Automatically included for errors and first occurrences.
/// * `include_backtrace` - Whether to capture and include a backtrace.
///   Automatically included for errors.
pub(crate) fn write_full_diagnostic(
    o: &mut String,
    s: &Style,
    sev: &Severity,
    code: &str,
    msg: &str,
    include_env: bool,
    include_backtrace: bool,
) {
    let count = record_diagnostic(sev, code);

    // Critical banner for errors.
    if matches!(sev, Severity::Error) {
        write_critical_banner(o, s, "IGNIS DIAGNOSTIC ERROR");
    }

    // Top border in severity color.
    let _ = writeln!(o, "\n {}", sev.border(s, DIAG_WIDTH));

    // Header line: emoji + severity[CODE]: message in bright white.
    let _ = writeln!(
        o,
        " {} {}{}: {}",
        sev.icon(),
        sev.label(s),
        s.bold(&format!("[{code}]")),
        s.bright_white(msg)
    );

    // Context line: timestamp, thread, PID, uptime.
    let ts = wall_timestamp();
    let thread = current_thread_name();
    let pid = std::process::id();
    let uptime = app_uptime();
    let _ = writeln!(
        o,
        " {}",
        s.dim(&format!(
            "at {ts} │ thread=\"{thread}\" │ pid={pid} │ uptime={uptime}"
        ))
    );

    // Vulkan spec reference if available.
    if let Some(section) = spec_reference(code) {
        let _ = writeln!(o, " {}", s.dim(&format!("spec: Vulkan {section}")));
    }

    // Repeat notice for diagnostics seen more than once.
    write_repeat_notice(o, s, code, count);

    // Environment block: included for errors, or on first occurrence.
    if include_env && (matches!(sev, Severity::Error) || count == 1) {
        write_environment_block(o, s);
    }

    // Backtrace: included for errors.
    if include_backtrace && matches!(sev, Severity::Error) {
        write_backtrace(o, s, 12);
    }
}

/// Close a diagnostic block with a bottom border in severity color.
///
/// Always call this after [`write_header`] or [`write_full_diagnostic`]
/// to produce a visually complete framed diagnostic.
pub(crate) fn write_diagnostic_end(o: &mut String, s: &Style, sev: &Severity) {
    let _ = writeln!(o, " {}", sev.border(s, DIAG_WIDTH));
}

/// Build a location arrow: `  --> location_text`
///
/// Used immediately after the header to show which resource, object, or
/// code location the diagnostic pertains to.
pub(crate) fn write_location(o: &mut String, s: &Style, location: &str) {
    let _ = writeln!(o, "  {} {location}", s.blue("-->"));
}

/// Write an empty pipe line: `   |`
///
/// Used for visual spacing within diagnostic blocks without breaking
/// the framed appearance.
pub(crate) fn write_pipe_empty(o: &mut String, s: &Style) {
    let _ = writeln!(o, "   {}", s.blue("|"));
}

/// Write a pipe line with content in bright white: `   |  content`
///
/// The primary content writing primitive. Text is automatically
/// wrapped in `bright_white` for high contrast against the terminal
/// background.
pub(crate) fn write_pipe(o: &mut String, s: &Style, text: &str) {
    let _ = writeln!(o, "   {}  {}", s.blue("|"), s.bright_white(text));
}

/// Write a pipe line with raw (pre-colored) content: `   |  content`
///
/// Use when the content already contains ANSI codes and should not
/// be wrapped in `bright_white`. Suitable for hex diffs, colored
/// markers, and pre-formatted table rows.
pub(crate) fn write_pipe_raw(o: &mut String, s: &Style, text: &str) {
    let _ = writeln!(o, "   {}  {text}", s.blue("|"));
}

/// Write a thin separator line within a diagnostic block.
///
/// Lighter than the top/bottom borders, useful for grouping sections
/// within a single diagnostic. Rendered as a dimmed horizontal line
/// inside the pipe frame.
pub(crate) fn write_separator(o: &mut String, s: &Style) {
    let line = "─".repeat(DIAG_WIDTH - 4);
    let _ = writeln!(o, "   {}  {}", s.blue("|"), s.dim(&line));
}

/// Write a section header within a diagnostic block.
///
/// Renders as a bold bright-white label with a thin underline,
/// preceded by an empty pipe line for spacing. Use to introduce
/// major sections within a diagnostic (e.g., "Environment",
/// "Backtrace", "Corruption Analysis").
pub(crate) fn write_section(o: &mut String, s: &Style, title: &str) {
    write_pipe_empty(o, s);
    write_pipe_raw(o, s, &s.bold(&s.bright_white(&format!("── {title} ──"))));
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment block
// ─────────────────────────────────────────────────────────────────────────────

/// Write the full GPU environment context block.
///
/// Includes GPU name, driver version, Vulkan API version, operating
/// system, process ID, application uptime, memory heap count, and
/// enabled feature flags.
///
/// Only produces output if the diagnostic context has been initialized
/// via [`init_diagnostic_context`]. If not initialized, this is a no-op.
pub(crate) fn write_environment_block(o: &mut String, s: &Style) {
    if let Some(ctx) = DIAG_CTX.get() {
        write_separator(o, s);
        write_section(o, s, "Environment");
        write_kv(o, s, "GPU", &ctx.gpu_name);
        write_kv(o, s, "Driver", &ctx.driver_version);
        write_kv(o, s, "Vulkan API", &ctx.api_version);
        write_kv(o, s, "OS", &ctx.os);
        write_kv(o, s, "PID", &ctx.pid.to_string());
        write_kv(o, s, "Uptime", &app_uptime());
        write_kv(o, s, "Memory heaps", &ctx.heap_count.to_string());
        if !ctx.features.is_empty() {
            write_kv(o, s, "Features", &ctx.features.join(", "));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backtrace capture
// ─────────────────────────────────────────────────────────────────────────────

/// Capture the current backtrace and format it compactly, filtering out
/// internal Rust runtime frames.
///
/// Frames belonging to `ignis::` are highlighted in bold cyan for easy
/// identification. Standard library and runtime frames are dimmed or
/// skipped entirely.
///
/// # Arguments
///
/// * `o` - Output string buffer.
/// * `s` - Terminal style configuration.
/// * `max_frames` - Maximum number of frames to include. A note is
///   appended if more frames exist beyond this limit.
pub(crate) fn write_backtrace(o: &mut String, s: &Style, max_frames: usize) {
    let bt = std::backtrace::Backtrace::force_capture();
    let bt_str = bt.to_string();

    let frames: Vec<&str> = bt_str
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with("at ")           // "at /rustc/..." location lines
                && !trimmed.contains("std::rt::")
                && !trimmed.contains("std::panic")
                && !trimmed.contains("std::backtrace")   // NEW: filter backtrace internals
                && !trimmed.contains("backtrace::backtrace") // NEW: backtrace-rs internals
                && !trimmed.contains("core::ops::function")
                && !trimmed.contains("__rust_begin_short_backtrace")
                && !trimmed.contains("__rust_end_short_backtrace")
                && !trimmed.contains("std::sys::")
                && !trimmed.contains("lang_start")
                && !trimmed.contains("BaseThreadInitThunk") // NEW: Windows CRT
                && !trimmed.contains("RtlUserThreadStart")  // NEW: Windows CRT
                && !trimmed.contains("__scrt_common_main")  // NEW: Windows CRT
        })
        .take(max_frames)
        .collect();

    if frames.is_empty() {
        return;
    }

    write_separator(o, s);
    write_section(o, s, "Backtrace");

    for (i, frame) in frames.iter().enumerate() {
        let trimmed = frame.trim();
        // Strip the original frame number ("5: ") to avoid double-numbering.
        let cleaned = if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit()) {
            // Handle multi-digit: "12: foo" -> strip digits then ": "
            let after_digits = trimmed.trim_start_matches(|c: char| c.is_ascii_digit());
            if let Some(rest) = after_digits.strip_prefix(": ") {
                rest
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        let colored = if cleaned.contains("ignis::") {
            s.bold_cyan(cleaned)
        } else {
            s.dim(cleaned)
        };
        write_pipe_raw(o, s, &format!("  {:>3}: {colored}", i));
    }

    let total_meaningful = bt_str
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("at "))
        .count();
    if total_meaningful > max_frames {
        write_pipe_raw(
            o,
            s,
            &s.dim(&format!(
                "       ... {} more frames (set RUST_BACKTRACE=full for all)",
                total_meaningful - max_frames
            )),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Repeat notice
// ─────────────────────────────────────────────────────────────────────────────

/// Write a notice if this diagnostic code has been emitted more than once.
///
/// On the second occurrence, shows a simple count. On 10+ occurrences,
/// adds a stronger suggestion to fix the root cause to reduce noise.
///
/// Does nothing on the first occurrence.
pub(crate) fn write_repeat_notice(o: &mut String, s: &Style, code: &str, count: u64) {
    if count > 1 {
        write_pipe_raw(
            o,
            s,
            &s.dim(&format!(
                "⚠ This diagnostic ({code}) has been emitted {count} time(s) this session"
            )),
        );
        if count >= 10 {
            write_pipe_raw(
                o,
                s,
                &s.bold_yellow(
                    "  Consider fixing the root cause to reduce diagnostic noise",
                ),
            );
        }
        write_pipe_empty(o, s);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Critical banner
// ─────────────────────────────────────────────────────────────────────────────

/// Write an attention-grabbing banner for critical errors.
///
/// Renders a solid block of red `▓` characters with a centered message.
/// This is the most visually intense element in the diagnostic system
/// and should be reserved for true errors (corruption, hang, device lost).
///
/// # Example Output
///
/// ```text
///  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
///  ▓▓                         CRITICAL ERROR DETECTED                           ▓▓
///  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
/// ```
pub(crate) fn write_critical_banner(o: &mut String, s: &Style, msg: &str) {
    let line = "▓".repeat(DIAG_WIDTH);
    let _ = writeln!(o);
    let _ = writeln!(o, " {}", s.bold_red(&line));
    let _ = writeln!(
        o,
        " {}  {}  {}",
        s.bold_red("▓▓"),
        s.bold_red(&pad_center(msg, DIAG_WIDTH - 6, ' ')),
        s.bold_red("▓▓"),
    );
    let _ = writeln!(o, " {}", s.bold_red(&line));
}

// ─────────────────────────────────────────────────────────────────────────────
// Annotation lines (note, help, warn)
// ─────────────────────────────────────────────────────────────────────────────

/// Write a note line in the standard format: `   = note: text`
///
/// Notes provide additional context about the diagnostic, such as
/// statistics, internal state, or explanations of why the condition
/// is problematic.
pub(crate) fn write_note(o: &mut String, s: &Style, text: &str) {
    let label = format!("   {} {}: ", s.bold_cyan("="), s.bold("note"));
    write_labeled(o, &label, text, s);
}

/// Write a help line in the standard format: `   = help: text`
///
/// Help lines provide actionable suggestions for fixing the issue.
/// They should be concrete and specific, not generic advice.
pub(crate) fn write_help(o: &mut String, s: &Style, text: &str) {
    let label = format!("   {} {}: ", s.bold_green("="), s.bold("help"));
    write_labeled(o, &label, text, s);
}

/// Write a warning note in the standard format: `   = warn: text`
///
/// Warning notes highlight aspects of the diagnostic that are not
/// errors themselves but indicate elevated risk.
pub(crate) fn write_warn(o: &mut String, s: &Style, text: &str) {
    let label = format!("   {} {}: ", s.bold_yellow("="), s.bold("warn"));
    write_labeled(o, &label, text, s);
}

/// Internal helper to write a labeled multi-line annotation.
///
/// The first line gets the label prefix; subsequent lines are indented
/// to align with the first line's content.
fn write_labeled(o: &mut String, label: &str, text: &str, s: &Style) {
    let lines: Vec<&str> = text.lines().collect();
    if let Some((first, rest)) = lines.split_first() {
        let _ = writeln!(o, "{label}{}", s.bright_white(first));
        let indent: String = " ".repeat(strip_ansi_len(label));
        for line in rest {
            let _ = writeln!(o, "{indent}{}", s.bright_white(line));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Key-value pairs and tables
// ─────────────────────────────────────────────────────────────────────────────

/// Write a key-value pair on a pipe line.
///
/// `key` is rendered in dim, `value` in bright white. Used for
/// structured metadata output in environment blocks, statistics
/// reports, and configuration dumps.
///
/// # Example Output
///
/// ```text
///    |  GPU: NVIDIA GeForce RTX 4090
/// ```
pub(crate) fn write_kv(o: &mut String, s: &Style, key: &str, value: &str) {
    let formatted = format!("{}: {}", s.dim(key), s.bright_white(value));
    write_pipe_raw(o, s, &formatted);
}

/// Write a table header row. Columns are right-padded to `widths`.
///
/// Produces a bold bright-white header followed by a dimmed underline
/// separator. Use with [`write_table_row`] for aligned tabular data.
///
/// # Arguments
///
/// * `columns` - Slice of `(column_name, width)` tuples. Each column
///   name is right-padded to the specified width.
pub(crate) fn write_table_header(o: &mut String, s: &Style, columns: &[(&str, usize)]) {
    let mut row = String::new();
    for (name, width) in columns {
        let _ = write!(row, "{:>width$}  ", name, width = width);
    }
    write_pipe_raw(o, s, &s.bold(&s.bright_white(row.trim_end())));
    let underline: String = columns
        .iter()
        .map(|(_, w)| "─".repeat(*w))
        .collect::<Vec<_>>()
        .join("──");
    write_pipe_raw(o, s, &s.dim(&underline));
}

/// Write a table data row. Values are right-padded to match header `widths`.
///
/// Used in conjunction with [`write_table_header`] for aligned tabular data
/// within diagnostic blocks.
pub(crate) fn write_table_row(o: &mut String, s: &Style, cells: &[(&str, usize)]) {
    let mut row = String::new();
    for (value, width) in cells {
        let _ = write!(row, "{:>width$}  ", value, width = width);
    }
    write_pipe(o, s, row.trim_end());
}

// ─────────────────────────────────────────────────────────────────────────────
// Numbered lists
// ─────────────────────────────────────────────────────────────────────────────

/// Write a numbered list item: `[N] content`
///
/// The index is rendered in dim for visual separation from the content.
/// Used for ordered lists of suggestions, leaked objects, or
/// chronological events.
pub(crate) fn write_numbered(o: &mut String, s: &Style, index: usize, text: &str) {
    let num = s.dim(&format!("[{index}]"));
    write_pipe_raw(o, s, &format!("{num} {}", s.bright_white(text)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress / utilization bars
// ─────────────────────────────────────────────────────────────────────────────

/// Render a progress/utilization bar with percentage and optional label.
///
/// The bar changes color based on the fill fraction:
/// - Green (< 50%): healthy utilization
/// - Yellow (50-80%): moderate utilization
/// - Red (80-95%): high utilization
/// - Bold red (≥ 95%): critical utilization
///
/// # Arguments
///
/// * `fraction` - Value between 0.0 and 1.0 representing utilization.
///   Clamped to `[0.0, 1.0]`.
/// * `width` - Total bar width in characters (including brackets).
/// * `label` - Optional label to display after the bar (e.g., "256/512 MiB").
///
/// # Example Output
///
/// ```text
/// [████████████████████████████████░░░░░░░░░░░░░░░░░░] 85.3% (256/512 MiB)
/// ```
pub(crate) fn render_bar(fraction: f64, width: usize, label: Option<&str>, s: &Style) -> String {
    let inner_width = width.saturating_sub(2); // Subtract brackets.
    let filled = (fraction * inner_width as f64).round() as usize;
    let filled = filled.min(inner_width);
    let empty = inner_width - filled;

    let fill_str: String = "█".repeat(filled);
    let empty_str: String = "░".repeat(empty);

    let colored_fill = if fraction >= 0.95 {
        s.bold_red(&fill_str)
    } else if fraction >= 0.80 {
        s.yellow(&fill_str)
    } else if fraction >= 0.50 {
        s.green(&fill_str)
    } else {
        s.bold_green(&fill_str)
    };

    let pct = format!("{:.1}%", fraction * 100.0);

    let bar = format!("[{colored_fill}{}] {}", s.dim(&empty_str), s.bright_white(&pct));

    match label {
        Some(l) => format!("{bar} {}", s.bright_white(l)),
        None => bar,
    }
}

/// Render a compact mini-bar (smaller characters, for table cells).
///
/// Uses `#` and `-` characters instead of block elements for
/// compatibility with terminals that don't support Unicode blocks.
///
/// # Arguments
///
/// * `fraction` - Value between 0.0 and 1.0.
/// * `width` - Number of characters in the bar body (excluding brackets).
pub(crate) fn render_mini_bar(fraction: f64, width: usize, s: &Style) -> String {
    let filled = (fraction * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    let f: String = "#".repeat(filled);
    let e: String = "-".repeat(empty);
    let cf = if fraction >= 0.9 {
        s.bold_red(&f)
    } else if fraction >= 0.7 {
        s.yellow(&f)
    } else {
        s.green(&f)
    };
    format!("[{cf}{}]", s.dim(&e))
}

// ─────────────────────────────────────────────────────────────────────────────
// Hex dumps
// ─────────────────────────────────────────────────────────────────────────────

/// Format bytes as space-separated hex pairs: `"d8 08 96 f8"`.
///
/// Used for compact inline hex display of small byte sequences
/// (guard bands, canary words, prefixes).
pub(crate) fn hex_line(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build diff markers showing `^^` under each differing byte pair.
///
/// Aligned to match the output of [`hex_line`] — each byte takes 2
/// characters plus a space separator.
///
/// # Example
///
/// ```text
/// expect: cd cd cd cd cd cd cd cd
/// actual: cd cd ff cd cd cd cd cd
///               ^^
/// ```
pub(crate) fn diff_markers(expected: &[u8], actual: &[u8]) -> String {
    let len = expected.len().min(actual.len());
    let mut markers = String::with_capacity(len * 3);
    for i in 0..len {
        if expected[i] == actual[i] {
            markers.push(' ');
            markers.push(' ');
        } else {
            markers.push('^');
            markers.push('^');
        }
        if i < len - 1 {
            markers.push(' ');
        }
    }
    markers
}

/// Format a hex dump with offset, hex bytes, and ASCII representation.
///
/// Renders up to `max_rows` rows of 16 bytes each. If the data exceeds
/// the row limit, a "... N more bytes" trailer is appended.
///
/// # Example output
///
/// ```text
/// 00000000: 48 65 6c 6c 6f 20 57 6f 72 6c 64 21 0a 00 00 00  Hello World!....
/// 00000010: ff ff ff ff 00 00 00 00 ab cd ef 01 23 45 67 89  ............#Eg.
/// ```
///
/// # Arguments
///
/// * `data` - Raw byte slice to format.
/// * `base_offset` - Starting offset for the left-hand address column.
///   Pass 0 for data starting at the beginning of a buffer, or the
///   actual offset for data within a larger allocation.
/// * `max_rows` - Maximum number of 16-byte rows to render.
pub(crate) fn hex_dump(data: &[u8], base_offset: usize, max_rows: usize) -> String {
    let mut o = String::new();
    let bytes_per_row = 16;
    let rows = (data.len() + bytes_per_row - 1) / bytes_per_row;
    let rows = rows.min(max_rows);

    for row in 0..rows {
        let start = row * bytes_per_row;
        let end = (start + bytes_per_row).min(data.len());
        let chunk = &data[start..end];

        // Offset column.
        let _ = write!(o, "{:08x}: ", base_offset + start);

        // Hex bytes column.
        for (i, byte) in chunk.iter().enumerate() {
            let _ = write!(o, "{byte:02x}");
            if i < bytes_per_row - 1 {
                o.push(' ');
            }
        }
        // Pad if last row is short.
        for _ in chunk.len()..bytes_per_row {
            o.push_str("   ");
        }

        o.push_str("  ");

        // ASCII representation column.
        for &byte in chunk {
            if byte.is_ascii_graphic() || byte == b' ' {
                o.push(byte as char);
            } else {
                o.push('.');
            }
        }

        if row < rows - 1 {
            o.push('\n');
        }
    }

    if data.len() > rows * bytes_per_row {
        let _ = write!(
            o,
            "\n... {} more bytes",
            data.len() - rows * bytes_per_row
        );
    }

    o
}

// ─────────────────────────────────────────────────────────────────────────────
// Corruption pattern analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Analyze what kind of data overwrote a guard band or zero-prefix.
///
/// Attempts to classify the corrupted bytes into recognizable patterns:
///
/// - All zeros → `memset`/`calloc` or zero-init
/// - `0xCD` fill → MSVC uninitialized heap marker
/// - `0xDD` fill → MSVC freed heap marker
/// - `0xFE`/`0xFD` fill → Guard page or sentinel pattern
/// - ASCII text → String or path data overwrite
/// - Float-like → Vertex/uniform data overwrite
/// - Pointer-like → Pointer table or vtable overwrite
/// - Unrecognized → Shannon entropy analysis
///
/// # Arguments
///
/// * `actual` - The corrupted bytes found in the guard region.
/// * `_expected` - The expected canary pattern (currently unused but
///   reserved for future diff-based analysis).
///
/// # Returns
///
/// A human-readable description of the likely data pattern.
pub(crate) fn analyze_corruption_pattern(actual: &[u8], _expected: &[u8]) -> String {
    if actual.is_empty() {
        return "empty region".into();
    }

    // Check for uniform fill patterns.
    if actual.iter().all(|&b| b == 0) {
        return "all zeros (likely memset/calloc or zero-init overwrite)".into();
    }
    if actual.iter().all(|&b| b == 0xCD) {
        return "0xCD fill (MSVC uninitialized heap pattern — write from uninit buffer)".into();
    }
    if actual.iter().all(|&b| b == 0xDD) {
        return "0xDD fill (MSVC freed heap pattern — write from freed buffer, use-after-free)"
            .into();
    }
    if actual.iter().all(|&b| b == 0xFE || b == 0xFD) {
        return "0xFE/0xFD fill (guard page sentinel — possible guard region overwrite)".into();
    }
    if actual.iter().all(|&b| b == 0xAB) {
        return "0xAB fill (common test pattern — check test/mock code for oversized writes)"
            .into();
    }
    if actual.iter().all(|&b| b == actual[0]) {
        return format!(
            "uniform {:#04x} fill ({} bytes) — likely memset with wrong size or offset",
            actual[0],
            actual.len()
        );
    }

    // Check for ASCII text content.
    let ascii_count = actual
        .iter()
        .filter(|b| b.is_ascii_graphic() || **b == b' ')
        .count();
    if ascii_count > actual.len() * 3 / 4 && actual.len() >= 4 {
        let text: String = actual
            .iter()
            .map(|&b| {
                if b.is_ascii_graphic() || b == b' ' {
                    b as char
                } else {
                    '.'
                }
            })
            .collect();
        return format!(
            "likely ASCII text ({}/{} printable bytes): \"{}\"",
            ascii_count,
            actual.len(),
            text
        );
    }

    // Check for float-like patterns (4-byte aligned, reasonable exponents).
    if actual.len() >= 4 {
        let f = f32::from_ne_bytes([actual[0], actual[1], actual[2], actual[3]]);
        if f.is_finite() && f.abs() < 1e10 && f.abs() > 1e-10 {
            return format!(
                "possible float data (first 4 bytes = {f:.6}, may indicate vertex/uniform overflow)"
            );
        }
    }

    // Check for pointer-like patterns (8-byte aligned, plausible address range).
    if actual.len() >= 8 {
        let ptr = u64::from_ne_bytes(actual[0..8].try_into().unwrap());
        if ptr > 0x1000 && ptr < 0x0000_FFFF_FFFF_FFFF {
            return format!(
                "possible pointer value: {ptr:#018x} (may indicate pointer table overflow)"
            );
        }
    }

    // Count distinct byte values and compute entropy.
    let mut byte_freq = [0u32; 256];
    for &b in actual {
        byte_freq[b as usize] += 1;
    }
    let distinct = byte_freq.iter().filter(|&&c| c > 0).count();
    let entropy = shannon_entropy(actual);

    format!(
        "unrecognized pattern ({distinct} distinct byte values across {} bytes, \
         Shannon entropy = {entropy:.2} bits/byte)",
        actual.len()
    )
}

/// Compute Shannon entropy of a byte sequence in bits per byte.
///
/// Returns a value between 0.0 (perfectly uniform, e.g., all zeros)
/// and 8.0 (perfectly random). Values above 7.0 suggest compressed
/// or encrypted data; values below 3.0 suggest structured data.
pub(crate) fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let len = data.len() as f64;
    freq.iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let p = f as f64 / len;
            -p * p.log2()
        })
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format a Duration compactly: `"142.3us"`, `"3.21ms"`, `"5.02s"`.
///
/// Chooses the most appropriate unit automatically:
/// - Nanoseconds for durations < 1μs
/// - Microseconds for durations < 1ms
/// - Milliseconds for durations < 1s
/// - Seconds for durations ≥ 1s
pub(crate) fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1_000 {
        format!("{nanos}ns")
    } else if nanos < 1_000_000 {
        format!("{:.1}us", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}

/// Format bytes in human-readable units: `B`, `KiB`, `MiB`, `GiB`.
///
/// Uses binary units (1024-based) per IEC convention, matching Vulkan
/// memory reporting conventions.
pub(crate) fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Format a raw Vulkan handle as `0x{value:016x}`.
///
/// Uses lowercase hex with `0x` prefix for consistency with Vulkan
/// validation layer output and RenderDoc.
pub(crate) fn format_handle(raw: u64) -> String {
    format!("{raw:#x}")
}

/// Pad a label centered within `width` characters using `fill`.
///
/// If the label (plus 2 spaces of padding) exceeds `width`, the label
/// is returned as-is with minimal surrounding spaces.
///
/// # Example
///
/// ```text
/// pad_center("hello", 20, '=')  →  "======= hello ======="
/// ```
pub(crate) fn pad_center(label: &str, width: usize, fill: char) -> String {
    if width <= label.len() + 2 {
        return format!(" {label} ");
    }
    let pad = width - label.len() - 2;
    let lp = pad / 2;
    let rp = pad - lp;
    let l: String = std::iter::repeat(fill).take(lp).collect();
    let r: String = std::iter::repeat(fill).take(rp).collect();
    format!("{l} {label} {r}")
}

/// Compute the visible character count of a string, ignoring ANSI escape
/// sequences.
///
/// Walks the string character by character, skipping everything between
/// `\x1b` and `m` (the standard ANSI CSI sequence terminator). Returns
/// the count of non-escape characters.
///
/// Used to compute correct indentation for multi-line annotations where
/// the label contains color codes.
pub(crate) fn strip_ansi_len(s: &str) -> usize {
    let mut len = 0usize;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch == 'm' {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            len += 1;
        }
    }
    len
}

/// Pluralize a word based on count: `"1 buffer"` vs `"3 buffers"`.
///
/// # Arguments
///
/// * `count` - The number to display.
/// * `singular` - The word to use when count == 1.
/// * `plural` - The word to use when count != 1.
pub(crate) fn pluralize(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("{count} {singular}")
    } else {
        format!("{count} {plural}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan type names
// ─────────────────────────────────────────────────────────────────────────────

/// Format a Vulkan object type as a readable string matching the
/// official Vulkan type naming convention (`VkPipeline`, `VkBuffer`, etc.).
///
/// Covers the most common object types used in typical rendering
/// applications. Returns `"VkUnknown"` for unrecognized types.
pub(crate) fn object_type_name(ty: ash::vk::ObjectType) -> &'static str {
    use ash::vk::ObjectType;
    match ty {
        ObjectType::INSTANCE => "VkInstance",
        ObjectType::PHYSICAL_DEVICE => "VkPhysicalDevice",
        ObjectType::DEVICE => "VkDevice",
        ObjectType::QUEUE => "VkQueue",
        ObjectType::SEMAPHORE => "VkSemaphore",
        ObjectType::COMMAND_BUFFER => "VkCommandBuffer",
        ObjectType::FENCE => "VkFence",
        ObjectType::DEVICE_MEMORY => "VkDeviceMemory",
        ObjectType::BUFFER => "VkBuffer",
        ObjectType::IMAGE => "VkImage",
        ObjectType::EVENT => "VkEvent",
        ObjectType::QUERY_POOL => "VkQueryPool",
        ObjectType::BUFFER_VIEW => "VkBufferView",
        ObjectType::IMAGE_VIEW => "VkImageView",
        ObjectType::SHADER_MODULE => "VkShaderModule",
        ObjectType::PIPELINE_CACHE => "VkPipelineCache",
        ObjectType::PIPELINE_LAYOUT => "VkPipelineLayout",
        ObjectType::RENDER_PASS => "VkRenderPass",
        ObjectType::PIPELINE => "VkPipeline",
        ObjectType::DESCRIPTOR_SET_LAYOUT => "VkDescriptorSetLayout",
        ObjectType::SAMPLER => "VkSampler",
        ObjectType::DESCRIPTOR_POOL => "VkDescriptorPool",
        ObjectType::DESCRIPTOR_SET => "VkDescriptorSet",
        ObjectType::FRAMEBUFFER => "VkFramebuffer",
        ObjectType::COMMAND_POOL => "VkCommandPool",
        ObjectType::SWAPCHAIN_KHR => "VkSwapchainKHR",
        ObjectType::ACCELERATION_STRUCTURE_KHR => "VkAccelerationStructureKHR",
        _ => "VkUnknown",
    }
}

/// Format a Vulkan result code as a human-readable string matching
/// the official Vulkan enum naming convention.
///
/// Covers the most common result codes. Returns `"VK_UNKNOWN"` for
/// unrecognized codes.
pub(crate) fn vk_result_name(result: ash::vk::Result) -> &'static str {
    use ash::vk::Result;
    match result {
        Result::SUCCESS => "VK_SUCCESS",
        Result::NOT_READY => "VK_NOT_READY",
        Result::TIMEOUT => "VK_TIMEOUT",
        Result::EVENT_SET => "VK_EVENT_SET",
        Result::EVENT_RESET => "VK_EVENT_RESET",
        Result::INCOMPLETE => "VK_INCOMPLETE",
        Result::ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY",
        Result::ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        Result::ERROR_INITIALIZATION_FAILED => "VK_ERROR_INITIALIZATION_FAILED",
        Result::ERROR_DEVICE_LOST => "VK_ERROR_DEVICE_LOST",
        Result::ERROR_MEMORY_MAP_FAILED => "VK_ERROR_MEMORY_MAP_FAILED",
        Result::ERROR_LAYER_NOT_PRESENT => "VK_ERROR_LAYER_NOT_PRESENT",
        Result::ERROR_EXTENSION_NOT_PRESENT => "VK_ERROR_EXTENSION_NOT_PRESENT",
        Result::ERROR_FEATURE_NOT_PRESENT => "VK_ERROR_FEATURE_NOT_PRESENT",
        Result::ERROR_INCOMPATIBLE_DRIVER => "VK_ERROR_INCOMPATIBLE_DRIVER",
        Result::ERROR_TOO_MANY_OBJECTS => "VK_ERROR_TOO_MANY_OBJECTS",
        Result::ERROR_FORMAT_NOT_SUPPORTED => "VK_ERROR_FORMAT_NOT_SUPPORTED",
        Result::ERROR_FRAGMENTED_POOL => "VK_ERROR_FRAGMENTED_POOL",
        Result::ERROR_OUT_OF_POOL_MEMORY => "VK_ERROR_OUT_OF_POOL_MEMORY",
        Result::ERROR_SURFACE_LOST_KHR => "VK_ERROR_SURFACE_LOST_KHR",
        Result::ERROR_OUT_OF_DATE_KHR => "VK_ERROR_OUT_OF_DATE_KHR",
        _ => "VK_UNKNOWN",
    }
}

/// Format a pipeline stage flags bitfield as a compact `|`-separated string.
///
/// Uses abbreviated names for each stage flag:
/// - `TOP` for `TOP_OF_PIPE`
/// - `VS` for `VERTEX_SHADER`
/// - `FS` for `FRAGMENT_SHADER`
/// - `CS` for `COMPUTE_SHADER`
/// - `COLOR_OUT` for `COLOR_ATTACHMENT_OUTPUT`
/// - etc.
///
/// Returns `"NONE"` if no flags are set.
pub(crate) fn stage_flags_short(flags: ash::vk::PipelineStageFlags) -> String {
    use ash::vk::PipelineStageFlags;
    let mut parts = Vec::new();
    if flags.contains(PipelineStageFlags::TOP_OF_PIPE) {
        parts.push("TOP");
    }
    if flags.contains(PipelineStageFlags::DRAW_INDIRECT) {
        parts.push("INDIRECT");
    }
    if flags.contains(PipelineStageFlags::VERTEX_INPUT) {
        parts.push("VTX_IN");
    }
    if flags.contains(PipelineStageFlags::VERTEX_SHADER) {
        parts.push("VS");
    }
    if flags.contains(PipelineStageFlags::TESSELLATION_CONTROL_SHADER) {
        parts.push("TCS");
    }
    if flags.contains(PipelineStageFlags::TESSELLATION_EVALUATION_SHADER) {
        parts.push("TES");
    }
    if flags.contains(PipelineStageFlags::GEOMETRY_SHADER) {
        parts.push("GS");
    }
    if flags.contains(PipelineStageFlags::FRAGMENT_SHADER) {
        parts.push("FS");
    }
    if flags.contains(PipelineStageFlags::EARLY_FRAGMENT_TESTS) {
        parts.push("EARLY_Z");
    }
    if flags.contains(PipelineStageFlags::LATE_FRAGMENT_TESTS) {
        parts.push("LATE_Z");
    }
    if flags.contains(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) {
        parts.push("COLOR_OUT");
    }
    if flags.contains(PipelineStageFlags::COMPUTE_SHADER) {
        parts.push("CS");
    }
    if flags.contains(PipelineStageFlags::TRANSFER) {
        parts.push("TRANSFER");
    }
    if flags.contains(PipelineStageFlags::BOTTOM_OF_PIPE) {
        parts.push("BOTTOM");
    }
    if flags.contains(PipelineStageFlags::HOST) {
        parts.push("HOST");
    }
    if flags.contains(PipelineStageFlags::ALL_GRAPHICS) {
        parts.push("ALL_GFX");
    }
    if flags.contains(PipelineStageFlags::ALL_COMMANDS) {
        parts.push("ALL_CMD");
    }
    if parts.is_empty() {
        "NONE".to_string()
    } else {
        parts.join("|")
    }
}

/// Format access flags as a compact `|`-separated string.
///
/// Uses abbreviated names for each access flag:
/// - `SH_R` for `SHADER_READ`
/// - `SH_W` for `SHADER_WRITE`
/// - `COL_W` for `COLOR_ATTACHMENT_WRITE`
/// - `XFR_R` for `TRANSFER_READ`
/// - etc.
///
/// Returns `"NONE"` if no flags are set.
pub(crate) fn access_flags_short(flags: ash::vk::AccessFlags) -> String {
    use ash::vk::AccessFlags;
    let mut parts = Vec::new();
    if flags.contains(AccessFlags::INDIRECT_COMMAND_READ) {
        parts.push("IND_R");
    }
    if flags.contains(AccessFlags::INDEX_READ) {
        parts.push("IDX_R");
    }
    if flags.contains(AccessFlags::VERTEX_ATTRIBUTE_READ) {
        parts.push("VTX_R");
    }
    if flags.contains(AccessFlags::UNIFORM_READ) {
        parts.push("UNI_R");
    }
    if flags.contains(AccessFlags::INPUT_ATTACHMENT_READ) {
        parts.push("INPUT_R");
    }
    if flags.contains(AccessFlags::SHADER_READ) {
        parts.push("SH_R");
    }
    if flags.contains(AccessFlags::SHADER_WRITE) {
        parts.push("SH_W");
    }
    if flags.contains(AccessFlags::COLOR_ATTACHMENT_READ) {
        parts.push("COL_R");
    }
    if flags.contains(AccessFlags::COLOR_ATTACHMENT_WRITE) {
        parts.push("COL_W");
    }
    if flags.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ) {
        parts.push("DS_R");
    }
    if flags.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
        parts.push("DS_W");
    }
    if flags.contains(AccessFlags::TRANSFER_READ) {
        parts.push("XFR_R");
    }
    if flags.contains(AccessFlags::TRANSFER_WRITE) {
        parts.push("XFR_W");
    }
    if flags.contains(AccessFlags::HOST_READ) {
        parts.push("HOST_R");
    }
    if flags.contains(AccessFlags::HOST_WRITE) {
        parts.push("HOST_W");
    }
    if flags.contains(AccessFlags::MEMORY_READ) {
        parts.push("MEM_R");
    }
    if flags.contains(AccessFlags::MEMORY_WRITE) {
        parts.push("MEM_W");
    }
    if parts.is_empty() {
        "NONE".to_string()
    } else {
        parts.join("|")
    }
}

/// Format an image layout as a short human-readable string.
///
/// Uses abbreviated names matching common Vulkan shorthand:
/// - `COLOR_ATT` for `COLOR_ATTACHMENT_OPTIMAL`
/// - `SHADER_RO` for `SHADER_READ_ONLY_OPTIMAL`
/// - `XFR_SRC` for `TRANSFER_SRC_OPTIMAL`
/// - etc.
pub(crate) fn layout_short(layout: ash::vk::ImageLayout) -> &'static str {
    use ash::vk::ImageLayout;
    match layout {
        ImageLayout::UNDEFINED => "UNDEFINED",
        ImageLayout::GENERAL => "GENERAL",
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL => "COLOR_ATT",
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => "DS_ATT",
        ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => "DS_RO",
        ImageLayout::SHADER_READ_ONLY_OPTIMAL => "SHADER_RO",
        ImageLayout::TRANSFER_SRC_OPTIMAL => "XFR_SRC",
        ImageLayout::TRANSFER_DST_OPTIMAL => "XFR_DST",
        ImageLayout::PREINITIALIZED => "PREINIT",
        ImageLayout::PRESENT_SRC_KHR => "PRESENT",
        _ => "OTHER",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardened allocator diagnostic formatters
// ─────────────────────────────────────────────────────────────────────────────

/// All data needed to format a guard band corruption report.
///
/// Populated by [`HardenedAllocator`] when a canary check fails and
/// passed to [`format_guard_report`] for rich diagnostic output.
pub(crate) struct GuardReport {
    /// Diagnostic code (e.g., "IGN-H001" for front, "IGN-H002" for back).
    pub code: &'static str,
    /// Severity level.
    pub severity: Severity,
    /// Which guard region was corrupted: `"front"` or `"back"`.
    pub region: &'static str,
    /// Raw handle of the `VkDeviceMemory` containing the allocation.
    pub memory_handle: u64,
    /// Byte offset of the user data within the memory object.
    pub user_offset: u64,
    /// Size in bytes that the user originally requested.
    pub user_size: u64,
    /// Size of the guard band that was corrupted.
    pub guard_size: u64,
    /// Index of the first corrupted byte within the guard region.
    pub first_corrupted: usize,
    /// Total number of corrupted bytes found.
    pub total_corrupted: usize,
    /// The expected canary word for this allocation.
    pub canary: u64,
    /// The expected byte value at the first corruption site.
    pub expected_byte: u8,
    /// The actual byte value found at the first corruption site.
    pub actual_byte: u8,
    /// Context string describing when the corruption was detected
    /// (e.g., `"Allocator::free()"`, `"quarantine eviction"`).
    pub source: &'static str,
    /// How long the allocation has been alive (if known).
    pub age: Option<Duration>,
    /// Name of the thread that detected the corruption.
    pub thread: String,
    /// Byte offset within the guard region where the hex window starts.
    pub hex_offset: usize,
    /// Expected bytes in the hex comparison window.
    pub hex_expected: Vec<u8>,
    /// Actual bytes found in the hex comparison window.
    pub hex_actual: Vec<u8>,
}

/// All data needed to format a memory leak report entry.
pub(crate) struct LeakEntry {
    /// Raw handle of the `VkDeviceMemory` containing the leaked allocation.
    pub memory_handle: u64,
    /// Byte offset of the leaked allocation within the memory object.
    pub user_offset: u64,
    /// Size in bytes of the leaked allocation.
    pub user_size: u64,
    /// How long the allocation has been alive.
    pub age: Duration,
}

/// Format a complete guard band corruption diagnostic.
///
/// Produces a framed report with:
/// - Severity-colored header with diagnostic code
/// - Memory location arrow
/// - ASCII art layout diagram showing guard/user regions
/// - Annotated arrow pointing to the first corrupted byte
/// - Hex comparison of expected vs actual guard bytes
/// - Diff markers under differing bytes
/// - Concrete byte values at the corruption site
/// - Corruption statistics (count, percentage)
/// - Canary value, timing, and thread context
/// - Corruption pattern analysis
/// - Targeted help suggestion based on corruption location
/// - Environment block and backtrace (for errors)
pub(crate) fn format_guard_report(r: &GuardReport) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(4096);

    // Use full diagnostic header for errors (includes env + backtrace).
    let is_error = matches!(r.severity, Severity::Error);
    if is_error {
        write_full_diagnostic(
            &mut o,
            &s,
            &r.severity,
            r.code,
            &format!("{} guard band corruption", r.region),
            true,
            true,
        );
    } else {
        write_header(
            &mut o,
            &s,
            &r.severity,
            r.code,
            &format!("{} guard band corruption", r.region),
        );
    }

    write_location(
        &mut o,
        &s,
        &format!(
            "VkDeviceMemory({}) offset={} size={}B",
            format_handle(r.memory_handle),
            r.user_offset,
            r.user_size
        ),
    );
    write_pipe_empty(&mut o, &s);

    // ── Layout diagram ──

    let (diagram, fw, uw, _bw) = layout_diagram(r.guard_size, r.user_size, r.guard_size);
    write_pipe(&mut o, &s, &diagram);

    let arrow_pos = match r.region {
        "front" => 1 + (fw * r.first_corrupted) / r.guard_size as usize,
        "back" => {
            let back_start = fw + uw + 5;
            back_start + (_bw * r.first_corrupted) / r.guard_size as usize
        }
        _ => 1,
    };
    let pad: String = " ".repeat(arrow_pos);
    write_pipe_raw(
        &mut o,
        &s,
        &format!(
            "{pad}{}",
            s.bold_red(&format!("^-- byte {}", r.first_corrupted))
        ),
    );

    // ── Hex comparison ──

    write_pipe_empty(&mut o, &s);
    write_pipe(
        &mut o,
        &s,
        &format!(
            "guard hex at {}:",
            s.dim(&format!("+{:#06x}", r.hex_offset))
        ),
    );

    let expected_hex = hex_line(&r.hex_expected);
    let actual_hex = hex_line(&r.hex_actual);
    write_pipe_raw(
        &mut o,
        &s,
        &format!(" {} {expected_hex}", s.green("expect:")),
    );
    write_pipe_raw(&mut o, &s, &format!(" {} {actual_hex}", s.red("actual:")));

    let markers = diff_markers(&r.hex_expected, &r.hex_actual);
    if markers.contains('^') {
        let marker_pad = " ".repeat("actual: ".len() + 1);
        write_pipe_raw(&mut o, &s, &format!("{marker_pad}{}", s.bold_red(&markers)));
    }

    // ── Concrete byte values ──

    write_pipe_empty(&mut o, &s);
    write_pipe_raw(
        &mut o,
        &s,
        &format!(
            "at byte {}: expected {}, found {}",
            r.first_corrupted,
            s.green(&format!("{:#04x}", r.expected_byte)),
            s.bold_red(&format!("{:#04x}", r.actual_byte)),
        ),
    );

    // ── Corruption pattern analysis ──

    write_separator(&mut o, &s);
    write_section(&mut o, &s, "Corruption Analysis");

    let pattern = analyze_corruption_pattern(&r.hex_actual, &r.hex_expected);
    write_pipe(&mut o, &s, &format!("pattern: {}", s.bright_white(&pattern)));

    let pct = (r.total_corrupted as f64 / r.guard_size as f64) * 100.0;
    write_pipe(
        &mut o,
        &s,
        &format!(
            "extent: {}/{} {} guard bytes corrupted ({pct:.1}%)",
            s.bold_red(&r.total_corrupted.to_string()),
            r.guard_size,
            r.region
        ),
    );

    if r.total_corrupted == r.guard_size as usize {
        write_pipe(
            &mut o,
            &s,
            &format!(
                "{}",
                s.bold_red("entire guard band corrupted — large overwrite detected")
            ),
        );
    }

    // ── Metadata ──

    write_pipe_empty(&mut o, &s);
    write_note(&mut o, &s, &format!("canary={:#018x}", r.canary));

    match r.age {
        Some(age) => write_note(
            &mut o,
            &s,
            &format!(
                "allocation alive={} thread=\"{}\"",
                format_duration(age),
                r.thread
            ),
        ),
        None => write_note(&mut o, &s, &format!("thread=\"{}\"", r.thread)),
    }
    write_note(&mut o, &s, &format!("detected during {}", r.source));

    // ── Help suggestion ──

    let suggestion = corruption_suggestion(r.region, r.first_corrupted, r.guard_size as usize);
    write_help(&mut o, &s, &suggestion);

    write_diagnostic_end(&mut o, &s, &r.severity);

    o
}

/// Format a double-free or invalid-free diagnostic.
///
/// Produced when [`HardenedAllocator::free`] is called with an allocation
/// that does not exist in the tracking table.
pub(crate) fn format_double_free(memory_handle: u64, offset: u64, size: u64) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(1024);

    write_full_diagnostic(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-H003",
        "invalid free (allocation not found)",
        true,
        true,
    );
    write_location(
        &mut o,
        &s,
        &format!(
            "VkDeviceMemory({}) offset={offset} size={size}B",
            format_handle(memory_handle)
        ),
    );
    write_pipe_empty(&mut o, &s);
    write_note(&mut o, &s, "allocation not found in tracking table");
    write_note(&mut o, &s, &format!("thread=\"{}\"", current_thread_name()));

    write_separator(&mut o, &s);
    write_section(&mut o, &s, "Probable Causes");
    write_numbered(&mut o, &s, 1, "Double free — the same buffer/image was dropped twice");
    write_numbered(
        &mut o,
        &s,
        2,
        "Freeing memory from a different allocator instance",
    );
    write_numbered(
        &mut o,
        &s,
        3,
        "Memory corruption overwrote allocation metadata",
    );

    write_help(
        &mut o,
        &s,
        "ensure each Buffer/Image is dropped exactly once\n\
         use RAII ownership (let the Drop impl handle cleanup)\n\
         check for manual mem::forget followed by explicit free",
    );

    write_diagnostic_end(&mut o, &s, &Severity::Error);

    o
}

/// Format a memory leak report for allocations still live at allocator
/// shutdown.
///
/// Lists each leaked allocation with its memory handle, offset, size,
/// and age, along with actionable cleanup advice.
pub(crate) fn format_memory_leaks(entries: &[LeakEntry]) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(256 + entries.len() * 128);

    write_header(
        &mut o,
        &s,
        &Severity::Warning,
        "IGN-H005",
        &format!(
            "{} allocation(s) leaked at allocator shutdown",
            entries.len()
        ),
    );
    write_pipe_empty(&mut o, &s);

    // Total leaked bytes.
    let total_bytes: u64 = entries.iter().map(|e| e.user_size).sum();
    write_pipe(
        &mut o,
        &s,
        &format!(
            "total leaked: {} across {} allocation(s)",
            s.bold_yellow(&format_bytes(total_bytes)),
            entries.len()
        ),
    );
    write_pipe_empty(&mut o, &s);

    for (i, e) in entries.iter().enumerate() {
        write_numbered(
            &mut o,
            &s,
            i,
            &format!(
                "VkDeviceMemory({}) offset={} size={}  alive={}",
                format_handle(e.memory_handle),
                e.user_offset,
                format_bytes(e.user_size),
                format_duration(e.age),
            ),
        );
    }

    write_pipe_empty(&mut o, &s);
    write_note(
        &mut o,
        &s,
        "leaking GPU memory can exhaust device-local heaps\n\
         and cause allocation failures for other resources",
    );
    write_help(
        &mut o,
        &s,
        "ensure all Buffers and Images are dropped before\n\
         the allocator is destroyed\n\
         use LifetimeTracker to find leak sources by creation site",
    );

    write_diagnostic_end(&mut o, &s, &Severity::Warning);

    o
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build an ASCII art layout diagram showing front guard, user data,
/// and back guard regions with proportional sizing.
///
/// Returns `(diagram_string, front_width, user_width, back_width)`.
///
/// The diagram looks like:
///
/// ```text
/// [= front 64B =][--- user 128B ---][= back 64B =]
/// ```
fn layout_diagram(front: u64, user: u64, back: u64) -> (String, usize, usize, usize) {
    let total = (front + user + back) as f64;
    let target = 60usize;

    let fl = format!("front {front}B");
    let ul = format!("user {user}B");
    let bl = format!("back {back}B");

    let fw = ((target as f64 * front as f64 / total).round() as usize)
        .max(fl.len() + 4)
        .min(target / 2);
    let bw = ((target as f64 * back as f64 / total).round() as usize)
        .max(bl.len() + 4)
        .min(target / 2);
    let uw = target
        .saturating_sub(fw)
        .saturating_sub(bw)
        .max(ul.len() + 4);

    let diagram = format!(
        "[{}][{}][{}]",
        pad_center(&fl, fw, '='),
        pad_center(&ul, uw, '-'),
        pad_center(&bl, bw, '='),
    );

    (diagram, fw, uw, bw)
}

/// Generate a targeted help suggestion based on which guard region was
/// corrupted and where within the guard the corruption occurred.
///
/// Distinguishes between:
/// - **Boundary corruption** (near the user data edge): typical overflow/underflow
/// - **Far corruption** (far from user data): wild pointer or large offset error
/// - **Mid-range corruption**: moderate overwrite
fn corruption_suggestion(region: &str, byte: usize, guard_size: usize) -> String {
    let near_boundary = match region {
        "front" => byte >= guard_size.saturating_sub(4),
        "back" => byte < 4,
        _ => false,
    };

    let far = match region {
        "front" => byte < 4,
        "back" => byte >= guard_size.saturating_sub(4),
        _ => false,
    };

    match (region, near_boundary, far) {
        ("front", true, _) => format!(
            "byte {byte}/{guard_size} of front guard (boundary with user data)\n\
             typically indicates buffer underflow: write before offset 0\n\
             check for off-by-one errors in buffer write offsets"
        ),
        ("front", _, true) => format!(
            "byte {byte}/{guard_size} of front guard (far from user data)\n\
             may indicate wild pointer or large negative offset\n\
             check for uninitialized pointer or integer underflow in offset calculation"
        ),
        ("front", _, _) => format!(
            "byte {byte}/{guard_size} of front guard\n\
             may indicate wild pointer or substantial underflow\n\
             check stack-allocated buffers for size miscalculation"
        ),
        ("back", true, _) => format!(
            "byte {byte}/{guard_size} of back guard (boundary with user data)\n\
             typically indicates buffer overflow: write past allocation end\n\
             check for off-by-one errors, strlen vs sizeof, or missing null terminator accounting"
        ),
        ("back", _, true) => format!(
            "byte {byte}/{guard_size} of back guard (far from user data)\n\
             may indicate wild pointer or large overflow\n\
             check for loop bounds errors or unclamped index access"
        ),
        ("back", _, _) => format!(
            "byte {byte}/{guard_size} of back guard\n\
             may indicate wild pointer or substantial overflow\n\
             check memcpy/write size calculations and destination offsets"
        ),
        _ => String::new(),
    }
}