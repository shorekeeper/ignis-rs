//! VL diagnostic pipeline.
//!
//! Five-stage processing chain that sits between [`validation_forensic`]
//! (parsing) and the legacy handler/stderr sink. Completely optional: if
//! the application never calls [`install`] or uses any of the macros, the
//! pipeline stays at its default configuration which is behaviorally
//! identical to the pre-pipeline code (print to stderr + dispatch to the
//! handler registered via [`set_validation_handler`]).
//!
//! # Stages
//!
//! ```text
//! parsed ValidationDiagnostic
//!   ↓
//! 1. apply scope-stack severity overrides (thread-local)
//! 2. apply global severity overrides (escalate/demote)
//! 3. check suppression filters (scope, then global)
//! 4. deduplicate per policy
//! 5. capture-mode short-circuit (thread-local)
//! 6. format + fan out to sinks
//! 7. legacy stderr + dispatch_to_handler
//! 8. severity action (panic/abort/breakpoint/callback)
//! ```

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::validation_forensic::{
    dispatch_to_handler, format_forensic_diagnostic, DiagnosticCategory, LayerSeverity,
    ValidationDiagnostic,
};

// Selectors

/// Matches a parsed validation diagnostic by VUID, category, function, etc.
#[derive(Clone, Debug)]
pub enum VlSelector {
    /// VUID glob pattern (supports `*` for zero or more chars).
    /// Matches both the full VUID ("VUID-vkFoo-bar-01234") and the suffix ("01234").
    Vuid(String),
    /// Exact category match.
    Category(DiagnosticCategory),
    /// Vulkan function name glob pattern.
    Function(String),
    /// Any involved object of this Vulkan type (e.g. "VkImage").
    ObjectType(String),
    /// Severity equality.
    Severity(LayerSeverity),
    /// Matches everything.
    All,
}

impl VlSelector {
    /// Test whether the selector matches a diagnostic.
    pub fn matches(&self, d: &ValidationDiagnostic) -> bool {
        match self {
            VlSelector::Vuid(p) => glob_match(p, &d.vuid) || glob_match(p, &d.vuid_suffix),
            VlSelector::Category(c) => d.category == *c,
            VlSelector::Function(p) => glob_match(p, &d.function),
            VlSelector::ObjectType(t) => d.objects.iter().any(|o| &o.vk_type == t),
            VlSelector::Severity(s) => d.severity == *s,
            VlSelector::All => true,
        }
    }
}

/// Simple glob matcher: `*` matches zero or more characters.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    fn rec(p: &[u8], pi: usize, t: &[u8], ti: usize) -> bool {
        if pi == p.len() {
            return ti == t.len();
        }
        if p[pi] == b'*' {
            for k in ti..=t.len() {
                if rec(p, pi + 1, t, k) {
                    return true;
                }
            }
            false
        } else if ti < t.len() && p[pi] == t[ti] {
            rec(p, pi + 1, t, ti + 1)
        } else {
            false
        }
    }
    rec(p, 0, t, 0)
}

// Actions

/// Action taken after a diagnostic has been emitted to all sinks.
#[derive(Clone)]
pub enum VlAction {
    /// Do nothing.
    Nothing,
    /// Already-printed stderr is enough; effectively same as `Nothing`.
    Log,
    /// `panic!` with the formatted diagnostic.
    Panic,
    /// `std::process::abort()` immediately.
    Abort,
    /// Debugger breakpoint (INT3 on x86_64 in debug builds, no-op otherwise).
    Breakpoint,
    /// Custom callback.
    Callback(Arc<dyn Fn(&ValidationDiagnostic) + Send + Sync>),
}

impl std::fmt::Debug for VlAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nothing => write!(f, "Nothing"),
            Self::Log => write!(f, "Log"),
            Self::Panic => write!(f, "Panic"),
            Self::Abort => write!(f, "Abort"),
            Self::Breakpoint => write!(f, "Breakpoint"),
            Self::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}

#[inline(never)]
fn trigger_breakpoint() {
    #[cfg(all(debug_assertions, target_arch = "x86_64"))]
    unsafe {
        std::arch::asm!("int3");
    }
    #[cfg(all(debug_assertions, not(target_arch = "x86_64")))]
    {
        // Fallback: just produce a controlled abort in debug.
        eprintln!("[ignis-vl] breakpoint (no INT3 on this arch)");
    }
    // In release builds breakpoints are no-ops.
}

// Sinks

/// Sink receives a formatted diagnostic string and the structured form.
pub trait VlSink: Send + Sync {
    /// Called once per diagnostic that passes filtering.
    fn emit(&self, diag: &ValidationDiagnostic, formatted: &str);
}

/// Writes formatted output to stderr.
pub struct StderrSink;

impl VlSink for StderrSink {
    fn emit(&self, _diag: &ValidationDiagnostic, formatted: &str) {
        eprint!("{formatted}");
    }
}

/// Appends formatted output to a file. Opens lazily on first write.
pub struct FileSink {
    path: std::path::PathBuf,
    file: Mutex<Option<std::fs::File>>,
}

impl FileSink {
    /// Create a file sink. Path is opened in append mode on first emit.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            file: Mutex::new(None),
        }
    }
}

impl VlSink for FileSink {
    fn emit(&self, _diag: &ValidationDiagnostic, formatted: &str) {
        use std::io::Write;
        let mut guard = self.file.lock().unwrap();
        if guard.is_none() {
            *guard = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .ok();
        }
        if let Some(f) = guard.as_mut() {
            let _ = f.write_all(formatted.as_bytes());
        }
    }
}

/// Sends the structured diagnostic to a user-supplied closure.
pub struct CallbackSink(Arc<dyn Fn(&ValidationDiagnostic) + Send + Sync>);

impl CallbackSink {
    /// Wrap a closure as a sink.
    pub fn new<F: Fn(&ValidationDiagnostic) + Send + Sync + 'static>(f: F) -> Self {
        Self(Arc::new(f))
    }
}

impl VlSink for CallbackSink {
    fn emit(&self, diag: &ValidationDiagnostic, _formatted: &str) {
        (self.0)(diag);
    }
}

/// In-memory ring buffer of the last N diagnostics. Useful for UI panels.
pub struct RingSink {
    capacity: usize,
    inner: Mutex<VecDeque<ValidationDiagnostic>>,
}

impl RingSink {
    /// Create a ring sink holding at most `capacity` diagnostics.
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            inner: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
        })
    }

    /// Snapshot all currently held diagnostics.
    pub fn snapshot(&self) -> Vec<ValidationDiagnostic> {
        self.inner.lock().unwrap().iter().cloned().collect()
    }

    /// Drop everything from the ring.
    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    /// Current size.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl VlSink for RingSink {
    fn emit(&self, diag: &ValidationDiagnostic, _formatted: &str) {
        let mut q = self.inner.lock().unwrap();
        if q.len() >= self.capacity {
            q.pop_front();
        }
        q.push_back(diag.clone());
    }
}

// Dedup

/// How to suppress repeated diagnostics.
#[derive(Clone, Debug)]
pub enum DedupPolicy {
    /// No deduplication.
    Off,
    /// Allow up to N occurrences of each VUID, then silently drop.
    PerVuid(u32),
    /// Allow up to N total diagnostics, then silently drop all.
    Global(u32),
    /// For each VUID, drop repeated occurrences that happen within the window.
    TimeWindow(Duration),
}

struct DedupState {
    policy: DedupPolicy,
    per_vuid: HashMap<String, u32>,
    global_count: u32,
    last_seen: HashMap<String, Instant>,
}

impl DedupState {
    fn new(policy: DedupPolicy) -> Self {
        Self {
            policy,
            per_vuid: HashMap::new(),
            global_count: 0,
            last_seen: HashMap::new(),
        }
    }

    fn should_drop(&mut self, diag: &ValidationDiagnostic) -> bool {
        match &self.policy {
            DedupPolicy::Off => false,
            DedupPolicy::PerVuid(limit) => {
                let c = self.per_vuid.entry(diag.vuid.clone()).or_insert(0);
                *c += 1;
                *c > *limit
            }
            DedupPolicy::Global(limit) => {
                self.global_count += 1;
                self.global_count > *limit
            }
            DedupPolicy::TimeWindow(window) => {
                let now = Instant::now();
                if let Some(t) = self.last_seen.get(&diag.vuid) {
                    if now.duration_since(*t) < *window {
                        return true;
                    }
                }
                self.last_seen.insert(diag.vuid.clone(), now);
                false
            }
        }
    }
}

/// Whether to capture a Rust backtrace alongside the diagnostic.
#[derive(Clone, Copy, Debug)]
pub enum BacktracePolicy {
    /// Never capture.
    None,
    /// Capture for errors only.
    ErrorsOnly,
    /// Capture for warnings and errors.
    WarningsAndErrors,
    /// Always capture.
    All,
}

impl BacktracePolicy {
    #[allow(dead_code)]
    fn should_capture(self, sev: LayerSeverity) -> bool {
        match (self, sev) {
            (Self::None, _) => false,
            (Self::ErrorsOnly, LayerSeverity::Error) => true,
            (Self::WarningsAndErrors, LayerSeverity::Error | LayerSeverity::Warning) => true,
            (Self::All, _) => true,
            _ => false,
        }
    }
}

// Pipeline config

/// Severity key used for the action dispatch table.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum SeverityKey {
    /// Error level.
    Error,
    /// Warning level.
    Warning,
    /// Info level.
    Info,
}

impl From<LayerSeverity> for SeverityKey {
    fn from(s: LayerSeverity) -> Self {
        match s {
            LayerSeverity::Error => Self::Error,
            LayerSeverity::Warning => Self::Warning,
            LayerSeverity::Info => Self::Info,
        }
    }
}

/// Immutable pipeline configuration snapshot. Install via [`install`].
pub struct PipelineConfig {
    /// Suppression selectors (ANY match → drop).
    pub suppress: Vec<VlSelector>,
    /// Severity escalation rules applied in order.
    pub escalate: Vec<(VlSelector, LayerSeverity)>,
    /// Severity demotion rules applied in order.
    pub demote: Vec<(VlSelector, LayerSeverity)>,
    /// Selectors that trigger a debugger breakpoint.
    pub breakpoints: Vec<VlSelector>,
    /// Per-severity post-emit actions.
    pub actions: HashMap<SeverityKey, VlAction>,
    /// Fallback action if no per-severity entry matches.
    pub default_action: VlAction,
    /// Deduplication policy.
    pub dedup: DedupPolicy,
    /// Backtrace capture policy (reserved; processing not yet wired).
    pub backtrace: BacktracePolicy,
    /// Sinks that receive formatted output.
    pub sinks: Vec<Arc<dyn VlSink>>,
    /// Whether to print to stderr if no sinks are registered.
    pub print_stderr: bool,
    /// Whether to forward to the legacy `set_validation_handler` callback.
    pub forward_to_legacy_handler: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            suppress: Vec::new(),
            escalate: Vec::new(),
            demote: Vec::new(),
            breakpoints: Vec::new(),
            actions: HashMap::new(),
            default_action: VlAction::Nothing,
            dedup: DedupPolicy::Off,
            backtrace: BacktracePolicy::ErrorsOnly,
            sinks: Vec::new(),
            print_stderr: true,
            forward_to_legacy_handler: true,
        }
    }
}

struct PipelineInner {
    config: PipelineConfig,
    dedup_state: DedupState,
    _frame_counter: AtomicU64,
}

/// Global pipeline singleton. Lazily initialized with default config.
pub struct VlPipeline {
    inner: Mutex<PipelineInner>,
}

static GLOBAL_PIPELINE: OnceLock<VlPipeline> = OnceLock::new();

/// Access the global pipeline (lazily initialized).
pub fn global() -> &'static VlPipeline {
    GLOBAL_PIPELINE.get_or_init(|| VlPipeline {
        inner: Mutex::new(PipelineInner {
            config: PipelineConfig::default(),
            dedup_state: DedupState::new(DedupPolicy::Off),
            _frame_counter: AtomicU64::new(0),
        }),
    })
}

/// Replace the global pipeline configuration.
///
/// Resets dedup counters. Idempotent: subsequent calls replace the
/// previous configuration entirely.
pub fn install(config: PipelineConfig) {
    let pipeline = global();
    let mut inner = pipeline.inner.lock().unwrap();
    inner.dedup_state = DedupState::new(config.dedup.clone());
    inner.config = config;
}

impl VlPipeline {
    /// Run a parsed diagnostic through the full pipeline.
    pub fn process(&self, mut diag: ValidationDiagnostic) {
        // 1 + 2: apply severity overrides from scope stack then global config.
        SCOPE_STACK.with(|s| {
            for scope in s.borrow().iter() {
                for (sel, sev) in &scope.escalate {
                    if sel.matches(&diag) {
                        diag.severity = *sev;
                    }
                }
                for (sel, sev) in &scope.demote {
                    if sel.matches(&diag) {
                        diag.severity = *sev;
                    }
                }
            }
        });
        {
            let inner = self.inner.lock().unwrap();
            for (sel, sev) in &inner.config.escalate {
                if sel.matches(&diag) {
                    diag.severity = *sev;
                }
            }
            for (sel, sev) in &inner.config.demote {
                if sel.matches(&diag) {
                    diag.severity = *sev;
                }
            }
        }

        // 3: suppression (scope first, global after).
        let scope_suppressed = SCOPE_STACK.with(|s| {
            s.borrow()
                .iter()
                .any(|scope| scope.suppress.iter().any(|sel| sel.matches(&diag)))
        });
        if scope_suppressed {
            return;
        }
        {
            let inner = self.inner.lock().unwrap();
            if inner.config.suppress.iter().any(|sel| sel.matches(&diag)) {
                return;
            }
        }

        // 4: deduplication.
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.dedup_state.should_drop(&diag) {
                return;
            }
        }

        // 5: capture short-circuit (tests).
        let captured_now = CAPTURE.with(|c| {
            if let Some(buf) = c.borrow().as_ref() {
                buf.lock().unwrap().push(diag.clone());
                true
            } else {
                false
            }
        });
        if captured_now {
            return;
        }

        // 6: format and fan out to sinks.
        let formatted = format_forensic_diagnostic(&diag);
        let (sinks, print_stderr, forward_legacy, action, hit_bp) = {
            let inner = self.inner.lock().unwrap();
            let hit_bp = inner
                .config
                .breakpoints
                .iter()
                .any(|sel| sel.matches(&diag));
            let action = inner
                .config
                .actions
                .get(&SeverityKey::from(diag.severity))
                .cloned()
                .unwrap_or_else(|| inner.config.default_action.clone());
            (
                inner.config.sinks.clone(),
                inner.config.print_stderr,
                inner.config.forward_to_legacy_handler,
                action,
                hit_bp,
            )
        };

        for sink in &sinks {
            sink.emit(&diag, &formatted);
        }

        // 7: legacy stderr + dispatch.
        if print_stderr && sinks.is_empty() {
            eprint!("{formatted}");
        }
        if forward_legacy {
            dispatch_to_handler(&diag);
        }

        // Breakpoints happen before severity action so that panics don't
        // unwind past the INT3.
        if hit_bp {
            trigger_breakpoint();
        }

        // 8: severity action.
        match action {
            VlAction::Nothing | VlAction::Log => {}
            VlAction::Panic => panic!("[ignis-vl][{}] {}", diag.vuid, diag.function),
            VlAction::Abort => {
                eprintln!("[ignis-vl] aborting due to {}", diag.vuid);
                std::process::abort();
            }
            VlAction::Breakpoint => trigger_breakpoint(),
            VlAction::Callback(cb) => cb(&diag),
        }
    }
}

// Thread-local scope stack

#[derive(Default)]
#[doc(hidden)]
pub struct ScopeConfig {
    pub suppress: Vec<VlSelector>,
    pub escalate: Vec<(VlSelector, LayerSeverity)>,
    pub demote: Vec<(VlSelector, LayerSeverity)>,
}

thread_local! {
    static SCOPE_STACK: RefCell<Vec<ScopeConfig>> = const { RefCell::new(Vec::new()) };
    static TAGS: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
    static CAPTURE: RefCell<Option<Arc<Mutex<Vec<ValidationDiagnostic>>>>> =
        const { RefCell::new(None) };
}

/// RAII guard that pops a scope off the thread-local stack on drop.
pub struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE_STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Push a scope onto the thread-local stack. Returns an RAII guard.
pub fn push_scope(cfg: ScopeConfig) -> ScopeGuard {
    SCOPE_STACK.with(|s| s.borrow_mut().push(cfg));
    ScopeGuard
}

// Tags

/// Set a persistent thread-local tag.
pub fn set_tag(key: String, value: String) {
    TAGS.with(|t| {
        t.borrow_mut().insert(key, value);
    });
}

/// Remove a thread-local tag.
pub fn remove_tag(key: &str) {
    TAGS.with(|t| {
        t.borrow_mut().remove(key);
    });
}

/// RAII guard that removes a tag on drop.
pub struct TagGuard(String);

impl Drop for TagGuard {
    fn drop(&mut self) {
        remove_tag(&self.0);
    }
}

/// Set a tag that lives until the returned guard is dropped.
pub fn tag_scoped(key: String, value: String) -> TagGuard {
    set_tag(key.clone(), value);
    TagGuard(key)
}

/// Read all current tags on this thread.
pub fn current_tags() -> Vec<(String, String)> {
    TAGS.with(|t| {
        t.borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    })
}

// Capture

/// Activate capture mode on this thread and return the shared buffer.
pub fn begin_capture() -> Arc<Mutex<Vec<ValidationDiagnostic>>> {
    let buf = Arc::new(Mutex::new(Vec::new()));
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(Arc::clone(&buf));
    });
    buf
}

/// Deactivate capture mode on this thread.
pub fn end_capture() {
    CAPTURE.with(|c| {
        *c.borrow_mut() = None;
    });
}

/// Run a closure with capture mode active. Returns captured diagnostics
/// plus the closure's result.
pub fn capture<F, R>(f: F) -> (CapturedDiagnostics, R)
where
    F: FnOnce() -> R,
{
    let buf = begin_capture();
    let result = f();
    end_capture();
    let diags = buf.lock().unwrap().clone();
    (CapturedDiagnostics { inner: diags }, result)
}

/// A collection of captured diagnostics with filter helpers.
pub struct CapturedDiagnostics {
    pub(crate) inner: Vec<ValidationDiagnostic>,
}

impl CapturedDiagnostics {
    /// All captured diagnostics.
    pub fn all(&self) -> &[ValidationDiagnostic] {
        &self.inner
    }

    /// Total count.
    pub fn count(&self) -> usize {
        self.inner.len()
    }

    /// Whether the capture is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Error-severity diagnostics.
    pub fn errors(&self) -> Vec<&ValidationDiagnostic> {
        self.inner
            .iter()
            .filter(|d| matches!(d.severity, LayerSeverity::Error))
            .collect()
    }

    /// Warning-severity diagnostics.
    pub fn warnings(&self) -> Vec<&ValidationDiagnostic> {
        self.inner
            .iter()
            .filter(|d| matches!(d.severity, LayerSeverity::Warning))
            .collect()
    }

    /// Info-severity diagnostics.
    pub fn infos(&self) -> Vec<&ValidationDiagnostic> {
        self.inner
            .iter()
            .filter(|d| matches!(d.severity, LayerSeverity::Info))
            .collect()
    }

    /// Diagnostics whose VUID matches the glob pattern.
    pub fn by_vuid(&self, pattern: &str) -> Vec<&ValidationDiagnostic> {
        self.inner
            .iter()
            .filter(|d| glob_match(pattern, &d.vuid) || glob_match(pattern, &d.vuid_suffix))
            .collect()
    }

    /// Diagnostics in the given category.
    pub fn by_category(&self, cat: DiagnosticCategory) -> Vec<&ValidationDiagnostic> {
        self.inner.iter().filter(|d| d.category == cat).collect()
    }

    /// Diagnostics whose function name matches the glob pattern.
    pub fn by_function(&self, pattern: &str) -> Vec<&ValidationDiagnostic> {
        self.inner
            .iter()
            .filter(|d| glob_match(pattern, &d.function))
            .collect()
    }
}

// Expectations

/// Count constraint for [`ExpectRule`].
#[derive(Debug, Clone, Copy)]
pub enum ExpectCount {
    /// Must occur exactly N times.
    Exactly(usize),
    /// Must occur at least N times.
    AtLeast(usize),
    /// Must occur at most N times.
    AtMost(usize),
    /// Must not occur.
    Never,
}

/// Single expectation rule.
#[derive(Debug, Clone)]
pub struct ExpectRule {
    /// What to match.
    pub selector: VlSelector,
    /// How many matches are allowed / required.
    pub count: ExpectCount,
    /// Human-readable description for error messages.
    pub description: String,
}

/// Check that captured diagnostics satisfy all rules. Returns a textual
/// error describing the first unmet expectation, or Ok(()).
pub fn verify_expectations(
    captured: &CapturedDiagnostics,
    rules: &[ExpectRule],
) -> std::result::Result<(), String> {
    for rule in rules {
        let matching = captured
            .inner
            .iter()
            .filter(|d| rule.selector.matches(d))
            .count();
        let ok = match rule.count {
            ExpectCount::Exactly(n) => matching == n,
            ExpectCount::AtLeast(n) => matching >= n,
            ExpectCount::AtMost(n) => matching <= n,
            ExpectCount::Never => matching == 0,
        };
        if !ok {
            return Err(format!(
                "expectation `{}` failed: got {} match(es), required {:?}",
                rule.description, matching, rule.count
            ));
        }
    }
    Ok(())
}

// Builder

/// Fluent builder for [`PipelineConfig`].
#[derive(Default)]
pub struct VlConfigBuilder {
    config: PipelineConfig,
}

impl VlConfigBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Suppress a VUID glob pattern.
    pub fn suppress_vuid(mut self, pattern: impl Into<String>) -> Self {
        self.config
            .suppress
            .push(VlSelector::Vuid(pattern.into()));
        self
    }

    /// Suppress an entire category.
    pub fn suppress_category(mut self, cat: DiagnosticCategory) -> Self {
        self.config.suppress.push(VlSelector::Category(cat));
        self
    }

    /// Suppress diagnostics coming from a function-name glob.
    pub fn suppress_function(mut self, pattern: impl Into<String>) -> Self {
        self.config
            .suppress
            .push(VlSelector::Function(pattern.into()));
        self
    }

    /// Suppress diagnostics involving an object of this Vulkan type.
    pub fn suppress_object_type(mut self, ty: impl Into<String>) -> Self {
        self.config
            .suppress
            .push(VlSelector::ObjectType(ty.into()));
        self
    }

    /// Escalate diagnostics matching a VUID glob to `to`.
    pub fn escalate_vuid(mut self, pattern: impl Into<String>, to: LayerSeverity) -> Self {
        self.config
            .escalate
            .push((VlSelector::Vuid(pattern.into()), to));
        self
    }

    /// Escalate an entire category to `to`.
    pub fn escalate_category(mut self, cat: DiagnosticCategory, to: LayerSeverity) -> Self {
        self.config.escalate.push((VlSelector::Category(cat), to));
        self
    }

    /// Demote diagnostics matching a VUID glob to `to`.
    pub fn demote_vuid(mut self, pattern: impl Into<String>, to: LayerSeverity) -> Self {
        self.config
            .demote
            .push((VlSelector::Vuid(pattern.into()), to));
        self
    }

    /// Demote an entire category to `to`.
    pub fn demote_category(mut self, cat: DiagnosticCategory, to: LayerSeverity) -> Self {
        self.config.demote.push((VlSelector::Category(cat), to));
        self
    }

    /// Trigger a debugger breakpoint when the selector matches.
    pub fn breakpoint_on(mut self, sel: VlSelector) -> Self {
        self.config.breakpoints.push(sel);
        self
    }

    /// Set the action for a specific severity.
    pub fn action(mut self, sev: LayerSeverity, action: VlAction) -> Self {
        self.config.actions.insert(sev.into(), action);
        self
    }

    /// Set the default action when no severity-specific action matches.
    pub fn default_action(mut self, action: VlAction) -> Self {
        self.config.default_action = action;
        self
    }

    /// Set the dedup policy.
    pub fn dedup(mut self, policy: DedupPolicy) -> Self {
        self.config.dedup = policy;
        self
    }

    /// Set the backtrace policy.
    pub fn backtrace(mut self, policy: BacktracePolicy) -> Self {
        self.config.backtrace = policy;
        self
    }

    /// Register a custom sink.
    pub fn sink(mut self, s: Arc<dyn VlSink>) -> Self {
        self.config.sinks.push(s);
        self
    }

    /// Register the stderr sink.
    pub fn sink_stderr(self) -> Self {
        self.sink(Arc::new(StderrSink))
    }

    /// Register a file sink.
    pub fn sink_file(self, path: impl Into<std::path::PathBuf>) -> Self {
        self.sink(Arc::new(FileSink::new(path)))
    }

    /// Register a ring-buffer sink. Returns the ring handle for later read.
    pub fn sink_ring(mut self, capacity: usize) -> (Self, Arc<RingSink>) {
        let ring = RingSink::new(capacity);
        self.config.sinks.push(ring.clone());
        (self, ring)
    }

    /// Register a callback sink.
    pub fn sink_callback<F: Fn(&ValidationDiagnostic) + Send + Sync + 'static>(
        self,
        f: F,
    ) -> Self {
        self.sink(Arc::new(CallbackSink::new(f)))
    }

    /// Disable the implicit stderr fallback used when no sinks are registered.
    pub fn no_stderr(mut self) -> Self {
        self.config.print_stderr = false;
        self
    }

    /// Skip forwarding to the legacy `set_validation_handler` callback.
    pub fn no_legacy_forward(mut self) -> Self {
        self.config.forward_to_legacy_handler = false;
        self
    }

    /// Produce the final config without installing.
    pub fn build(self) -> PipelineConfig {
        self.config
    }

    /// Install the resulting config as the global pipeline configuration.
    pub fn install(self) {
        install(self.build());
    }
}