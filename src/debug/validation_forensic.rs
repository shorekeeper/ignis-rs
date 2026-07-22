//! Forensic analysis of Vulkan validation layer messages.
//!
//! Parses raw VL output, extracts structured information (VUID, function,
//! objects, offending values), cross-references with ignis-side registries
//! (object resolvers, submit backtraces), and produces human-readable
//! diagnostics with actionable ignis-specific fixes.
//!
//! # Architecture
//!
//! The validation layer emits every rule violation through a single
//! `VK_EXT_debug_utils` messenger callback. This module takes the raw
//! text of such messages and turns them into structured diagnostics
//! that reference the user's ignis API calls instead of anonymous
//! Vulkan handles.
//!
//! # What gets parsed
//!
//! - **VUID identifier** (e.g. `VUID-VkImageMemoryBarrier-oldLayout-01213`)
//! - **Vulkan function name** that triggered the check
//! - **Parameter path** inside that function call
//! - **Object handles** referenced in the message (type + raw u64)
//! - **Vulkan enum values** mentioned (flags, layouts, formats, etc)
//!
//! # What gets cross-referenced
//!
//! - Object handles are resolved to debug names and creation locations
//!   via an optional `ObjectResolver` trait. When unregistered, only
//!   the raw handle is shown.
//! - Thread-local submit backtrace stack captures the user's call stack
//!   at the time of `vkQueueSubmit`, since the layer callback fires
//!   later (asynchronously from submit) with no stack link.
//!
//! # Knowledge base
//!
//! A small static database maps common VUID suffixes to human-readable
//! explanations plus ignis-specific fix suggestions. Coverage is not
//! complete; unknown VUIDs still get a structured diagnostic, just
//! without the explanatory sections.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};

use crate::diagnostic::{self, Severity, Style};

// Structured diagnostic types

/// Severity reported by the validation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSeverity {
    /// Spec violation. The error is real.
    Error,
    /// Suboptimal usage or performance hint.
    Warning,
    /// Informational message (usually benign).
    Info,
}

/// Classification of what went wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticCategory {
    /// Image or buffer missing a required usage flag.
    UsageFlagMismatch,
    /// Image layout is incompatible with the operation or usage.
    LayoutTransition,
    /// Access without a matching prior barrier.
    SynchronizationHazard,
    /// Descriptor set mismatch with bound pipeline layout.
    DescriptorMismatch,
    /// Pipeline binding or compatibility issue.
    PipelineMismatch,
    /// Child object outlived its parent or was double-freed.
    ObjectLifetime,
    /// Memory binding, allocation size, or alignment problem.
    MemoryBinding,
    /// Queue submission or presentation error.
    QueueSubmission,
    /// Required feature or extension was not enabled at device creation.
    FeatureNotEnabled,
    /// Anything not classified above.
    Other,
}

/// A single Vulkan object mentioned in the validation message.
#[derive(Debug, Clone)]
pub struct InvolvedObject {
    /// Vulkan type name like `VkImage`, `VkBuffer`.
    pub vk_type: String,
    /// Raw handle value as reported by the layer.
    pub handle: u64,
    /// Debug name from `VK_EXT_debug_utils`, if the resolver returns one.
    pub debug_name: Option<String>,
    /// Source location where the object was created, if known.
    pub creation_location: Option<String>,
}

/// Static knowledge base entry for a specific VUID.
pub struct KnowledgeEntry {
    /// Numeric suffix of the VUID (e.g. `"01213"`).
    pub vuid_suffix: &'static str,
    /// Short human-readable title.
    pub title: &'static str,
    /// Classification.
    pub category: DiagnosticCategory,
    /// Plain-English description of what the user did that triggered this.
    pub what_happened: &'static str,
    /// Why the Vulkan spec prohibits it.
    pub why_rejected: &'static str,
    /// Concrete fix using ignis API patterns.
    pub ignis_fix: &'static str,
    /// Spec section reference for further reading.
    pub spec_section: &'static str,
}

/// Fully parsed and cross-referenced validation diagnostic.
#[derive(Clone)]
pub struct ValidationDiagnostic {
    /// Full VUID identifier, e.g. `"VUID-VkImageMemoryBarrier-oldLayout-01213"`.
    pub vuid: String,
    /// Numeric suffix of the VUID.
    pub vuid_suffix: String,
    /// Vulkan function name that triggered the check.
    pub function: String,
    /// Parameter path inside the function call, if extractable.
    pub parameter: Option<String>,
    /// Objects mentioned in the message, with cross-referenced metadata.
    pub objects: Vec<InvolvedObject>,
    /// Vulkan enum constants named in the message.
    pub values: Vec<String>,
    /// Original raw message body from the layer.
    pub raw_body: String,
    /// Category derived from the knowledge base or heuristics.
    pub category: DiagnosticCategory,
    /// Severity from the layer callback.
    pub severity: LayerSeverity,
    /// Matching knowledge base entry, if any.
    pub knowledge: Option<super::vuid_kb::KnowledgeLookup>,
    /// Captured submit backtrace from thread-local stack.
    pub submit_backtrace: Option<Vec<String>>,
}

// Object resolver (cross-reference with user-side registries)

/// Returned by `ObjectResolver` implementations to attach metadata to a handle.
pub struct ResolvedObject {
    /// Debug name if the object was registered.
    pub debug_name: Option<String>,
    /// Where the object was created (file:line:col), if tracked.
    pub creation_location: Option<String>,
}

/// User-implementable trait for resolving Vulkan handles into names.
///
/// Attach an implementation via `Ignis::set_object_resolver` to enrich
/// every validation diagnostic with your application's view of what
/// each handle represents.
pub trait ObjectResolver: Send + Sync {
    /// Resolve a (type, handle) pair to metadata. Return `None` when the
    /// object is not known to this resolver.
    fn resolve(&self, vk_type: &str, handle: u64) -> Option<ResolvedObject>;
}

static GLOBAL_RESOLVER: OnceLock<Mutex<Option<Arc<dyn ObjectResolver>>>> = OnceLock::new();

/// Install a global object resolver. Replaces any previous resolver.
pub fn set_object_resolver(resolver: Arc<dyn ObjectResolver>) {
    let slot = GLOBAL_RESOLVER.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap() = Some(resolver);
}

/// Clear the installed resolver, reverting to raw handles.
pub fn clear_object_resolver() {
    if let Some(slot) = GLOBAL_RESOLVER.get() {
        *slot.lock().unwrap() = None;
    }
}

fn resolve_object(vk_type: &str, handle: u64) -> (Option<String>, Option<String>) {
    let Some(slot) = GLOBAL_RESOLVER.get() else {
        return (None, None);
    };
    let guard = slot.lock().unwrap();
    let Some(resolver) = guard.as_ref() else {
        return (None, None);
    };
    match resolver.resolve(vk_type, handle) {
        Some(r) => (r.debug_name, r.creation_location),
        None => (None, None),
    }
}

// Submit backtrace capture

thread_local! {
    /// Stack of backtraces captured at each submit site on this thread.
    /// Lookup from the validation callback reads the last entry.
    static SUBMIT_BT_STACK: RefCell<Vec<Vec<String>>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard that captures a backtrace on creation and pops it on drop.
///
/// # Limitations
///
/// Thread-local storage. The validation layer invokes its callback on
/// whichever thread the offending command was processed on, which is
/// normally the submit thread but not always:
///
/// - GPU-assisted validation runs checks on a background thread. The
///   backtrace will be empty for those messages.
/// - Some layer implementations batch messages and deliver them lazily
///   from a dedicated worker thread.
///
/// For best results combine with object naming via `DebugUtils` or the
/// object resolver so messages identify your objects even when the
/// stack is not recoverable.
pub struct SubmitBacktraceGuard;

impl SubmitBacktraceGuard {
    /// Capture the current backtrace and push it onto this thread's stack.
    pub fn new() -> Self {
        let bt = std::backtrace::Backtrace::force_capture();
        let frames = parse_backtrace_frames(&bt);
        SUBMIT_BT_STACK.with(|s| s.borrow_mut().push(frames));
        Self
    }
}

impl Drop for SubmitBacktraceGuard {
    fn drop(&mut self) {
        SUBMIT_BT_STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

impl Default for SubmitBacktraceGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Read the most recent submit backtrace on this thread. Called from
/// the validation messenger callback.
fn peek_submit_backtrace() -> Option<Vec<String>> {
    SUBMIT_BT_STACK.with(|s| s.borrow().last().cloned())
}

fn parse_backtrace_frames(bt: &std::backtrace::Backtrace) -> Vec<String> {
    let text = bt.to_string();
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("at ") {
            continue;
        }
        // Filter stdlib, runtime, and our own capture frames.
        if trimmed.contains("backtrace::")
            || trimmed.contains("std::backtrace")
            || trimmed.contains("std::rt::")
            || trimmed.contains("std::panic")
            || trimmed.contains("std::sys::")
            || trimmed.contains("core::ops::")
            || trimmed.contains("__rust_")
            || trimmed.contains("BaseThreadInitThunk")
            || trimmed.contains("RtlUserThreadStart")
            || trimmed.contains("validation_forensic::")
            || trimmed.contains("SubmitBacktraceGuard")
        {
            continue;
        }
        // Strip the leading frame number ("5: ") added by std::backtrace.
        let cleaned = if trimmed
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            let stripped = trimmed.trim_start_matches(|c: char| c.is_ascii_digit());
            stripped.strip_prefix(": ").unwrap_or(trimmed).to_string()
        } else {
            trimmed.to_string()
        };
        out.push(cleaned);
        if out.len() >= 16 {
            break;
        }
    }
    out
}

// Handler dispatch

/// User callback for structured validation diagnostics.
pub type ValidationHandler = Box<dyn Fn(&ValidationDiagnostic) + Send + Sync>;

static GLOBAL_HANDLER: OnceLock<Mutex<Option<ValidationHandler>>> = OnceLock::new();

/// Register a handler that receives every parsed validation diagnostic.
/// Replaces any previous handler. Called in addition to (not instead of)
/// the default stderr output unless the handler calls
/// `suppress_default_output` internally.
pub fn set_handler(handler: ValidationHandler) {
    let slot = GLOBAL_HANDLER.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap() = Some(handler);
}

pub(crate) fn dispatch_to_handler(diag: &ValidationDiagnostic) {
    let Some(slot) = GLOBAL_HANDLER.get() else {
        return;
    };
    let guard = slot.lock().unwrap();
    if let Some(h) = guard.as_ref() {
        h(diag);
    }
}

// Parser

/// Parse a raw validation layer message into structured form.
///
/// Returns `None` if the message does not look like a VUID-tagged error
/// (e.g. it is a general informational message from the layer itself).
pub fn parse_validation_message(
    raw: &str,
    message_id_name: &str,
    severity: LayerSeverity,
) -> Option<ValidationDiagnostic> {
    // Extract VUID. Try id_name first (most reliable), fall back to body scan.
    let vuid = if message_id_name.starts_with("VUID-") {
        message_id_name.to_string()
    } else {
        extract_vuid(raw)?
    };
    let vuid_suffix = vuid.rsplit('-').next().unwrap_or("").to_string();

    let function = extract_function(raw).unwrap_or_default();
    let parameter = extract_parameter(raw);
    let raw_objects = extract_objects(raw);
    let values = extract_enum_values(raw);

    let objects: Vec<InvolvedObject> = raw_objects
        .into_iter()
        .map(|(vk_type, handle)| {
            let (debug_name, creation_location) = resolve_object(&vk_type, handle);
            InvolvedObject {
                vk_type,
                handle,
                debug_name,
                creation_location,
            }
        })
        .collect();

    let knowledge = lookup_knowledge(&vuid_suffix);
    let category = match &knowledge {
        Some(k) => k.category,
        None => infer_category_from_function(&function),
    };

    let submit_backtrace = peek_submit_backtrace();

    Some(ValidationDiagnostic {
        vuid,
        vuid_suffix,
        function,
        parameter,
        objects,
        values,
        raw_body: raw.to_string(),
        category,
        severity,
        knowledge,
        submit_backtrace,
    })
}

/// Guess a diagnostic category from the Vulkan function name when the
/// knowledge base does not have an explicit entry. Helps dashboards
/// group similar errors without every VUID being catalogued.
fn infer_category_from_function(function: &str) -> DiagnosticCategory {
    if function.contains("Barrier") || function.contains("PipelineBarrier") {
        DiagnosticCategory::SynchronizationHazard
    } else if function.contains("Copy") || function.contains("Clear") || function.contains("Blit") {
        DiagnosticCategory::MemoryBinding
    } else if function.contains("Submit") || function.contains("Present") {
        DiagnosticCategory::QueueSubmission
    } else if function.contains("Draw") || function.contains("Dispatch") {
        DiagnosticCategory::PipelineMismatch
    } else if function.contains("Descriptor") {
        DiagnosticCategory::DescriptorMismatch
    } else if function.contains("Destroy") || function.contains("Free") {
        DiagnosticCategory::ObjectLifetime
    } else {
        DiagnosticCategory::Other
    }
}

fn extract_vuid(s: &str) -> Option<String> {
    let start = s.find("VUID-")?;
    let rest = &s[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
        .unwrap_or(rest.len());
    let candidate = &rest[..end];
    if candidate.len() > 10 {
        Some(candidate.to_string())
    } else {
        None
    }
}

fn extract_function(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 4 < bytes.len() {
        // Match "vk" with uppercase third character.
        if bytes[i] == b'v'
            && bytes[i + 1] == b'k'
            && bytes[i + 2].is_ascii_uppercase()
            && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
        {
            // Scan forward for either an open paren or a non-alphanumeric char.
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' && j - i < 64 {
                // Verified: function name followed by paren.
                return Some(s[i..j].to_string());
            }
            i = j.max(i + 2);
            continue;
        }
        i += 1;
    }
    None
}

fn extract_parameter(s: &str) -> Option<String> {
    // Typical forms: pImageMemoryBarriers[0].newLayout, pSubmits[0], pCreateInfo->usage
    let mut idx = 0;
    while idx < s.len() {
        let slice = &s[idx..];
        let Some(pos) = slice.find('p') else { break };
        let abs = idx + pos;
        let rest = &s[abs..];
        let bytes = rest.as_bytes();
        // Must be preceded by non-alpha (word boundary) and followed by uppercase.
        let prev_ok = abs == 0 || !s.as_bytes()[abs - 1].is_ascii_alphanumeric();
        if bytes.len() < 3 || !prev_ok || !bytes[1].is_ascii_uppercase() {
            idx = abs + 1;
            continue;
        }
        let end = rest
            .find(|c: char| {
                !c.is_ascii_alphanumeric()
                    && c != '['
                    && c != ']'
                    && c != '.'
                    && c != '-'
                    && c != '>'
            })
            .unwrap_or(rest.len());
        let candidate = &rest[..end];
        if candidate.len() > 3 && candidate.len() < 80 {
            return Some(candidate.to_string());
        }
        idx = abs + 1;
    }
    None
}

fn extract_objects(s: &str) -> Vec<(String, u64)> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < bytes.len() {
        // Match "Vk" with uppercase third character at word boundary.
        if bytes[i] == b'V'
            && bytes[i + 1] == b'k'
            && bytes[i + 2].is_ascii_uppercase()
            && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
        {
            // Scan type name.
            let type_start = i;
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
                j += 1;
            }
            let type_len = j - type_start;
            if type_len < 4 || type_len > 64 {
                i = j.max(i + 2);
                continue;
            }
            let vk_type = &s[type_start..j];

            // Skip whitespace, then look for 0x...
            let mut k = j;
            while k < bytes.len() && bytes[k] == b' ' {
                k += 1;
            }
            if k + 2 < bytes.len() && bytes[k] == b'0' && bytes[k + 1] == b'x' {
                let hex_start = k + 2;
                let mut hex_end = hex_start;
                while hex_end < bytes.len()
                    && hex_end - hex_start < 16
                    && bytes[hex_end].is_ascii_hexdigit()
                {
                    hex_end += 1;
                }
                if hex_end > hex_start {
                    if let Ok(handle) = u64::from_str_radix(&s[hex_start..hex_end], 16) {
                        let entry = (vk_type.to_string(), handle);
                        if !out.contains(&entry) {
                            out.push(entry);
                        }
                    }
                    i = hex_end;
                    continue;
                }
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

fn extract_enum_values(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < s.len() {
        let slice = &s[idx..];
        let Some(pos) = slice.find("VK_") else { break };
        let abs = idx + pos;
        let prev_ok = abs == 0 || !s.as_bytes()[abs - 1].is_ascii_alphanumeric();
        if !prev_ok {
            idx = abs + 3;
            continue;
        }
        let rest = &s[abs..];
        let end = rest
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        let val = &rest[..end];
        if val.len() > 4 && val.len() < 80 && !out.iter().any(|v: &String| v == val) {
            out.push(val.to_string());
        }
        idx = abs + end.max(3);
    }
    out
}

/// Look up a VUID suffix in the knowledge base (static + runtime entries).
pub fn lookup_knowledge(suffix: &str) -> Option<super::vuid_kb::KnowledgeLookup> {
    super::vuid_kb::lookup(suffix)
}

/// Format a parsed diagnostic into the standard ignis framed output.
pub fn format_forensic_diagnostic(diag: &ValidationDiagnostic) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(4096);

    let severity = match diag.severity {
        LayerSeverity::Error => Severity::Error,
        LayerSeverity::Warning => Severity::Warning,
        LayerSeverity::Info => Severity::Info,
    };

    let title = diag
        .knowledge
        .as_ref()
        .map(|k| k.title.as_str())
        .unwrap_or("validation layer reported a rule violation");

    diagnostic::write_header(
        &mut o,
        &s,
        &severity,
        "IGN-V002",
        &format!("{title} [{}]", diag.vuid),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    // What the layer said (structured)
    diagnostic::write_section(&mut o, &s, "Layer Report");
    diagnostic::write_kv(&mut o, &s, "VUID", &diag.vuid);
    if !diag.function.is_empty() {
        diagnostic::write_kv(&mut o, &s, "function", &diag.function);
    }
    if let Some(p) = &diag.parameter {
        diagnostic::write_kv(&mut o, &s, "parameter", p);
    }
    diagnostic::write_kv(&mut o, &s, "category", &format!("{:?}", diag.category));

    // Involved objects
    if !diag.objects.is_empty() {
        diagnostic::write_section(&mut o, &s, "Involved Objects");
        for obj in &diag.objects {
            let name_str = match &obj.debug_name {
                Some(n) => format!(" = {}", s.bold_cyan(&format!("\"{n}\""))),
                None => String::new(),
            };
            diagnostic::write_pipe(
                &mut o,
                &s,
                &format!("{}({:#x}){name_str}", obj.vk_type, obj.handle),
            );
            if let Some(loc) = &obj.creation_location {
                diagnostic::write_pipe_raw(
                    &mut o,
                    &s,
                    &format!("     {} {}", s.dim("created at"), s.underline(loc)),
                );
            }
        }
    }

    // Vulkan enum values mentioned
    if !diag.values.is_empty() {
        diagnostic::write_section(&mut o, &s, "Vulkan Values in Message");
        for v in diag.values.iter().take(12) {
            diagnostic::write_pipe(&mut o, &s, &format!("• {v}"));
        }
        if diag.values.len() > 12 {
            diagnostic::write_pipe_raw(
                &mut o,
                &s,
                &s.dim(&format!("  ... {} more", diag.values.len() - 12)),
            );
        }
    }

    // Knowledge base explanation
    if let Some(kb) = &diag.knowledge {
        diagnostic::write_section(&mut o, &s, "What You Did");
        for line in kb.what_happened.lines() {
            diagnostic::write_pipe(&mut o, &s, line);
        }

        diagnostic::write_section(&mut o, &s, "Why Vulkan Rejected It");
        for line in kb.why_rejected.lines() {
            diagnostic::write_pipe(&mut o, &s, line);
        }

        diagnostic::write_section(&mut o, &s, "Ignis-Specific Fix");
        for line in kb.ignis_fix.lines() {
            diagnostic::write_pipe(&mut o, &s, line);
        }

        diagnostic::write_pipe_empty(&mut o, &s);
        diagnostic::write_kv(&mut o, &s, "spec reference", &kb.spec_section);
    } else {
        diagnostic::write_section(&mut o, &s, "Knowledge Base");
        diagnostic::write_pipe(
            &mut o,
            &s,
            "this VUID is not yet catalogued in the ignis knowledge base.",
        );
        diagnostic::write_pipe(
            &mut o,
            &s,
            "see the raw layer message below for details, or consult the Vulkan",
        );
        diagnostic::write_pipe(&mut o, &s, "spec via the VUID identifier shown above.");
    }

    // Submit stack if captured
    if let Some(bt) = &diag.submit_backtrace {
        if !bt.is_empty() {
            diagnostic::write_section(&mut o, &s, "Submit Call Stack");
            for (i, frame) in bt.iter().take(10).enumerate() {
                let colored = if frame.contains("ignis::") {
                    s.dim(frame)
                } else {
                    s.bright_white(frame)
                };
                diagnostic::write_pipe_raw(&mut o, &s, &format!("  {i:>2}: {colored}"));
            }
        }
    }

    // Raw layer message (truncated)
    diagnostic::write_section(&mut o, &s, "Raw Layer Message");
    let lines: Vec<&str> = diag.raw_body.lines().collect();
    for line in lines.iter().take(12) {
        diagnostic::write_pipe_raw(&mut o, &s, &format!("  {}", s.dim(line)));
    }
    if lines.len() > 12 {
        diagnostic::write_pipe_raw(
            &mut o,
            &s,
            &s.dim(&format!("     ... {} more lines", lines.len() - 12)),
        );
    }

    diagnostic::write_diagnostic_end(&mut o, &s, &severity);
    o
}
