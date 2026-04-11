//! Object lifetime tracking with caller-location capture.
//!
//! [`LifetimeTracker`] registers every Vulkan object created through ignis,
//! recording the creation site via `#[track_caller]`. When objects are
//! destroyed, they are unregistered. At any point (and automatically at
//! tracker drop), a full leak report can be generated showing every live
//! object with its creation site, age, and usage count.
//!
//! # Usage
//!
//! ```rust,no_run
//! # use ignis::lifetime::LifetimeTracker;
//! # use ash::vk;
//! let tracker = LifetimeTracker::new();
//!
//! // Register objects as they are created.
//! tracker.register(vk::ObjectType::PIPELINE, 0x42, Some("shadow_pipeline"));
//!
//! // Record usage (e.g., binding to a command buffer).
//! tracker.record_usage(vk::ObjectType::PIPELINE, 0x42);
//!
//! // Unregister on destroy.
//! tracker.unregister(vk::ObjectType::PIPELINE, 0x42);
//!
//! // At shutdown, any remaining objects are reported as leaks.
//! drop(tracker);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ash::vk;

use crate::diagnostic::{self, Severity, Style};

/// Unique key for a tracked Vulkan object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ObjectKey {
    object_type: vk::ObjectType,
    handle: u64,
}

/// Metadata stored per tracked object.
struct TrackedObject {
    object_type: vk::ObjectType,
    handle: u64,
    name: Option<String>,
    created_at: Instant,
    caller_file: &'static str,
    caller_line: u32,
    caller_column: u32,
    usage_count: AtomicU64,
}

/// Action taken when leaks are detected at tracker shutdown.
#[derive(Default)]
pub enum LeakAction {
    /// Print the report to stderr and continue.
    #[default]
    Log,
    /// Panic with the full report. Good for CI.
    Panic,
    /// Call a user-provided function with the formatted report.
    Callback(Box<dyn Fn(&str) + Send + Sync>),
    /// Do nothing. Leaks are silently ignored.
    Ignore,
}


impl std::fmt::Debug for LeakAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Log => write!(f, "Log"),
            Self::Panic => write!(f, "Panic"),
            Self::Callback(_) => write!(f, "Callback(...)"),
            Self::Ignore => write!(f, "Ignore"),
        }
    }
}

/// Tracks the lifetime of Vulkan objects with creation-site capture.
///
/// Thread-safe: all operations are mutex-protected.
pub struct LifetimeTracker {
    objects: Mutex<HashMap<ObjectKey, TrackedObject>>,
    on_leak: LeakAction,
}

impl LifetimeTracker {
    /// Create a new lifetime tracker.
    pub fn new() -> Self {
        Self {
            objects: Mutex::new(HashMap::new()),
            on_leak: LeakAction::default(),
        }
    }

    /// Set the action taken when leaks are detected at shutdown.
    pub fn on_leak(mut self, action: LeakAction) -> Self {
        self.on_leak = action;
        self
    }

    /// Register a new object.
    ///
    /// Call this immediately after creating a Vulkan object. The caller
    /// location is captured automatically via `#[track_caller]`.
    #[track_caller]
    pub fn register(&self, object_type: vk::ObjectType, handle: u64, name: Option<&str>) {
        let loc = std::panic::Location::caller();
        let key = ObjectKey {
            object_type,
            handle,
        };
        let obj = TrackedObject {
            object_type,
            handle,
            name: name.map(String::from),
            created_at: Instant::now(),
            caller_file: loc.file(),
            caller_line: loc.line(),
            caller_column: loc.column(),
            usage_count: AtomicU64::new(0),
        };
        self.objects.lock().unwrap().insert(key, obj);
    }

    /// Register with an explicit caller location.
    ///
    /// Use when the registration site is not the logical creation site
    /// (e.g., wrapping a factory method).
    pub fn register_at(
        &self,
        object_type: vk::ObjectType,
        handle: u64,
        name: Option<&str>,
        file: &'static str,
        line: u32,
        column: u32,
    ) {
        let key = ObjectKey {
            object_type,
            handle,
        };
        let obj = TrackedObject {
            object_type,
            handle,
            name: name.map(String::from),
            created_at: Instant::now(),
            caller_file: file,
            caller_line: line,
            caller_column: column,
            usage_count: AtomicU64::new(0),
        };
        self.objects.lock().unwrap().insert(key, obj);
    }

    /// Unregister an object (on destroy).
    ///
    /// Returns `true` if the object was found and removed, `false` if
    /// it was not registered (which may indicate a double-destroy bug).
    pub fn unregister(&self, object_type: vk::ObjectType, handle: u64) -> bool {
        let key = ObjectKey {
            object_type,
            handle,
        };
        self.objects.lock().unwrap().remove(&key).is_some()
    }

    /// Set or update the debug name of a tracked object.
    pub fn set_name(&self, object_type: vk::ObjectType, handle: u64, name: &str) {
        let key = ObjectKey {
            object_type,
            handle,
        };
        if let Some(obj) = self.objects.lock().unwrap().get_mut(&key) {
            obj.name = Some(name.to_string());
        }
    }

    /// Record that an object was used (e.g., bound to a command buffer).
    pub fn record_usage(&self, object_type: vk::ObjectType, handle: u64) {
        let key = ObjectKey {
            object_type,
            handle,
        };
        if let Some(obj) = self.objects.lock().unwrap().get(&key) {
            obj.usage_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of currently tracked (live) objects.
    pub fn live_count(&self) -> usize {
        self.objects.lock().unwrap().len()
    }

    /// Number of live objects of a specific type.
    pub fn live_count_of(&self, object_type: vk::ObjectType) -> usize {
        self.objects
            .lock()
            .unwrap()
            .values()
            .filter(|o| o.object_type == object_type)
            .count()
    }

    /// Check whether a specific object is currently tracked.
    pub fn is_alive(&self, object_type: vk::ObjectType, handle: u64) -> bool {
        let key = ObjectKey {
            object_type,
            handle,
        };
        self.objects.lock().unwrap().contains_key(&key)
    }

    /// Generate a leak report for all currently live objects.
    ///
    /// Returns `None` if there are no live objects.
    pub fn report_leaks(&self) -> Option<String> {
        let map = self.objects.lock().unwrap();
        if map.is_empty() {
            return None;
        }

        let mut entries: Vec<&TrackedObject> = map.values().collect();
        entries.sort_by_key(|o| std::cmp::Reverse(o.created_at));

        Some(format_leak_report(&entries))
    }
}

impl Default for LifetimeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LifetimeTracker {
    fn drop(&mut self) {
        let map = self.objects.get_mut().unwrap();
        if map.is_empty() {
            return;
        }

        let mut entries: Vec<&TrackedObject> = map.values().collect();
        entries.sort_by_key(|o| std::cmp::Reverse(o.created_at));
        let report = format_leak_report(&entries);

        match &self.on_leak {
            LeakAction::Log => eprint!("{report}"),
            LeakAction::Panic => panic!("{report}"),
            LeakAction::Callback(f) => f(&report),
            LeakAction::Ignore => {}
        }
    }
}

fn format_leak_report(objects: &[&TrackedObject]) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(512 + objects.len() * 300);

    diagnostic::write_header(
        &mut o,
        &s,
        &Severity::Warning,
        "IGN-L001",
        &format!("{} Vulkan object(s) leaked", objects.len()),
    );
    diagnostic::write_location(&mut o, &s, "LifetimeTracker shutdown");
    diagnostic::write_pipe_empty(&mut o, &s);

    // Group by object type for summary.
    let mut by_type: HashMap<vk::ObjectType, Vec<&&TrackedObject>> = HashMap::new();
    for obj in objects {
        by_type.entry(obj.object_type).or_default().push(obj);
    }

    // Type summary with counts.
    let mut type_summary: Vec<String> = by_type
        .iter()
        .map(|(ty, objs)| format!("{}× {}", objs.len(), diagnostic::object_type_name(*ty)))
        .collect();
    type_summary.sort();
    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!("summary: {}", type_summary.join(", ")),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    // Detailed per-object entries.
    for (i, obj) in objects.iter().enumerate() {
        let type_name = diagnostic::object_type_name(obj.object_type);
        let name_str = obj
            .name
            .as_deref()
            .map(|n| format!(" \"{}\"", s.bold_cyan(n)))
            .unwrap_or_default();

        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "{} {type_name}({:#x}){name_str}",
                s.dim(&format!("[{i}]")),
                obj.handle,
            ),
        );

        let loc = format!(
            "{}:{}:{}",
            obj.caller_file, obj.caller_line, obj.caller_column
        );
        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!("     created at {}", s.underline(&loc)),
        );

        let age = diagnostic::format_duration(obj.created_at.elapsed());
        let uses = obj.usage_count.load(Ordering::Relaxed);
        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!("     alive={age}, bound {uses} time(s)"),
        );

        if uses == 0 {
            diagnostic::write_pipe(
                &mut o,
                &s,
                &format!(
                    "     {} never used — likely orphaned (created but never bound/submitted)",
                    s.bold_yellow("⚠")
                ),
            );
        }

        diagnostic::write_pipe_empty(&mut o, &s);
    }

    // Count never-used objects for extra warning.
    let unused_count = objects
        .iter()
        .filter(|o| o.usage_count.load(Ordering::Relaxed) == 0)
        .count();
    if unused_count > 0 {
        diagnostic::write_warn(
            &mut o,
            &s,
            &format!(
                "{unused_count} of {} leaked objects were never used — these are likely\n\
                 created in error or in a code path that forgot to clean up",
                objects.len()
            ),
        );
    }

    diagnostic::write_note(
        &mut o,
        &s,
        "leaked objects consume device memory until process exit\n\
         on resource-constrained GPUs this can cause allocation failures",
    );
    diagnostic::write_help(
        &mut o,
        &s,
        "ensure all objects are dropped before the Ignis context\n\
         store objects in containers that implement Drop\n\
         use DeletionQueue for deferred cleanup tied to GPU completion",
    );

    diagnostic::write_diagnostic_end(&mut o, &s, &Severity::Warning);

    o
}