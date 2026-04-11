//! GPU hang detection with breadcrumb trail.
//!
//! [`HangDetector`] runs a watchdog thread that monitors submitted fences.
//! If any fence fails to signal within a configurable timeout, a rich
//! diagnostic is produced showing the breadcrumb trail of completed
//! operations.
//!
//! [`BreadcrumbBuffer`] allocates a small CPU-visible GPU buffer and
//! provides methods to insert marker writes into command buffers.
//! After a hang, reading back the buffer reveals which operations
//! completed before the hang occurred.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ash::vk;

use crate::device::SharedState;
use crate::diagnostic::{self, Severity, Style};
use crate::error::{Error, Result};

/// Configuration for the hang detector.
#[derive(Debug, Clone)]
pub struct HangConfig {
    /// How long a fence may remain unsignaled before declaring a hang.
    /// Default: 5 seconds.
    pub timeout: Duration,
    /// How often the watchdog thread checks fence status.
    /// Default: 100 milliseconds.
    pub check_interval: Duration,
}

impl Default for HangConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            check_interval: Duration::from_millis(100),
        }
    }
}

/// A fence being monitored by the hang detector.
struct WatchedSubmission {
    fence: vk::Fence,
    label: String,
    submitted_at: Instant,
    breadcrumbs: Option<Arc<BreadcrumbBuffer>>,
    reported: bool,
}

/// Action taken when a hang is detected.
#[derive(Default)]
pub enum HangAction {
    /// Print report to stderr.
    #[default]
    Log,
    /// Panic with the full report.
    Panic,
    /// Custom callback.
    Callback(Box<dyn Fn(&str) + Send + Sync>),
}


impl std::fmt::Debug for HangAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Log => write!(f, "Log"),
            Self::Panic => write!(f, "Panic"),
            Self::Callback(_) => write!(f, "Callback(...)"),
        }
    }
}

/// Background watchdog that detects GPU hangs.
///
/// Monitors fences registered via [`watch`](HangDetector::watch) and
/// produces a diagnostic report if any fence exceeds the configured
/// timeout.
pub struct HangDetector {
    #[allow(dead_code)]
    shared: Arc<SharedState>,
    submissions: Arc<Mutex<Vec<WatchedSubmission>>>,
    config: HangConfig,
    #[allow(dead_code)]
    on_hang: Arc<HangAction>,
    shutdown: Arc<AtomicBool>,
    notify: Arc<NotifyPair>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

struct NotifyPair {
    flag: Mutex<bool>,
    cvar: Condvar,
}

impl NotifyPair {
    fn new() -> Self {
        Self {
            flag: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    fn signal(&self) {
        let mut f = self.flag.lock().unwrap();
        *f = true;
        self.cvar.notify_one();
    }

    fn wait(&self, timeout: Duration) {
        let f = self.flag.lock().unwrap();
        let (mut f, _) = self.cvar.wait_timeout(f, timeout).unwrap();
        *f = false;
    }
}

impl HangDetector {
    /// Create and start a hang detector.
    pub fn new(shared: Arc<SharedState>, config: HangConfig, on_hang: HangAction) -> Self {
        let submissions: Arc<Mutex<Vec<WatchedSubmission>>> = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(NotifyPair::new());
        let on_hang = Arc::new(on_hang);

        let t_shared = Arc::clone(&shared);
        let t_subs = Arc::clone(&submissions);
        let t_shutdown = Arc::clone(&shutdown);
        let t_notify = Arc::clone(&notify);
        let t_config = config.clone();
        let t_action = Arc::clone(&on_hang);

        let handle = std::thread::Builder::new()
            .name("ignis-hang-detector".into())
            .spawn(move || {
                Self::watchdog_loop(
                    &t_shared,
                    &t_subs,
                    &t_shutdown,
                    &t_notify,
                    &t_config,
                    &t_action,
                );
            })
            .expect("failed to spawn hang detector thread");

        Self {
            shared,
            submissions,
            config,
            on_hang,
            shutdown,
            notify,
            handle: Mutex::new(Some(handle)),
        }
    }

    /// Register a fence for hang monitoring.
    ///
    /// If provided, `breadcrumbs` will be read back on hang to determine
    /// the last completed operation.
    pub fn watch(
        &self,
        fence: vk::Fence,
        label: &str,
        breadcrumbs: Option<&Arc<BreadcrumbBuffer>>,
    ) {
        let entry = WatchedSubmission {
            fence,
            label: label.to_string(),
            submitted_at: Instant::now(),
            breadcrumbs: breadcrumbs.cloned(),
            reported: false,
        };
        self.submissions.lock().unwrap().push(entry);
        self.notify.signal();
    }

    /// Number of fences currently being monitored.
    pub fn watched_count(&self) -> usize {
        self.submissions.lock().unwrap().len()
    }

    fn watchdog_loop(
        shared: &SharedState,
        submissions: &Mutex<Vec<WatchedSubmission>>,
        shutdown: &AtomicBool,
        notify: &NotifyPair,
        config: &HangConfig,
        action: &HangAction,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let mut subs = submissions.lock().unwrap();

            // Check each fence.
            let now = Instant::now();
            for sub in subs.iter_mut() {
                if sub.reported {
                    continue;
                }

                let signaled =
                    unsafe { shared.device.get_fence_status(sub.fence) }.unwrap_or(false);

                if signaled {
                    continue;
                }

                let elapsed = now.duration_since(sub.submitted_at);
                if elapsed >= config.timeout {
                    sub.reported = true;

                    // Read breadcrumbs if available.
                    let crumbs = sub.breadcrumbs.as_ref().map(|b| b.readback());

                    let report =
                        format_hang_report(&sub.label, sub.fence, elapsed, crumbs.as_deref());

                    match action {
                        HangAction::Log => eprint!("{report}"),
                        HangAction::Panic => {
                            drop(subs);
                            panic!("{report}");
                        }
                        HangAction::Callback(f) => f(&report),
                    }
                }
            }

            // Prune signaled fences.
            subs.retain(|sub| {
                if sub.reported {
                    return false;
                }
                let signaled =
                    unsafe { shared.device.get_fence_status(sub.fence) }.unwrap_or(false);
                !signaled
            });

            drop(subs);

            if !shutdown.load(Ordering::Relaxed) {
                notify.wait(config.check_interval);
            }
        }
    }

    /// Get the current hang detection configuration.
    pub fn config(&self) -> &HangConfig {
        &self.config
    }
}

impl Drop for HangDetector {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.notify.signal();
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

/// A breadcrumb entry describing one GPU operation marker.
#[derive(Debug, Clone)]
pub struct Breadcrumb {
    /// Sequential ID (1-based).
    pub id: u32,
    /// Human-readable label.
    pub label: String,
}

/// CPU-visible GPU buffer for breadcrumb markers.
///
/// Insert breadcrumbs into command buffers via
/// [`insert`](BreadcrumbBuffer::insert). After a hang,
/// [`readback`](BreadcrumbBuffer::readback) returns the trail
/// of completed operations.
///
/// The buffer stores a single u32 counter. Each breadcrumb writes
/// its ID to offset 0 via `vkCmdFillBuffer`. Since commands execute
/// in order on a single queue, the final value is the last completed
/// breadcrumb.
pub struct BreadcrumbBuffer {
    shared: Arc<SharedState>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u32,
    entries: Mutex<Vec<Breadcrumb>>,
    next_id: Mutex<u32>,
}

// SAFETY: the mapped pointer points to persistently mapped Vulkan memory.
// Access is synchronized by Mutex guards on the entry list.
unsafe impl Send for BreadcrumbBuffer {}
unsafe impl Sync for BreadcrumbBuffer {}

impl BreadcrumbBuffer {
    /// Create a new breadcrumb buffer.
    pub fn new(shared: Arc<SharedState>) -> Result<Self> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(4)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { shared.device.create_buffer(&buffer_ci, None)? };
        let mem_req = unsafe { shared.device.get_buffer_memory_requirements(buffer) };

        let mem_type =
            find_host_visible_memory(&shared, &mem_req).ok_or(Error::NoSuitableMemoryType)?;

        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);

        let memory = unsafe { shared.device.allocate_memory(&alloc_ci, None) }.map_err(|e| {
            unsafe { shared.device.destroy_buffer(buffer, None) };
            Error::Vulkan(e)
        })?;

        unsafe { shared.device.bind_buffer_memory(buffer, memory, 0)? };

        let ptr = unsafe {
            shared
                .device
                .map_memory(memory, 0, 4, vk::MemoryMapFlags::empty())?
        }.cast::<u32>();

        // Initialize to zero.
        unsafe { ptr.write(0) };

        Ok(Self {
            shared,
            buffer,
            memory,
            mapped: ptr,
            entries: Mutex::new(Vec::new()),
            next_id: Mutex::new(1),
        })
    }

    /// Insert a breadcrumb marker into a command buffer.
    ///
    /// Must be called outside a render pass (vkCmdFillBuffer restriction).
    /// Returns the breadcrumb ID assigned.
    pub fn insert(&self, device: &ash::Device, cmd: vk::CommandBuffer, label: &str) -> u32 {
        let mut next = self.next_id.lock().unwrap();
        let id = *next;
        *next += 1;

        self.entries.lock().unwrap().push(Breadcrumb {
            id,
            label: label.to_string(),
        });

        // SAFETY: buffer is valid, offset 0, size 4, data = id.
        unsafe {
            device.cmd_fill_buffer(cmd, self.buffer, 0, 4, id);
        }

        id
    }

    /// Read back the breadcrumb buffer and return the trail.
    ///
    /// Each entry is marked as completed or pending based on whether
    /// its ID is <= the last written value.
    pub fn readback(&self) -> Vec<(Breadcrumb, bool)> {
        let last_value = unsafe { self.mapped.read_volatile() };
        let entries = self.entries.lock().unwrap();

        entries
            .iter()
            .map(|b| (b.clone(), b.id <= last_value))
            .collect()
    }

    /// Reset the buffer and entry list.
    pub fn reset(&self) {
        unsafe { self.mapped.write_volatile(0) };
        self.entries.lock().unwrap().clear();
        *self.next_id.lock().unwrap() = 1;
    }

    /// Raw buffer handle (for barrier insertion if needed).
    pub fn buffer_handle(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for BreadcrumbBuffer {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.unmap_memory(self.memory);
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

fn find_host_visible_memory(shared: &SharedState, req: &vk::MemoryRequirements) -> Option<u32> {
    let props = &shared.memory_properties;
    for i in 0..props.memory_type_count {
        if req.memory_type_bits & (1 << i) == 0 {
            continue;
        }
        let flags = props.memory_types[i as usize].property_flags;
        if flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            return Some(i);
        }
    }
    None
}

fn format_hang_report(
    label: &str,
    fence: vk::Fence,
    elapsed: Duration,
    crumbs: Option<&[(Breadcrumb, bool)]>,
) -> String {
    use ash::vk::Handle;
    let s = Style::detect();
    let mut o = String::with_capacity(2048);

    diagnostic::write_header(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-W001",
        &format!(
            "GPU hang detected (fence timeout after {})",
            diagnostic::format_duration(elapsed)
        ),
    );
    diagnostic::write_location(
        &mut o,
        &s,
        &format!("submission \"{}\" fence={:#x}", label, fence.as_raw()),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    if let Some(trail) = crumbs {
        if trail.is_empty() {
            diagnostic::write_pipe(&mut o, &s, "no breadcrumbs recorded");
        } else {
            diagnostic::write_pipe(&mut o, &s, "breadcrumb trail:");
            diagnostic::write_pipe_empty(&mut o, &s);

            let last_completed = trail.iter().rev().find(|(_, done)| *done);
            let first_pending = trail.iter().find(|(_, done)| !*done);

            for (crumb, completed) in trail {
                let marker = if *completed {
                    s.green("OK")
                } else if Some(&crumb.id) == first_pending.map(|(c, _)| &c.id) {
                    s.bold_red("--> HUNG")
                } else {
                    s.dim("PENDING")
                };

                diagnostic::write_pipe(
                    &mut o,
                    &s,
                    &format!("  #{:<4} {:40} {}", crumb.id, crumb.label, marker,),
                );
            }
            diagnostic::write_pipe_empty(&mut o, &s);

            if let Some((last, _)) = last_completed {
                diagnostic::write_note(
                    &mut o,
                    &s,
                    &format!("last completed breadcrumb: #{} \"{}\"", last.id, last.label),
                );
            }
            if let Some((first, _)) = first_pending {
                diagnostic::write_note(
                    &mut o,
                    &s,
                    &format!(
                        "first pending breadcrumb: #{} \"{}\"",
                        first.id, first.label
                    ),
                );
            }
        }
    } else {
        diagnostic::write_pipe(
            &mut o,
            &s,
            "no breadcrumb buffer attached to this submission",
        );
    }

    diagnostic::write_pipe_empty(&mut o, &s);
    diagnostic::write_note(
        &mut o,
        &s,
        &format!("thread=\"{}\"", diagnostic::current_thread_name()),
    );
    diagnostic::write_help(
        &mut o,
        &s,
        "common causes: infinite shader loop, excessive draw, driver bug\n\
         check the hung operation's shader for infinite loops\n\
         try reducing dispatch/draw dimensions",
    );

    o
}
