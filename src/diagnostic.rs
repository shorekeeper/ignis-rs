//! Shared diagnostic formatting primitives for all ignis debug modules.
//!
//! Provides ANSI terminal styling, structured report builders, and
//! helper functions used by every debugging subsystem.

use std::fmt::Write;
use std::time::Duration;

/// ANSI terminal style controller.
///
/// Respects the `NO_COLOR` environment variable (<https://no-color.org/>).
pub(crate) struct Style {
    pub on: bool,
}

impl Style {
    pub fn detect() -> Self {
        Self {
            on: std::env::var_os("NO_COLOR").is_none(),
        }
    }

    fn esc(&self, code: &str, text: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    pub fn bold_red(&self, t: &str) -> String {
        self.esc("1;31", t)
    }
    pub fn bold_yellow(&self, t: &str) -> String {
        self.esc("1;33", t)
    }
    pub fn bold_green(&self, t: &str) -> String {
        self.esc("1;32", t)
    }
    pub fn bold_cyan(&self, t: &str) -> String {
        self.esc("1;36", t)
    }
    #[allow(dead_code)]
    pub fn bold_magenta(&self, t: &str) -> String {
        self.esc("1;35", t)
    }
    pub fn bold(&self, t: &str) -> String {
        self.esc("1", t)
    }
    pub fn blue(&self, t: &str) -> String {
        self.esc("34", t)
    }
    pub fn red(&self, t: &str) -> String {
        self.esc("31", t)
    }
    pub fn green(&self, t: &str) -> String {
        self.esc("32", t)
    }
    pub fn yellow(&self, t: &str) -> String {
        self.esc("33", t)
    }
    pub fn dim(&self, t: &str) -> String {
        self.esc("2", t)
    }
    #[allow(dead_code)]
    pub fn cyan(&self, t: &str) -> String {
        self.esc("36", t)
    }
    pub fn underline(&self, t: &str) -> String {
        self.esc("4", t)
    }
}

/// Severity level for a diagnostic.
pub(crate) enum Severity {
    Error,
    Warning,
    Info,
}

impl Severity {
    pub fn label(&self, s: &Style) -> String {
        match self {
            Severity::Error => s.bold_red("error"),
            Severity::Warning => s.bold_yellow("warning"),
            Severity::Info => s.bold_cyan("info"),
        }
    }
}

/// Write a note line in the standard format.
pub(crate) fn write_note(o: &mut String, s: &Style, text: &str) {
    let label = format!("   {} {}: ", s.bold_cyan("="), s.bold("note"));
    write_labeled(o, &label, text);
}

/// Write a help line in the standard format.
pub(crate) fn write_help(o: &mut String, s: &Style, text: &str) {
    let label = format!("   {} {}: ", s.bold_green("="), s.bold("help"));
    write_labeled(o, &label, text);
}

/// Write a warning note.
pub(crate) fn write_warn(o: &mut String, s: &Style, text: &str) {
    let label = format!("   {} {}: ", s.bold_yellow("="), s.bold("warn"));
    write_labeled(o, &label, text);
}

fn write_labeled(o: &mut String, label: &str, text: &str) {
    let lines: Vec<&str> = text.lines().collect();
    if let Some((first, rest)) = lines.split_first() {
        let _ = writeln!(o, "{label}{first}");
        let indent: String = " ".repeat(strip_ansi_len(label));
        for line in rest {
            let _ = writeln!(o, "{indent}{line}");
        }
    }
}

/// Build a diagnostic header: `error[CODE]: message`
pub(crate) fn write_header(o: &mut String, s: &Style, sev: &Severity, code: &str, msg: &str) {
    let _ = writeln!(o, "{}{}: {msg}", sev.label(s), s.bold(&format!("[{code}]")));
}

/// Build a location arrow: `  --> location_text`
pub(crate) fn write_location(o: &mut String, s: &Style, location: &str) {
    let _ = writeln!(o, "  {} {location}", s.blue("-->"));
}

/// Write an empty pipe line.
pub(crate) fn write_pipe_empty(o: &mut String, s: &Style) {
    let _ = writeln!(o, "   {}", s.blue("|"));
}

/// Write a pipe line with content.
pub(crate) fn write_pipe(o: &mut String, s: &Style, text: &str) {
    let _ = writeln!(o, "   {}  {text}", s.blue("|"));
}

/// Compute visible character count ignoring ANSI escape sequences.
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

/// Format bytes as hex: "d8 08 96 f8".
pub(crate) fn hex_line(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build diff markers: `^^` under each differing byte pair.
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

/// Format a Duration compactly: "142.3us", "3.21ms", "5.02s".
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

/// Pad a label centered within `width` chars using `fill`.
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

/// Format a Vulkan object type as a readable string.
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

/// Get the current thread name or "<unnamed>".
pub(crate) fn current_thread_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_string()
}

// Below: existing hardened-allocator-specific formatters.
// Kept as-is but now using the pub(crate) helpers above.

pub(crate) struct GuardReport {
    pub code: &'static str,
    pub severity: Severity,
    pub region: &'static str,
    pub memory_handle: u64,
    pub user_offset: u64,
    pub user_size: u64,
    pub guard_size: u64,
    pub first_corrupted: usize,
    pub total_corrupted: usize,
    pub canary: u64,
    pub expected_byte: u8,
    pub actual_byte: u8,
    pub source: &'static str,
    pub age: Option<Duration>,
    pub thread: String,
    pub hex_offset: usize,
    pub hex_expected: Vec<u8>,
    pub hex_actual: Vec<u8>,
}

pub(crate) struct LeakEntry {
    pub memory_handle: u64,
    pub user_offset: u64,
    pub user_size: u64,
    pub age: Duration,
}

pub(crate) fn format_guard_report(r: &GuardReport) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(2048);

    write_header(
        &mut o,
        &s,
        &r.severity,
        r.code,
        &format!("{} guard band corruption", r.region),
    );
    write_location(
        &mut o,
        &s,
        &format!(
            "VkDeviceMemory({:#x}) offset={} size={}B",
            r.memory_handle, r.user_offset, r.user_size
        ),
    );
    write_pipe_empty(&mut o, &s);

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
    write_pipe(
        &mut o,
        &s,
        &format!(
            "{pad}{}",
            s.bold_red(&format!("^-- byte {}", r.first_corrupted))
        ),
    );

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
    write_pipe(
        &mut o,
        &s,
        &format!(" {} {expected_hex}", s.green("expect:")),
    );
    write_pipe(&mut o, &s, &format!(" {} {actual_hex}", s.red("actual:")));

    let markers = diff_markers(&r.hex_expected, &r.hex_actual);
    if markers.contains('^') {
        let marker_pad = " ".repeat("actual: ".len() + 1);
        write_pipe(&mut o, &s, &format!("{marker_pad}{}", s.bold_red(&markers)));
    }

    write_pipe_empty(&mut o, &s);

    // Concrete byte values at first corruption site.
    write_pipe(
        &mut o,
        &s,
        &format!(
            "at byte {}: expected {}, found {}",
            r.first_corrupted,
            s.green(&format!("{:#04x}", r.expected_byte)),
            s.bold_red(&format!("{:#04x}", r.actual_byte)),
        ),
    );

    write_pipe_empty(&mut o, &s);

    let pct = (r.total_corrupted as f64 / r.guard_size as f64) * 100.0;
    write_note(
        &mut o,
        &s,
        &format!(
            "{}/{} {} guard bytes corrupted ({pct:.1}%)",
            r.total_corrupted, r.guard_size, r.region
        ),
    );
    write_note(&mut o, &s, &format!("canary={:#018x}", r.canary));

    match r.age {
        Some(age) => write_note(
            &mut o,
            &s,
            &format!("alive={} thread=\"{}\"", format_duration(age), r.thread),
        ),
        None => write_note(&mut o, &s, &format!("thread=\"{}\"", r.thread)),
    }
    write_note(&mut o, &s, &format!("detected during {}", r.source));

    let suggestion = corruption_suggestion(r.region, r.first_corrupted, r.guard_size as usize);
    write_help(&mut o, &s, &suggestion);

    o
}

pub(crate) fn format_double_free(memory_handle: u64, offset: u64, size: u64) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(512);

    write_header(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-H003",
        "invalid free (allocation not found)",
    );
    write_location(
        &mut o,
        &s,
        &format!("VkDeviceMemory({memory_handle:#x}) offset={offset} size={size}B"),
    );
    write_pipe_empty(&mut o, &s);
    write_note(&mut o, &s, "allocation not found in tracking table");
    write_note(&mut o, &s, &format!("thread=\"{}\"", current_thread_name()));
    write_help(
        &mut o,
        &s,
        "possible causes: double free, or freeing memory\nfrom a different allocator instance",
    );

    o
}

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

    for (i, e) in entries.iter().enumerate() {
        write_pipe(
            &mut o,
            &s,
            &format!(
                "{} VkDeviceMemory({:#x}) offset={} size={}B  alive={}",
                s.dim(&format!("[{i}]")),
                e.memory_handle,
                e.user_offset,
                e.user_size,
                format_duration(e.age),
            ),
        );
    }

    write_pipe_empty(&mut o, &s);
    write_note(
        &mut o,
        &s,
        "leaking GPU memory can exhaust device-local heaps",
    );
    write_help(
        &mut o,
        &s,
        "ensure all Buffers and Images are dropped before\nthe allocator is destroyed",
    );

    o
}

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
             typically indicates buffer underflow: write before offset 0"
        ),
        ("front", _, true) => format!(
            "byte {byte}/{guard_size} of front guard (far from user data)\n\
             may indicate wild pointer or large negative offset"
        ),
        ("front", _, _) => format!(
            "byte {byte}/{guard_size} of front guard\n\
             may indicate wild pointer or substantial underflow"
        ),
        ("back", true, _) => format!(
            "byte {byte}/{guard_size} of back guard (boundary with user data)\n\
             typically indicates buffer overflow: write past allocation end"
        ),
        ("back", _, true) => format!(
            "byte {byte}/{guard_size} of back guard (far from user data)\n\
             may indicate wild pointer or large overflow"
        ),
        ("back", _, _) => format!(
            "byte {byte}/{guard_size} of back guard\n\
             may indicate wild pointer or substantial overflow"
        ),
        _ => String::new(),
    }
}
