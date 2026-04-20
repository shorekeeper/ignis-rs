//! Validation layer message routing.
//!
//! Installs a `VK_EXT_debug_utils` messenger that routes every validation
//! layer message through the diagnostic formatter used by the rest of the
//! crate. Shader printf messages are detected and dispatched to the
//! printf handler instead.
//!
//! This replaces the default validation behavior of writing raw ANSI-less
//! text to stderr with a consistent formatted stream matching ignis
//! diagnostics.

use std::ffi::{c_void, CStr};

use ash::vk;

use crate::diagnostic::{self, Severity, Style};
use super::validation_forensic::{self, LayerSeverity};
use super::shader_printf::{self, PRINTF_REGISTRY};

/// Severity mapping policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Print every message through the diagnostic formatter.
    FormatAll,
    /// Only format errors, pass everything else through unchanged.
    ErrorsOnly,
    /// Format errors and warnings, suppress informational messages.
    /// This is the default as informational messages from the validation
    /// layer include both genuinely useful context (initialization notices,
    /// feature adjustments) and routine noise that drowns out real issues
    /// when a printf-enabled session emits hundreds per second.
    ErrorsAndWarnings,
    /// Suppress informational messages entirely.
    DropInfo,
}

impl Default for ValidationPolicy {
    fn default() -> Self {
        Self::ErrorsAndWarnings
    }
}

static POLICY: std::sync::Mutex<ValidationPolicy> =
    std::sync::Mutex::new(ValidationPolicy::DropInfo);

/// Configure how validation messages are filtered and formatted.
pub fn set_policy(policy: ValidationPolicy) {
    *POLICY.lock().unwrap() = policy;
}

/// Get the current validation policy.
pub fn policy() -> ValidationPolicy {
    *POLICY.lock().unwrap()
}

/// The `PFN_vkDebugUtilsMessengerCallbackEXT` that routes all validation
/// messages. Safe to install via `VkDebugUtilsMessengerCreateInfoEXT`.
///
/// # Safety
///
/// Pointer parameters must follow the Vulkan specification. The callback
/// reads them read-only and does not retain them past the call.
pub unsafe extern "system" fn debug_utils_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    if callback_data.is_null() {
        return vk::FALSE;
    }
    let data = &*callback_data;

    let raw_msg = if data.p_message.is_null() {
        ""
    } else {
        CStr::from_ptr(data.p_message).to_str().unwrap_or("")
    };

    let msg_id_name = if data.p_message_id_name.is_null() {
        ""
    } else {
        CStr::from_ptr(data.p_message_id_name)
            .to_str()
            .unwrap_or("")
    };

    // First: is this a shader printf payload? Detect via both the
    // message id name and the raw text so the parser works across SDK
    // versions (old: UNASSIGNED-DEBUG-PRINTF, new: WARNING-DEBUG-PRINTF).
    if let Some(parsed) =
        shader_printf::try_parse_printf(raw_msg, data.message_id_number, msg_id_name)
    {
        PRINTF_REGISTRY.dispatch(&parsed);
        // Printf messages do not get forwarded to diagnostic formatter;
        // they have their own handler.
        return vk::FALSE;
    }

    // Apply policy filter.
    let policy = *POLICY.lock().unwrap();
    let is_error = severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR);
    let is_warning = severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING);
    let is_info = severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO);

    match policy {
        ValidationPolicy::DropInfo if is_info => return vk::FALSE,
        ValidationPolicy::ErrorsAndWarnings if is_info => return vk::FALSE,
        ValidationPolicy::ErrorsOnly if !is_error => {
            eprintln!("[validation] {raw_msg}");
            return vk::FALSE;
        }
        _ => {}
    }

    let layer_severity = if is_error {
        LayerSeverity::Error
    } else if is_warning {
        LayerSeverity::Warning
    } else {
        LayerSeverity::Info
    };

    // Try forensic parse first. When it succeeds, we get a rich structured
    // diagnostic with objects resolved, knowledge base explanation, and the
    // submit backtrace if captured. Falls through to generic formatting
    // when the message does not look like a VUID violation.
    if let Some(forensic) =
        validation_forensic::parse_validation_message(raw_msg, msg_id_name, layer_severity)
    {
        let formatted = validation_forensic::format_forensic_diagnostic(&forensic);
        eprint!("{formatted}");
        validation_forensic::dispatch_to_handler(&forensic);
        return vk::FALSE;
    }

    // Fallback: generic framed formatting for non-VUID messages
    // (layer settings notices, WARNING-CreateInstance-status-message, etc).
    let sev = if is_error {
        Severity::Error
    } else if is_warning {
        Severity::Warning
    } else {
        Severity::Info
    };

    let s = Style::detect();
    let mut o = String::with_capacity(raw_msg.len() + 256);

    let type_label = if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "validation"
    } else if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "performance"
    } else {
        "general"
    };

    diagnostic::write_header(
        &mut o,
        &s,
        &sev,
        "IGN-V001",
        &format!(
            "{type_label}: {}",
            if msg_id_name.is_empty() {
                "UNKNOWN"
            } else {
                msg_id_name
            }
        ),
    );
    diagnostic::write_pipe_empty(&mut o, &s);
    for line in raw_msg.lines() {
        diagnostic::write_pipe(&mut o, &s, line);
    }
    diagnostic::write_diagnostic_end(&mut o, &s, &sev);

    eprint!("{o}");
    vk::FALSE
}

/// Create and install a debug utils messenger on the given instance.
///
/// Returns the handle so it can be destroyed when the instance is torn down.
///
/// # Errors
///
/// Returns a Vulkan error if the messenger cannot be created (usually
/// because `VK_EXT_debug_utils` is not enabled).
pub fn install_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> crate::Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
    let du = ash::ext::debug_utils::Instance::new(entry, instance);

    let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_utils_callback));

    let handle = unsafe { du.create_debug_utils_messenger(&info, None)? };
    Ok((du, handle))
}