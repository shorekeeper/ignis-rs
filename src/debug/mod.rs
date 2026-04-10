//! Debug and validation toolkit.
//!
//! Available when the `debug-tools` feature is enabled.
//! Contains 12 modules for development-time bug detection:
//!
//! | Module | What it catches |
//! |---|---|
//! | [`hardened`] | Buffer overflow, underflow, use-after-free, double-free |
//! | [`lifetime`] | Leaked Vulkan objects with creation-site tracking |
//! | [`cmd_state`] | Draw outside render pass, dispatch inside render pass |
//! | [`hang_detector`] | GPU hangs with breadcrumb trail |
//! | [`journal`] | Submission flight recorder for device-lost debugging |
//! | [`thread_audit`] | Command pool cross-thread access |
//! | [`aliasing`] | Read-after-write without barrier |
//! | [`barrier_opt`] | Overly broad ALL_COMMANDS barriers |
//! | [`budget`] | Memory heap exhaustion warnings |
//! | [`descriptor_audit`] | Stale descriptor references to destroyed resources |
//! | [`pipeline_audit`] | Pipeline/descriptor set layout mismatches |

/// Guard-band hardened allocator with canary verification.
pub mod hardened;

/// Object lifetime tracking with `#[track_caller]` creation sites.
pub mod lifetime;

/// Command buffer recording state machine validator.
pub mod cmd_state;

/// GPU hang detector with breadcrumb buffer support.
pub mod hang_detector;

/// Submission flight recorder (black box).
pub mod journal;

/// Command pool thread safety auditor.
pub mod thread_audit;

/// Resource aliasing detector (missing barriers).
pub mod aliasing;

/// Pipeline barrier analyzer and optimizer.
pub mod barrier_opt;

/// GPU memory budget monitor.
pub mod budget;

/// Descriptor set validator (stale resource references).
pub mod descriptor_audit;

/// Pipeline compatibility checker (layout mismatches).
pub mod pipeline_audit;