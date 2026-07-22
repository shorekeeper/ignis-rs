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

/// `VK_EXT_debug_utils` integration for object naming and command labels.
pub mod debug_utils;

/// GPU timestamp profiler with query pool management.
pub mod profiler;

/// `VK_EXT_debug_printf` integration with shader message routing.
pub mod shader_printf;

/// Validation layer bridging through the diagnostic formatter.
pub mod validation;

/// Automatic crash report generation on device lost.
pub mod crash_report;

/// Pipeline statistics queries (vertex/fragment/compute invocations, clipping, etc).
pub mod pipeline_stats;

/// Forensic analysis of validation layer messages with VUID knowledge base.
pub mod validation_forensic;

/// VUID knowledge base (static + runtime entries).
pub mod vuid_kb;

/// VL diagnostic pipeline (filters, sinks, actions, scopes, capture).
pub mod vl_pipeline;

/// Allocation site profiler (heaptrack for GPU memory).
pub mod alloc_profiler;

/// Memory layout SVG visualizer.
pub mod memory_viz;

/// Device fault diagnostics: NV checkpoints, EXT fault info, AMD markers.
pub mod device_fault;

/// Validation Layer baseline capture and CI-grade diff.
pub mod vl_baseline;

/// GPU determinism verifier with hash-based output comparison.
pub mod determinism;

/// Cross-queue submission tracker with cycle and orphan detection.
pub mod cross_queue;

/// Sync DAG visualizer (DOT, Mermaid, SVG).
pub mod sync_dag_viz;

/// Shared rasterizer primitives (framebuffer, font, BMP encoder).
pub mod raster_common;