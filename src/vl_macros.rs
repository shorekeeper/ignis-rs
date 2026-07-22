//! Declarative macros for configuring the VL forensic pipeline.
//!
//! All macros require the `debug-tools` feature. Without it, they either
//! expand to no-ops (for void-returning macros) or produce a compile error
//! that points at the missing feature.

/// Configure the global VL pipeline declaratively.
///
/// Each line is a statement ending with `;`. Supported statements:
///
/// ```text
/// suppress "VUID-*-01234";
/// suppress category ObjectLifetime;
/// suppress function "vkQueueSubmit";
/// escalate "VUID-*-00067" => error;
/// escalate category SynchronizationHazard => error;
/// demote "MY-DRIVER-BUG-*" => warning;
/// action error => panic;
/// action warning => log;
/// sink stderr;
/// sink file "validation.log";
/// dedup per_vuid 10;
/// dedup global 1000;
/// dedup off;
/// backtrace errors_only;
/// breakpoint "VUID-*-00067";
/// no_stderr;
/// no_legacy_forward;
/// ```
///
/// # Requires
///
/// The `debug-tools` feature must be enabled.
#[macro_export]
macro_rules! ignis_vl {
    ($($body:tt)*) => {{
        let mut __b = $crate::debug::vl_pipeline::VlConfigBuilder::new();
        $crate::__ignis_vl_rec!(__b; $($body)*);
        __b.install();
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __ignis_vl_rec {
    // Base case: empty input.
    ($b:ident;) => {};

    // suppress "pattern";
    ($b:ident; suppress $pat:literal ; $($rest:tt)*) => {
        $b = $b.suppress_vuid($pat);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // suppress category X;
    ($b:ident; suppress category $cat:ident ; $($rest:tt)*) => {
        $b = $b.suppress_category(
            $crate::debug::validation_forensic::DiagnosticCategory::$cat
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // suppress function "pattern";
    ($b:ident; suppress function $pat:literal ; $($rest:tt)*) => {
        $b = $b.suppress_function($pat);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // escalate "pat" => severity;
    ($b:ident; escalate $pat:literal => error ; $($rest:tt)*) => {
        $b = $b.escalate_vuid(
            $pat,
            $crate::debug::validation_forensic::LayerSeverity::Error,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; escalate $pat:literal => warning ; $($rest:tt)*) => {
        $b = $b.escalate_vuid(
            $pat,
            $crate::debug::validation_forensic::LayerSeverity::Warning,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; escalate $pat:literal => info ; $($rest:tt)*) => {
        $b = $b.escalate_vuid(
            $pat,
            $crate::debug::validation_forensic::LayerSeverity::Info,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // escalate category X => severity;
    ($b:ident; escalate category $cat:ident => error ; $($rest:tt)*) => {
        $b = $b.escalate_category(
            $crate::debug::validation_forensic::DiagnosticCategory::$cat,
            $crate::debug::validation_forensic::LayerSeverity::Error,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; escalate category $cat:ident => warning ; $($rest:tt)*) => {
        $b = $b.escalate_category(
            $crate::debug::validation_forensic::DiagnosticCategory::$cat,
            $crate::debug::validation_forensic::LayerSeverity::Warning,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // demote "pat" => severity;
    ($b:ident; demote $pat:literal => error ; $($rest:tt)*) => {
        $b = $b.demote_vuid(
            $pat,
            $crate::debug::validation_forensic::LayerSeverity::Error,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; demote $pat:literal => warning ; $($rest:tt)*) => {
        $b = $b.demote_vuid(
            $pat,
            $crate::debug::validation_forensic::LayerSeverity::Warning,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; demote $pat:literal => info ; $($rest:tt)*) => {
        $b = $b.demote_vuid(
            $pat,
            $crate::debug::validation_forensic::LayerSeverity::Info,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // action severity => kind;
    ($b:ident; action error => nothing ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Error,
            $crate::debug::vl_pipeline::VlAction::Nothing,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action error => log ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Error,
            $crate::debug::vl_pipeline::VlAction::Log,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action error => panic ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Error,
            $crate::debug::vl_pipeline::VlAction::Panic,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action error => abort ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Error,
            $crate::debug::vl_pipeline::VlAction::Abort,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action error => breakpoint ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Error,
            $crate::debug::vl_pipeline::VlAction::Breakpoint,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action warning => log ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Warning,
            $crate::debug::vl_pipeline::VlAction::Log,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action warning => nothing ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Warning,
            $crate::debug::vl_pipeline::VlAction::Nothing,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action warning => panic ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Warning,
            $crate::debug::vl_pipeline::VlAction::Panic,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action info => nothing ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Info,
            $crate::debug::vl_pipeline::VlAction::Nothing,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; action info => log ; $($rest:tt)*) => {
        $b = $b.action(
            $crate::debug::validation_forensic::LayerSeverity::Info,
            $crate::debug::vl_pipeline::VlAction::Log,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // sink stderr | sink file "path"
    ($b:ident; sink stderr ; $($rest:tt)*) => {
        $b = $b.sink_stderr();
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; sink file $path:literal ; $($rest:tt)*) => {
        $b = $b.sink_file($path);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // dedup per_vuid N | dedup global N | dedup off
    ($b:ident; dedup per_vuid $n:literal ; $($rest:tt)*) => {
        $b = $b.dedup($crate::debug::vl_pipeline::DedupPolicy::PerVuid($n));
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; dedup global $n:literal ; $($rest:tt)*) => {
        $b = $b.dedup($crate::debug::vl_pipeline::DedupPolicy::Global($n));
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; dedup off ; $($rest:tt)*) => {
        $b = $b.dedup($crate::debug::vl_pipeline::DedupPolicy::Off);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // backtrace mode
    ($b:ident; backtrace none ; $($rest:tt)*) => {
        $b = $b.backtrace($crate::debug::vl_pipeline::BacktracePolicy::None);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; backtrace errors_only ; $($rest:tt)*) => {
        $b = $b.backtrace($crate::debug::vl_pipeline::BacktracePolicy::ErrorsOnly);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; backtrace warnings_and_errors ; $($rest:tt)*) => {
        $b = $b.backtrace(
            $crate::debug::vl_pipeline::BacktracePolicy::WarningsAndErrors,
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; backtrace all ; $($rest:tt)*) => {
        $b = $b.backtrace($crate::debug::vl_pipeline::BacktracePolicy::All);
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // breakpoint "pattern"
    ($b:ident; breakpoint $pat:literal ; $($rest:tt)*) => {
        $b = $b.breakpoint_on(
            $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
        );
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };

    // misc flags
    ($b:ident; no_stderr ; $($rest:tt)*) => {
        $b = $b.no_stderr();
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
    ($b:ident; no_legacy_forward ; $($rest:tt)*) => {
        $b = $b.no_legacy_forward();
        $crate::__ignis_vl_rec!($b; $($rest)*);
    };
}

/// Push a scope onto the thread-local VL stack. Returns an RAII guard
/// that restores the previous scope on drop.
///
/// Same statement grammar as [`ignis_vl!`] but only accepts `suppress`,
/// `escalate`, and `demote` statements.
#[macro_export]
macro_rules! vl_scope {
    ($($body:tt)*) => {{
        let mut __scope = $crate::debug::vl_pipeline::ScopeConfig::default();
        $crate::__vl_scope_rec!(__scope; $($body)*);
        $crate::debug::vl_pipeline::push_scope(__scope)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __vl_scope_rec {
    ($s:ident;) => {};

    ($s:ident; suppress $pat:literal ; $($rest:tt)*) => {
        $s.suppress.push(
            $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string())
        );
        $crate::__vl_scope_rec!($s; $($rest)*);
    };
    ($s:ident; suppress category $cat:ident ; $($rest:tt)*) => {
        $s.suppress.push(
            $crate::debug::vl_pipeline::VlSelector::Category(
                $crate::debug::validation_forensic::DiagnosticCategory::$cat
            )
        );
        $crate::__vl_scope_rec!($s; $($rest)*);
    };

    ($s:ident; escalate $pat:literal => error ; $($rest:tt)*) => {
        $s.escalate.push((
            $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            $crate::debug::validation_forensic::LayerSeverity::Error,
        ));
        $crate::__vl_scope_rec!($s; $($rest)*);
    };
    ($s:ident; escalate $pat:literal => warning ; $($rest:tt)*) => {
        $s.escalate.push((
            $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            $crate::debug::validation_forensic::LayerSeverity::Warning,
        ));
        $crate::__vl_scope_rec!($s; $($rest)*);
    };

    ($s:ident; demote $pat:literal => warning ; $($rest:tt)*) => {
        $s.demote.push((
            $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            $crate::debug::validation_forensic::LayerSeverity::Warning,
        ));
        $crate::__vl_scope_rec!($s; $($rest)*);
    };
    ($s:ident; demote $pat:literal => info ; $($rest:tt)*) => {
        $s.demote.push((
            $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            $crate::debug::validation_forensic::LayerSeverity::Info,
        ));
        $crate::__vl_scope_rec!($s; $($rest)*);
    };
}

/// Set a thread-local tag that will be read by diagnostic handlers.
/// Lives until explicitly removed or the thread exits.
#[macro_export]
macro_rules! vl_tag {
    ($key:expr, $value:expr) => {{
        $crate::debug::vl_pipeline::set_tag(
            ($key).to_string(),
            ::std::format!("{}", $value),
        );
    }};
}

/// Set a thread-local tag that is automatically removed when the
/// returned guard is dropped.
#[macro_export]
macro_rules! vl_tag_scoped {
    ($key:expr, $value:expr) => {{
        $crate::debug::vl_pipeline::tag_scoped(
            ($key).to_string(),
            ::std::format!("{}", $value),
        )
    }};
}

/// Run a block with capture mode active. Returns [`CapturedDiagnostics`].
///
/// Diagnostics emitted inside the block bypass all sinks and are
/// collected into the returned value instead.
#[macro_export]
macro_rules! vl_capture {
    ($($body:tt)*) => {{
        let (__captured, ()) = $crate::debug::vl_pipeline::capture(|| {
            $($body)*
        });
        __captured
    }};
}

/// Run a block and assert that captured diagnostics satisfy a set of
/// expectations. Panics on the first unmet expectation.
///
/// ```ignore
/// vl_expect! {
///     rules: {
///         exactly 1 vuid "VUID-*-01047";
///         never errors;
///         never category SynchronizationHazard;
///     }
///     in: {
///         bind_buffer_twice();
///     }
/// }
/// ```
#[macro_export]
macro_rules! vl_expect {
    (
        rules: { $($rules:tt)* }
        in: $body:block
    ) => {{
        let mut __rules: ::std::vec::Vec<$crate::debug::vl_pipeline::ExpectRule> =
            ::std::vec::Vec::new();
        $crate::__vl_expect_rec!(__rules; $($rules)*);
        let __captured = $crate::vl_capture! { $body };
        match $crate::debug::vl_pipeline::verify_expectations(&__captured, &__rules) {
            Ok(()) => {}
            Err(msg) => panic!("vl_expect! failed: {msg}"),
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __vl_expect_rec {
    ($rules:ident;) => {};

    ($rules:ident; exactly $n:literal vuid $pat:literal ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            count: $crate::debug::vl_pipeline::ExpectCount::Exactly($n),
            description: ::std::format!("exactly {} of VUID `{}`", $n, $pat),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; at_most $n:literal vuid $pat:literal ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            count: $crate::debug::vl_pipeline::ExpectCount::AtMost($n),
            description: ::std::format!("at most {} of VUID `{}`", $n, $pat),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; at_least $n:literal vuid $pat:literal ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            count: $crate::debug::vl_pipeline::ExpectCount::AtLeast($n),
            description: ::std::format!("at least {} of VUID `{}`", $n, $pat),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; never vuid $pat:literal ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Vuid($pat.to_string()),
            count: $crate::debug::vl_pipeline::ExpectCount::Never,
            description: ::std::format!("no VUID matching `{}`", $pat),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; never errors ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Severity(
                $crate::debug::validation_forensic::LayerSeverity::Error
            ),
            count: $crate::debug::vl_pipeline::ExpectCount::Never,
            description: "no errors".to_string(),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; never warnings ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Severity(
                $crate::debug::validation_forensic::LayerSeverity::Warning
            ),
            count: $crate::debug::vl_pipeline::ExpectCount::Never,
            description: "no warnings".to_string(),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; never category $cat:ident ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Category(
                $crate::debug::validation_forensic::DiagnosticCategory::$cat
            ),
            count: $crate::debug::vl_pipeline::ExpectCount::Never,
            description: ::std::format!("no category {}", stringify!($cat)),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
    ($rules:ident; at_most $n:literal errors ; $($rest:tt)*) => {
        $rules.push($crate::debug::vl_pipeline::ExpectRule {
            selector: $crate::debug::vl_pipeline::VlSelector::Severity(
                $crate::debug::validation_forensic::LayerSeverity::Error
            ),
            count: $crate::debug::vl_pipeline::ExpectCount::AtMost($n),
            description: ::std::format!("at most {} errors", $n),
        });
        $crate::__vl_expect_rec!($rules; $($rest)*);
    };
}

/// Register a custom VUID in the knowledge base.
///
/// The registered entry is picked up automatically by the forensic
/// formatter whenever a diagnostic with a matching suffix is emitted.
/// Ideal for application-specific error codes that should use the same
/// rich diagnostic UI as the built-in Vulkan rules.
///
/// ```ignore
/// vuid! {
///     code: "MY-APP-A001",
///     title: "bindless heap near exhaustion",
///     severity: Warning,
///     category: Other,
///     what_happened: "the sampled-image slot pool is >90% full",
///     why_rejected: "BindlessHeap capacity is fixed at creation",
///     ignis_fix: "increase BindlessConfig::sampled_images or free unused handles",
///     spec_section: "N/A (application-defined)",
/// }
/// ```
#[macro_export]
macro_rules! vuid {
    (
        code: $code:expr,
        title: $title:expr,
        severity: $_sev:ident,
        category: $cat:ident,
        what_happened: $what:expr,
        why_rejected: $why:expr,
        ignis_fix: $fix:expr,
        spec_section: $spec:expr $(,)?
    ) => {{
        $crate::debug::vuid_kb::register_runtime_entry(
            $crate::debug::vuid_kb::RuntimeEntry {
                vuid_suffix: ($code).to_string(),
                title: ($title).to_string(),
                category: $crate::debug::validation_forensic::DiagnosticCategory::$cat,
                what_happened: ($what).to_string(),
                why_rejected: ($why).to_string(),
                ignis_fix: ($fix).to_string(),
                spec_section: ($spec).to_string(),
            }
        );
    }};
}