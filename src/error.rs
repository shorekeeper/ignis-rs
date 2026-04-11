//! Error types for the ignis crate.

use ash::vk;
use std::fmt;

use crate::QueueType;

/// Convenience alias used throughout ignis.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during ignis operations.
#[derive(Debug)]
pub enum Error {
    /// A Vulkan API call returned an error code.
    Vulkan(vk::Result),
    /// Failed to load the Vulkan shared library.
    LoadFailed,
    /// No physical device satisfies the requirements.
    NoSuitableDevice,
    /// No queue family provides the requested capability.
    NoSuitableQueueFamily(QueueType),
    /// A required device feature or extension was not enabled.
    FeatureNotEnabled(&'static str),
    /// A configuration parameter is invalid.
    InvalidConfig(&'static str),
    /// A worker thread panicked during parallel command recording.
    ThreadPanic,
    /// The provided SPIR-V data is invalid.
    InvalidSpirv,
    /// A fence wait timed out.
    Timeout,
    /// No memory type satisfies the requested requirements and location.
    NoSuitableMemoryType,
    /// The swapchain is out of date and must be recreated.
    SwapchainOutOfDate,
    /// The swapchain surface has been lost.
    SurfaceLost,
    /// An error with additional context describing what operation failed.
    Context(Box<Error>, String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(code) => write!(f, "Vulkan error: {code:?}"),
            Self::LoadFailed => write!(f, "failed to load Vulkan library"),
            Self::NoSuitableDevice => write!(f, "no suitable physical device found"),
            Self::NoSuitableQueueFamily(qt) => {
                write!(f, "no queue family with {qt:?} capability")
            }
            Self::FeatureNotEnabled(name) => {
                write!(f, "required feature/extension not enabled: {name}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::ThreadPanic => write!(f, "worker thread panicked"),
            Self::InvalidSpirv => write!(f, "invalid SPIR-V data"),
            Self::Timeout => write!(f, "GPU fence wait timed out"),
            Self::NoSuitableMemoryType => {
                write!(f, "no suitable memory type for the requested allocation")
            }
            Self::SwapchainOutOfDate => write!(f, "swapchain is out of date"),
            Self::SurfaceLost => write!(f, "surface has been lost"),
            Self::Context(inner, ctx) => {
                write!(f, "{inner}\n  context: {ctx}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<vk::Result> for Error {
    #[inline]
    fn from(result: vk::Result) -> Self {
        Error::Vulkan(result)
    }
}

/// Extension trait for enriching errors with context strings.
///
/// # Example
///
/// ```rust
/// use ignis::error::WithContext;
///
/// fn load_texture(ignis: &ignis::Ignis) -> ignis::Result<()> {
///     let buf = ignis.create_buffer(&ignis::BufferInfo::staging(4096))
///         .context("allocating staging buffer for texture upload")?;
///     Ok(())
/// }
/// ```
pub trait WithContext<T> {
    /// Attach a static context string to the error.
    fn context(self, ctx: &'static str) -> Result<T>;

    /// Attach a lazily-evaluated context string to the error.
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T>;
}

impl<T> WithContext<T> for Result<T> {
    fn context(self, ctx: &'static str) -> Result<T> {
        self.map_err(|e| Error::Context(Box::new(e), ctx.to_string()))
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| Error::Context(Box::new(e), f()))
    }
}