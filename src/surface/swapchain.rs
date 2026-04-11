//! Swapchain management.
//!
//! Wraps the VK_KHR_swapchain extension with a convenient lifecycle API.
//! The user creates the `VkSurfaceKHR` externally (via winit, raw platform
//! extensions, etc.) and passes it to [`Swapchain::new`].
//!
//! # Lifecycle
//!
//! ```text
//! let mut sc = Swapchain::new(&ignis, surface, &config)?;
//! loop {
//!     match sc.acquire_next_image(timeout, semaphore, fence) {
//!         Ok((index, suboptimal)) => { /* render */ }
//!         Err(Error::SwapchainOutOfDate) => {
//!             sc.recreate(new_width, new_height)?;
//!             continue;
//!         }
//!         Err(e) => return Err(e),
//!     }
//!     sc.present(queue_handle, index, &[wait_semaphore])?;
//! }
//! ```
//!
//! # Surface Ownership
//!
//! The `Swapchain` does NOT own or destroy the `VkSurfaceKHR`. The caller
//! must destroy it after the swapchain is dropped.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// Queried surface capabilities, formats, and present modes.
#[derive(Debug, Clone)]
pub struct SwapchainSupport {
    /// Surface capabilities (min/max image count, extents, transforms, etc.).
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    /// Supported surface formats.
    pub formats: Vec<vk::SurfaceFormatKHR>,
    /// Supported present modes.
    pub present_modes: Vec<vk::PresentModeKHR>,
}

/// Configuration for swapchain creation.
#[derive(Debug, Clone)]
pub struct SwapchainConfig {
    /// Preferred surface format. Falls back to the first available format
    /// if the preferred one is not supported.
    pub preferred_format: vk::SurfaceFormatKHR,
    /// Preferred present mode. Falls back to FIFO (always available) if
    /// the preferred one is not supported.
    pub preferred_present_mode: vk::PresentModeKHR,
    /// Desired image count (will be clamped to surface capabilities).
    pub image_count: u32,
    /// Image usage flags (typically `COLOR_ATTACHMENT`).
    pub image_usage: vk::ImageUsageFlags,
    /// Composite alpha mode.
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    /// Whether to clip pixels obscured by other windows.
    pub clipped: bool,
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            preferred_format: vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            preferred_present_mode: vk::PresentModeKHR::MAILBOX,
            image_count: 3,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            clipped: true,
        }
    }
}

/// Managed Vulkan swapchain with image views.
///
/// Created via [`Ignis::create_swapchain`](crate::Ignis::create_swapchain)
/// or [`Swapchain::new`].
pub struct Swapchain {
    shared: Arc<SharedState>,
    surface_fn: ash::khr::surface::Instance,
    swapchain_fn: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    config: SwapchainConfig,
}

impl Swapchain {
    /// Create a new swapchain for the given surface.
    ///
    /// The surface must have been created for the same Vulkan instance.
    /// Required extensions: `VK_KHR_surface` (instance), `VK_KHR_swapchain`
    /// (device).
    ///
    /// # Arguments
    ///
    /// * `shared` - Device state
    /// * `surface` - Externally-created surface handle
    /// * `config` - Desired swapchain parameters
    /// * `width` - Initial framebuffer width
    /// * `height` - Initial framebuffer height
    pub fn new(
        shared: Arc<SharedState>,
        surface: vk::SurfaceKHR,
        config: &SwapchainConfig,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let surface_fn = ash::khr::surface::Instance::new(&shared.entry, &shared.instance);
        let swapchain_fn = ash::khr::swapchain::Device::new(&shared.instance, &shared.device);

        let support = Self::query_support_inner(&surface_fn, shared.physical_device, surface)?;

        let format = Self::choose_format(&support.formats, &config.preferred_format);
        let present_mode =
            Self::choose_present_mode(&support.present_modes, config.preferred_present_mode);
        let extent = Self::choose_extent(&support.capabilities, width, height);
        let image_count = config
            .image_count
            .max(support.capabilities.min_image_count)
            .min(if support.capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                support.capabilities.max_image_count
            });

        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(config.image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(config.composite_alpha)
            .present_mode(present_mode)
            .clipped(config.clipped)
            .old_swapchain(vk::SwapchainKHR::null());

        let handle = unsafe { swapchain_fn.create_swapchain(&ci, None)? };
        let images = unsafe { swapchain_fn.get_swapchain_images(handle)? };
        let image_views = Self::create_image_views(&shared.device, &images, format.format)?;

        Ok(Self {
            shared,
            surface_fn,
            swapchain_fn,
            surface,
            handle,
            images,
            image_views,
            format,
            extent,
            config: config.clone(),
        })
    }

    /// Query surface capabilities, formats, and present modes.
    pub fn query_support(
        shared: &SharedState,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapchainSupport> {
        let surface_fn = ash::khr::surface::Instance::new(&shared.entry, &shared.instance);
        Self::query_support_inner(&surface_fn, shared.physical_device, surface)
    }

    fn query_support_inner(
        surface_fn: &ash::khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapchainSupport> {
        unsafe {
            let capabilities =
                surface_fn.get_physical_device_surface_capabilities(physical_device, surface)?;
            let formats =
                surface_fn.get_physical_device_surface_formats(physical_device, surface)?;
            let present_modes =
                surface_fn.get_physical_device_surface_present_modes(physical_device, surface)?;
            Ok(SwapchainSupport {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    /// Recreate the swapchain with new dimensions.
    ///
    /// Typically called after a window resize or when
    /// `acquire_next_image` returns [`Error::SwapchainOutOfDate`].
    ///
    /// Waits for the device to be idle before recreation.
    pub fn recreate(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.shared.device.device_wait_idle()?;
        }

        let support =
            Self::query_support_inner(&self.surface_fn, self.shared.physical_device, self.surface)?;

        let extent = Self::choose_extent(&support.capabilities, width, height);
        let image_count = self
            .config
            .image_count
            .max(support.capabilities.min_image_count)
            .min(if support.capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                support.capabilities.max_image_count
            });

        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(self.config.image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(self.config.composite_alpha)
            .present_mode(Self::choose_present_mode(
                &support.present_modes,
                self.config.preferred_present_mode,
            ))
            .clipped(self.config.clipped)
            .old_swapchain(self.handle);

        let new_handle = unsafe { self.swapchain_fn.create_swapchain(&ci, None)? };

        // Destroy old resources.
        self.destroy_image_views();
        unsafe {
            self.swapchain_fn.destroy_swapchain(self.handle, None);
        }

        self.handle = new_handle;
        self.images = unsafe { self.swapchain_fn.get_swapchain_images(new_handle)? };
        self.image_views =
            Self::create_image_views(&self.shared.device, &self.images, self.format.format)?;
        self.extent = extent;

        Ok(())
    }

    /// Acquire the next swapchain image.
    ///
    /// Returns `(image_index, suboptimal)`. If `suboptimal` is `true`,
    /// the swapchain still works but should be recreated for optimal
    /// performance.
    ///
    /// # Errors
    ///
    /// - [`Error::SwapchainOutOfDate`] - must recreate before using
    /// - [`Error::SurfaceLost`] - surface is no longer usable
    /// - [`Error::Timeout`] - `timeout` expired without acquiring
    pub fn acquire_next_image(
        &self,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<(u32, bool)> {
        match unsafe {
            self.swapchain_fn
                .acquire_next_image(self.handle, timeout, semaphore, fence)
        } {
            Ok((index, suboptimal)) => Ok((index, suboptimal)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::SwapchainOutOfDate),
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => Err(Error::SurfaceLost),
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => Err(Error::Timeout),
            Err(e) => Err(Error::Vulkan(e)),
        }
    }

    /// Present an image to the display.
    ///
    /// # Arguments
    ///
    /// * `queue` - Raw queue handle (must support presentation)
    /// * `image_index` - Index from `acquire_next_image`
    /// * `wait_semaphores` - Semaphores to wait on before presenting
    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool> {
        let swapchains = [self.handle];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match unsafe { self.swapchain_fn.queue_present(queue, &present_info) } {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::SwapchainOutOfDate),
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => Err(Error::SurfaceLost),
            Err(e) => Err(Error::Vulkan(e)),
        }
    }

    /// Current swapchain images.
    #[inline]
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Image views corresponding to each swapchain image.
    #[inline]
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    /// Number of swapchain images.
    #[inline]
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Current swapchain extent (width, height).
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Current surface format.
    #[inline]
    pub fn format(&self) -> vk::SurfaceFormatKHR {
        self.format
    }

    /// Raw swapchain handle.
    #[inline]
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    fn choose_format(
        available: &[vk::SurfaceFormatKHR],
        preferred: &vk::SurfaceFormatKHR,
    ) -> vk::SurfaceFormatKHR {
        available
            .iter()
            .find(|f| f.format == preferred.format && f.color_space == preferred.color_space)
            .copied()
            .unwrap_or(available[0])
    }

    fn choose_present_mode(
        available: &[vk::PresentModeKHR],
        preferred: vk::PresentModeKHR,
    ) -> vk::PresentModeKHR {
        if available.contains(&preferred) {
            preferred
        } else {
            vk::PresentModeKHR::FIFO // Always available per spec.
        }
    }

    fn choose_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        width: u32,
        height: u32,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: width
                    .max(capabilities.min_image_extent.width)
                    .min(capabilities.max_image_extent.width),
                height: height
                    .max(capabilities.min_image_extent.height)
                    .min(capabilities.max_image_extent.height),
            }
        } else {
            capabilities.current_extent
        }
    }

    fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| {
                let ci = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&ci, None).map_err(Error::from) }
            })
            .collect()
    }

    fn destroy_image_views(&mut self) {
        for &view in &self.image_views {
            unsafe {
                self.shared.device.destroy_image_view(view, None);
            }
        }
        self.image_views.clear();
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            let _ = self.shared.device.device_wait_idle();
        }
        self.destroy_image_views();
        unsafe {
            self.swapchain_fn.destroy_swapchain(self.handle, None);
            // NOTE: surface is NOT destroyed here. The caller owns it.
        }
    }
}
