//! Device management, physical device selection, and shared state.
//!
//! This module defines [`SharedState`] - the internal representation shared
//! across all ignis objects via `Arc` - and the two device creation paths:
//! managed and external.

use std::ffi::{c_char, CStr, CString};

use ash::vk;

use crate::error::{Error, Result};

/// Internal shared state held by all ignis objects.
///
/// Contains the core Vulkan handles, properties, and optional extension
/// function loaders. Wrapped in `Arc` for thread-safe sharing.
///
/// In managed mode, dropping the last `Arc<SharedState>` will wait for
/// the device to become idle and then destroy the device and instance.
/// In external mode, no destruction occurs.
pub struct SharedState {
    /// Vulkan entry point, always loaded.
    /// In managed mode created during initialization, in external mode
    /// loaded automatically for extension function queries.
    pub(crate) entry: ash::Entry,

    /// Ash instance handle.
    pub(crate) instance: ash::Instance,

    /// Ash logical device handle.
    pub(crate) device: ash::Device,

    /// Selected physical device.
    pub(crate) physical_device: vk::PhysicalDevice,

    /// Physical device properties (name, limits, API version, etc.).
    pub(crate) device_properties: vk::PhysicalDeviceProperties,

    /// Memory heaps and types available on the device.
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,

    /// Queue family properties for the physical device.
    pub(crate) queue_family_props: Vec<vk::QueueFamilyProperties>,

    /// Ray tracing pipeline extension function loader.
    /// `None` if the extension is not enabled.
    pub(crate) rt_pipeline_fn: Option<ash::khr::ray_tracing_pipeline::Device>,

    /// Acceleration structure extension function loader.
    /// `None` if the extension is not enabled.
    pub(crate) accel_struct_fn: Option<ash::khr::acceleration_structure::Device>,

    /// Ray tracing pipeline properties (handle sizes, alignment, max recursion).
    /// `None` if ray tracing is not enabled.
    pub(crate) rt_properties: Option<RayTracingProperties>,

    /// Whether ignis owns (and should destroy) the instance and device.
    pub(crate) is_managed: bool,
    /// Whether timeline semaphores are available (Vulkan 1.2+).
    pub(crate) supports_timelines: bool,
}

// Compile-time assertion that SharedState is Send + Sync.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    // ash::Device and ash::Instance implement Send + Sync.
    // vk::{PhysicalDevice, PhysicalDeviceProperties, ...} are plain data.
    // Option<ash::khr::*::Device> contain only fn pointers + handle -> Send + Sync.
    assert_send_sync::<SharedState>();
};

impl Drop for SharedState {
    fn drop(&mut self) {
        if self.is_managed {
            unsafe {
                // Wait for all device operations to complete before teardown.
                let _ = self.device.device_wait_idle();
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}

/// Extracted ray tracing pipeline properties.
///
/// Stored as plain values to avoid lifetime issues with the Vulkan
/// properties chain.
#[derive(Debug, Clone, Copy)]
pub struct RayTracingProperties {
    /// Size in bytes of a single shader group handle.
    pub shader_group_handle_size: u32,
    /// Maximum ray recursion depth supported by the device.
    pub max_ray_recursion_depth: u32,
    /// Required alignment for shader group handles within an SBT record.
    pub shader_group_handle_alignment: u32,
    /// Required alignment for the base of each SBT region.
    pub shader_group_base_alignment: u32,
    /// Maximum stride between shader group handles.
    pub max_shader_group_stride: u32,
}

/// Use the builder methods to customize, then pass to
/// [`crate::Ignis::managed`].
pub struct ManagedConfig {
    /// Application name reported to the Vulkan driver.
    pub app_name: CString,
    /// Application version.
    pub app_version: u32,
    /// Minimum required Vulkan API version (e.g. `vk::API_VERSION_1_3`).
    pub vulkan_version: u32,
    /// Whether to enable the Khronos validation layer.
    pub validation: bool,
    /// Whether to enable ray tracing extensions and features.
    pub raytracing: bool,
    /// Additional instance extensions to enable.
    pub instance_extensions: Vec<CString>,
    /// Additional device extensions to enable.
    pub device_extensions: Vec<CString>,
    /// Custom physical device selector.
    ///
    /// Receives a slice of candidate devices and returns the index of the
    /// chosen device. If `None`, ignis uses a default heuristic that prefers
    /// discrete GPUs.
    pub device_selector: Option<Box<dyn Fn(&[PhysicalDeviceInfo]) -> usize + Send>>,
}

impl ManagedConfig {
    /// Create a config with the given application name and Vulkan API version.
    ///
    /// Defaults: no validation, no ray tracing, no extra extensions,
    /// default device selector.
    pub fn new(app_name: &str, vulkan_version: u32) -> Self {
        Self {
            app_name: CString::new(app_name).unwrap_or_else(|_| CString::new("ignis").unwrap()),
            app_version: vk::make_api_version(0, 1, 0, 0),
            vulkan_version,
            validation: false,
            raytracing: false,
            instance_extensions: Vec::new(),
            device_extensions: Vec::new(),
            device_selector: None,
        }
    }

    /// Set the application version.
    pub fn app_version(mut self, version: u32) -> Self {
        self.app_version = version;
        self
    }

    /// Enable or disable the Khronos validation layer.
    pub fn enable_validation(mut self, enable: bool) -> Self {
        self.validation = enable;
        self
    }

    /// Enable or disable ray tracing extensions.
    ///
    /// When enabled, ignis will request `VK_KHR_ray_tracing_pipeline`,
    /// `VK_KHR_acceleration_structure`, and `VK_KHR_deferred_host_operations`,
    /// and enable the corresponding device features.
    ///
    /// Requires Vulkan 1.2 or later.
    pub fn enable_raytracing(mut self, enable: bool) -> Self {
        self.raytracing = enable;
        self
    }

    /// Add an instance extension by name.
    pub fn instance_extension(mut self, name: &CStr) -> Self {
        self.instance_extensions.push(name.to_owned());
        self
    }

    /// Add a device extension by name.
    pub fn device_extension(mut self, name: &CStr) -> Self {
        self.device_extensions.push(name.to_owned());
        self
    }

    /// Set a custom physical device selection function.
    pub fn device_selector<F>(mut self, f: F) -> Self
    where
        F: Fn(&[PhysicalDeviceInfo]) -> usize + Send + 'static,
    {
        self.device_selector = Some(Box::new(f));
        self
    }
}

/// Information about a physical device, used by the device selector.
#[derive(Debug, Clone)]
pub struct PhysicalDeviceInfo {
    /// Raw Vulkan handle.
    pub handle: vk::PhysicalDevice,
    /// Device properties (name, type, limits, etc.).
    pub properties: vk::PhysicalDeviceProperties,
    /// Supported features.
    pub features: vk::PhysicalDeviceFeatures,
    /// Queue family properties.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
}

/// Describes a queue handle and its capabilities.
#[derive(Debug, Clone)]
pub struct QueueAllocation {
    /// Queue family index.
    pub family_index: u32,
    /// Queue index within the family.
    pub queue_index: u32,
    /// Raw Vulkan queue handle.
    pub handle: vk::Queue,
    /// Capability flags of the queue family.
    pub capabilities: vk::QueueFlags,
}

/// Information for wrapping an externally-owned Vulkan device.
///
/// The caller owns all handles and is responsible for their lifetime.
///
/// # Example
///
/// ```rust,no_run
/// use ignis::{ExternalDeviceInfo, QueueAllocation};
///
/// let info = ExternalDeviceInfo {
///     instance: my_instance.clone(),
///     device: my_device.clone(),
///     physical_device: my_physical,
///     queue_allocations: vec![
///         QueueAllocation {
///             family_index: 0,
///             queue_index: 0,
///             handle: my_gfx_queue,
///             capabilities: ash::vk::QueueFlags::GRAPHICS,
///         },
///     ],
///     enable_raytracing: false,
/// };
/// ```
pub struct ExternalDeviceInfo {
    /// Ash instance (cloned from the external owner).
    pub instance: ash::Instance,
    /// Ash logical device (cloned from the external owner).
    pub device: ash::Device,
    /// Physical device handle.
    pub physical_device: vk::PhysicalDevice,
    /// Queue allocations that ignis may use.
    pub queue_allocations: Vec<QueueAllocation>,
    /// Whether to load ray tracing extension function pointers.
    /// The external device must have been created with the extensions enabled.
    pub enable_raytracing: bool,
}

/// Create a fully managed Vulkan device with instance, physical device selection,
/// logical device, and queue allocation.
pub(crate) fn create_managed_device(
    config: ManagedConfig,
) -> Result<(SharedState, Vec<QueueAllocation>)> {
    // Step 1: Load Vulkan.
    let entry = unsafe { ash::Entry::load() }.map_err(|_| Error::LoadFailed)?;

    // Step 2: Build instance.
    let engine_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Ignis\0") };
    let app_info = vk::ApplicationInfo::default()
        .application_name(&config.app_name)
        .application_version(config.app_version)
        .engine_name(engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(config.vulkan_version);

    let inst_ext_ptrs: Vec<*const c_char> = config
        .instance_extensions
        .iter()
        .map(|s| s.as_ptr())
        .chain(cfg_macos_instance_extensions())
        .collect();

    let mut layer_ptrs: Vec<*const c_char> = Vec::new();
    let validation_layer =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };
    if config.validation {
        layer_ptrs.push(validation_layer.as_ptr());
    }

    let instance_flags = base_instance_flags();

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&inst_ext_ptrs)
        .flags(instance_flags);

    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    // Step 3: Select physical device.
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    if physical_devices.is_empty() {
        // Cleanup before returning error.
        unsafe { instance.destroy_instance(None) };
        return Err(Error::NoSuitableDevice);
    }

    let device_infos: Vec<PhysicalDeviceInfo> = physical_devices
        .iter()
        .map(|&pd| {
            let properties = unsafe { instance.get_physical_device_properties(pd) };
            let features = unsafe { instance.get_physical_device_features(pd) };
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };
            PhysicalDeviceInfo {
                handle: pd,
                properties,
                features,
                queue_families,
            }
        })
        .collect();

    let selected_idx = if let Some(selector) = &config.device_selector {
        selector(&device_infos)
    } else {
        default_device_score(&device_infos)
    };
    let chosen = &device_infos[selected_idx];
    let physical_device = chosen.handle;

    // Step 4: Determine queue families.
    let queue_family_props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let graphics_family = queue_family_props
        .iter()
        .position(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .ok_or(Error::NoSuitableQueueFamily(crate::QueueType::Graphics))?
        as u32;

    // Prefer a dedicated compute family.
    let compute_family = queue_family_props
        .iter()
        .enumerate()
        .find(|(_, qf)| {
            qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .map_or(graphics_family, |(i, _)| i as u32);

    // Prefer a dedicated transfer family.
    let transfer_family = queue_family_props
        .iter()
        .enumerate()
        .find(|(_, qf)| {
            qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && !qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .map_or(graphics_family, |(i, _)| i as u32);

    // Deduplicate families.
    let mut unique_families = vec![graphics_family];
    if !unique_families.contains(&compute_family) {
        unique_families.push(compute_family);
    }
    if !unique_families.contains(&transfer_family) {
        unique_families.push(transfer_family);
    }

    let priority = [1.0_f32];
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
        .iter()
        .map(|&family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priority)
        })
        .collect();

    // Step 5: Device extensions.
    let mut dev_ext_ptrs: Vec<*const c_char> = config
        .device_extensions
        .iter()
        .map(|s| s.as_ptr())
        .chain(cfg_macos_device_extensions())
        .chain(if config.raytracing {
            vec![
                ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
                ash::khr::acceleration_structure::NAME.as_ptr(),
                ash::khr::deferred_host_operations::NAME.as_ptr(),
            ]
        } else {
            vec![]
        })
        .collect();
    if config.raytracing {
        dev_ext_ptrs.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
        dev_ext_ptrs.push(ash::khr::acceleration_structure::NAME.as_ptr());
        dev_ext_ptrs.push(ash::khr::deferred_host_operations::NAME.as_ptr());
    }

    // Step 6: Features chain.
    // We always enable Vulkan 1.2 features if the API version permits.
    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();

    // Timeline semaphores: mandatory in Vulkan 1.2 core.
    // Enables O(1) GPU completion tracking instead of per-fence polling.
    if config.vulkan_version >= vk::API_VERSION_1_2 {
        vulkan12_features = vulkan12_features.timeline_semaphore(true);
    }

    if config.raytracing {
        vulkan12_features = vulkan12_features
            .buffer_device_address(true)
            .descriptor_indexing(true);
    }

    let mut rt_pipe_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let mut accel_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan12_features);
    if config.raytracing {
        features2 = features2
            .push_next(&mut rt_pipe_features)
            .push_next(&mut accel_features);
    }

    let device_info = vk::DeviceCreateInfo::default()
        .push_next(&mut features2)
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&dev_ext_ptrs);

    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

    // Step 7: Retrieve queues.
    let mut allocations = Vec::with_capacity(unique_families.len());
    for &family in &unique_families {
        let handle = unsafe { device.get_device_queue(family, 0) };
        allocations.push(QueueAllocation {
            family_index: family,
            queue_index: 0,
            handle,
            capabilities: queue_family_props[family as usize].queue_flags,
        });
    }

    // Step 8: Load extension function pointers.
    let (rt_pipeline_fn, accel_struct_fn, rt_properties) = if config.raytracing {
        let rt_fn = ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
        let accel_fn = ash::khr::acceleration_structure::Device::new(&instance, &device);
        let props = query_rt_properties(&instance, physical_device);
        (Some(rt_fn), Some(accel_fn), Some(props))
    } else {
        (None, None, None)
    };

    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let supports_timelines = config.vulkan_version >= vk::API_VERSION_1_2;

    let shared = SharedState {
        entry,
        instance,
        device,
        physical_device,
        device_properties,
        memory_properties,
        queue_family_props,
        rt_pipeline_fn,
        accel_struct_fn,
        rt_properties,
        is_managed: true,
        supports_timelines,
    };


    Ok((shared, allocations))
}

/// Wrap an externally-owned device.
pub(crate) fn create_external_device(
    info: ExternalDeviceInfo,
) -> Result<(SharedState, Vec<QueueAllocation>)> {
    if info.queue_allocations.is_empty() {
        return Err(Error::InvalidConfig(
            "at least one queue allocation is required",
        ));
    }

    // Load entry for extension function loading (surface, swapchain, etc.).
    // This is cheap and idempotent - the Vulkan library is already loaded
    // by the external owner.
    let entry = unsafe { ash::Entry::load() }.map_err(|_| Error::LoadFailed)?;

    let device_properties = unsafe {
        info.instance
            .get_physical_device_properties(info.physical_device)
    };
    let memory_properties = unsafe {
        info.instance
            .get_physical_device_memory_properties(info.physical_device)
    };
    let queue_family_props = unsafe {
        info.instance
            .get_physical_device_queue_family_properties(info.physical_device)
    };

    let api_version = device_properties.api_version;
    let supports_timelines = vk::api_version_major(api_version) > 1
        || (vk::api_version_major(api_version) == 1
            && vk::api_version_minor(api_version) >= 2);

    let (rt_pipeline_fn, accel_struct_fn, rt_properties) = if info.enable_raytracing {
        let rt_fn = ash::khr::ray_tracing_pipeline::Device::new(&info.instance, &info.device);
        let accel_fn = ash::khr::acceleration_structure::Device::new(&info.instance, &info.device);
        let props = query_rt_properties(&info.instance, info.physical_device);
        (Some(rt_fn), Some(accel_fn), Some(props))
    } else {
        (None, None, None)
    };

    let allocations = info.queue_allocations;

    let shared = SharedState {
        entry,
        instance: info.instance,
        device: info.device,
        physical_device: info.physical_device,
        device_properties,
        memory_properties,
        queue_family_props,
        rt_pipeline_fn,
        accel_struct_fn,
        rt_properties,
        is_managed: false,
        supports_timelines,
    };

    Ok((shared, allocations))
}

/// Default device selection heuristic. Prefers discrete GPUs, then
/// integrated, then any other type. Breaks ties by device local memory size.
fn default_device_score(devices: &[PhysicalDeviceInfo]) -> usize {
    devices
        .iter()
        .enumerate()
        .max_by_key(|(_, d)| {
            let type_score = match d.properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 1,
                _ => 0,
            };
            (type_score, d.properties.limits.max_image_dimension2_d)
        })
        .map_or(0, |(i, _)| i)
}

/// Query ray tracing pipeline properties from the physical device.
fn query_rt_properties(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> RayTracingProperties {
    let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut props2);
    }
    RayTracingProperties {
        shader_group_handle_size: rt_props.shader_group_handle_size,
        max_ray_recursion_depth: rt_props.max_ray_recursion_depth,
        shader_group_handle_alignment: rt_props.shader_group_handle_alignment,
        shader_group_base_alignment: rt_props.shader_group_base_alignment,
        max_shader_group_stride: rt_props.max_shader_group_stride,
    }
}

/// Returns additional instance extension pointers required on macOS.
/// On other platforms returns an empty iterator.
#[cfg(target_os = "macos")]
fn cfg_macos_instance_extensions() -> impl Iterator<Item = *const c_char> {
    std::iter::once(ash::khr::portability_enumeration::NAME.as_ptr())
}

#[cfg(not(target_os = "macos"))]
fn cfg_macos_instance_extensions() -> impl Iterator<Item = *const c_char> {
    std::iter::empty()
}

/// Returns instance creation flags. Includes portability enumeration on macOS.
#[cfg(target_os = "macos")]
fn base_instance_flags() -> vk::InstanceCreateFlags {
    vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
}

#[cfg(not(target_os = "macos"))]
fn base_instance_flags() -> vk::InstanceCreateFlags {
    vk::InstanceCreateFlags::empty()
}

#[cfg(target_os = "macos")]
fn cfg_macos_device_extensions() -> impl Iterator<Item = *const c_char> {
    std::iter::once(ash::khr::portability_subset::NAME.as_ptr())
}

#[cfg(not(target_os = "macos"))]
fn cfg_macos_device_extensions() -> impl Iterator<Item = *const c_char> {
    std::iter::empty()
}
