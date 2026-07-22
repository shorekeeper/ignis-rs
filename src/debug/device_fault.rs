//! Device fault diagnostics via three vendor extensions.
//!
//! Bridges `VK_EXT_device_fault` (vendor-neutral fault info),
//! `VK_NV_device_diagnostic_checkpoints` (NVIDIA per-queue checkpoint
//! readback), and `VK_AMD_buffer_marker` (AMD pipeline-stage-bound
//! marker writes). Each extension is independent; the [`DeviceFaultRecorder`]
//! discovers which are available at construction time and exposes a
//! unified API. Subsystems that have no support quietly become no-ops.
//!
//! # Why three extensions
//!
//! - `VK_EXT_device_fault` runs after `VK_ERROR_DEVICE_LOST` and returns
//!   a vendor-formatted description of what went wrong, plus optional
//!   address-space and vendor-binary blobs. This is the closest thing to
//!   a Vulkan-native crash log.
//! - `VK_NV_device_diagnostic_checkpoints` lets you tag command stream
//!   positions with arbitrary 64-bit user values. After a hang or crash,
//!   `vkGetQueueCheckpointDataNV` returns the LAST checkpoint that
//!   completed at each pipeline stage. No buffer needed.
//! - `VK_AMD_buffer_marker` writes a 32-bit value into a buffer at a
//!   specific pipeline stage. Conceptually similar to ignis's existing
//!   [`BreadcrumbBuffer`] but with explicit pipeline-stage selection,
//!   which makes the trail much more accurate on tile-based and
//!   asynchronous architectures.
//!
//! # Integration with `CrashReporter`
//!
//! Attach a recorder via [`CrashReporter::attach_device_fault`] and any
//! data the recorder has captured will appear in the markdown report
//! produced on `trigger`. On a healthy device the section is short
//! (function pointers loaded, no checkpoints recorded yet); after
//! DEVICE_LOST it can be the most actionable block in the entire report.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ignis::debug::device_fault::*;
//! # use std::sync::Arc;
//! # fn example(ignis: &Ignis) -> Result<()> {
//! let recorder = ignis.create_device_fault_recorder();
//! let reporter = ignis.create_crash_reporter();
//! reporter.attach_device_fault(Arc::clone(&recorder));
//!
//! // During recording:
//! //   recorder.cmd_checkpoint(&rec, 42);
//! //   recorder.cmd_buffer_marker(&rec, &markers, "after_geometry",
//! //                              vk::PipelineStageFlags::FRAGMENT_SHADER);
//!
//! // On VK_ERROR_DEVICE_LOST:
//! //   reporter.trigger(vk::Result::ERROR_DEVICE_LOST);
//! # Ok(())
//! # }
//! ```
//!
//! [`BreadcrumbBuffer`]: super::hang_detector::BreadcrumbBuffer
//! [`CrashReporter::attach_device_fault`]: super::crash_report::CrashReporter::attach_device_fault

use std::ffi::{c_void, CStr, CString};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::command::CommandRecorder;
use crate::device::SharedState;
use crate::error::{Error, Result};

// ---- AMD buffer marker buffer -------------------------------------------

/// CPU-visible buffer of 32-bit slots written by `vkCmdWriteBufferMarkerAMD`.
///
/// One slot per `insert` call. Each slot's value is its 1-based id; the
/// initial buffer state is zeros, so a slot reads as nonzero exactly when
/// that marker fired on the GPU. Reading back the buffer after a fault
/// reveals the precise high-water mark of execution per pipeline stage.
pub struct AmdMarkerBuffer {
    shared: Arc<SharedState>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u32,
    capacity: u32,
    next_slot: AtomicU32,
    labels: Mutex<Vec<MarkerLabel>>,
}

#[derive(Debug, Clone)]
struct MarkerLabel {
    label: String,
    stage: vk::PipelineStageFlags,
}

// SAFETY: mapped pointer points to persistently mapped Vulkan memory.
// All access is synchronized through the labels Mutex and the atomic
// next_slot counter.
unsafe impl Send for AmdMarkerBuffer {}
unsafe impl Sync for AmdMarkerBuffer {}

impl AmdMarkerBuffer {
    /// Allocate a marker buffer with the given slot capacity. Each slot
    /// holds a single 32-bit marker value and one row in the readback
    /// table. 256 to 1024 slots is appropriate for most workloads.
    pub fn new(shared: Arc<SharedState>, capacity: u32) -> Result<Self> {
        let cap = capacity.max(1);
        let bytes = (cap as u64) * 4;

        let buffer_ci = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { shared.device.create_buffer(&buffer_ci, None)? };
        let req = unsafe { shared.device.get_buffer_memory_requirements(buffer) };

        let mt = find_host_visible(&shared.memory_properties, req.memory_type_bits)
            .ok_or_else(|| {
                unsafe { shared.device.destroy_buffer(buffer, None) };
                Error::NoSuitableMemoryType
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mt);
        let memory = unsafe { shared.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { shared.device.destroy_buffer(buffer, None) };
            Error::Vulkan(e)
        })?;

        unsafe { shared.device.bind_buffer_memory(buffer, memory, 0)? };
        let ptr = unsafe {
            shared
                .device
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())?
        }
        .cast::<u32>();
        // Initialize all slots to zero so non-fired markers stay obvious.
        unsafe { std::ptr::write_bytes(ptr, 0, cap as usize) };

        Ok(Self {
            shared,
            buffer,
            memory,
            mapped: ptr,
            capacity: cap,
            next_slot: AtomicU32::new(0),
            labels: Mutex::new(Vec::with_capacity(cap as usize)),
        })
    }

    /// Total slot count.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of `insert` calls performed so far.
    pub fn used(&self) -> u32 {
        self.next_slot.load(Ordering::Relaxed).min(self.capacity)
    }

    /// Reset the buffer to its initial zeroed state. Subsequent `insert`
    /// calls start at slot 0 again. Call between major workloads if you
    /// want a clean trail; otherwise the buffer wraps around at capacity.
    pub fn reset(&self) {
        unsafe { std::ptr::write_bytes(self.mapped, 0, self.capacity as usize) };
        self.next_slot.store(0, Ordering::Relaxed);
        self.labels.lock().unwrap().clear();
    }

    /// Raw buffer handle, useful if you want to barrier or copy from it.
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Snapshot of all recorded markers along with whether each fired.
    /// A marker is considered fired when its slot in the buffer reads
    /// nonzero (the marker writes its slot index + 1).
    pub fn readback(&self) -> Vec<MarkerEntry> {
        let labels = self.labels.lock().unwrap();
        labels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let value = unsafe { self.mapped.add(i).read_volatile() };
                MarkerEntry {
                    slot: i as u32,
                    label: l.label.clone(),
                    stage: l.stage,
                    fired: value != 0,
                    value,
                }
            })
            .collect()
    }
}

impl Drop for AmdMarkerBuffer {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.unmap_memory(self.memory);
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

/// One AMD marker readback row.
#[derive(Debug, Clone)]
pub struct MarkerEntry {
    /// Slot index (0-based).
    pub slot: u32,
    /// User label.
    pub label: String,
    /// Pipeline stage selected for the write.
    pub stage: vk::PipelineStageFlags,
    /// True if the GPU executed the marker write.
    pub fired: bool,
    /// Raw 32-bit value the GPU wrote (slot + 1 for fired markers, 0 otherwise).
    pub value: u32,
}

// ---- NV checkpoint readback ---------------------------------------------

/// One NV checkpoint readback row.
#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    /// Pipeline stage at which this checkpoint was last seen by the GPU.
    pub stage: vk::PipelineStageFlags,
    /// User-supplied 64-bit value.
    pub value: u64,
}

// ---- EXT device fault info ----------------------------------------------

/// Decoded fault info from `vkGetDeviceFaultInfoEXT`.
#[derive(Debug, Clone, Default)]
pub struct DeviceFaultInfo {
    /// Vendor-supplied human-readable description.
    pub description: String,
    /// Per-address fault descriptors (read/write violations, etc).
    pub address_infos: Vec<DeviceFaultAddressInfo>,
    /// Vendor-specific structured info (driver version, fault stage, etc).
    pub vendor_infos: Vec<DeviceFaultVendorInfo>,
    /// Vendor binary blob, opaque to ignis. Pass to vendor support.
    pub vendor_binary: Vec<u8>,
}

/// One entry from `pAddressInfos` in `VkDeviceFaultInfoEXT`.
#[derive(Debug, Clone)]
pub struct DeviceFaultAddressInfo {
    /// Type of address-space access that faulted (read, write, execute).
    pub address_type: vk::DeviceFaultAddressTypeEXT,
    /// Reported faulting address.
    pub reported_address: u64,
    /// Address granularity (size of the access unit) in bytes.
    pub address_precision: u64,
}

/// One entry from `pVendorInfos`.
#[derive(Debug, Clone)]
pub struct DeviceFaultVendorInfo {
    /// Free-form description from the driver.
    pub description: String,
    /// Vendor-specific 64-bit fault code.
    pub vendor_fault_code: u64,
    /// Vendor-specific 64-bit fault data.
    pub vendor_fault_data: u64,
}

// ---- Unified recorder ---------------------------------------------------

/// Aggregates the three GPU fault diagnostic extensions.
///
/// Construct via [`Ignis::create_device_fault_recorder`]. Extensions that
/// were not enabled at device creation simply do not contribute; calls
/// to their methods become no-ops and the corresponding sections are
/// absent from the report.
///
/// [`Ignis::create_device_fault_recorder`]: crate::Ignis::create_device_fault_recorder
pub struct DeviceFaultRecorder {
    shared: Arc<SharedState>,
    nv_fn: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    fault_fn: Option<ash::ext::device_fault::Device>,
    amd_fn: Option<ash::amd::buffer_marker::Device>,
}

impl DeviceFaultRecorder {
    /// Discover available extensions and load their function tables.
    pub fn new(shared: Arc<SharedState>) -> Self {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(_) => {
                // If the loader is gone we cannot query anything.
                return Self {
                    shared,
                    nv_fn: None,
                    fault_fn: None,
                    amd_fn: None,
                };
            }
        };

        let nv_fn = if probe_proc(&entry, &shared.instance, "vkCmdSetCheckpointNV") {
            Some(ash::nv::device_diagnostic_checkpoints::Device::new(
                &shared.instance,
                &shared.device,
            ))
        } else {
            None
        };

        let fault_fn = if probe_proc(&entry, &shared.instance, "vkGetDeviceFaultInfoEXT") {
            Some(ash::ext::device_fault::Device::new(
                &shared.instance,
                &shared.device,
            ))
        } else {
            None
        };

        let amd_fn = if probe_proc(&entry, &shared.instance, "vkCmdWriteBufferMarkerAMD") {
            Some(ash::amd::buffer_marker::Device::new(
                &shared.instance,
                &shared.device,
            ))
        } else {
            None
        };

        Self {
            shared,
            nv_fn,
            fault_fn,
            amd_fn,
        }
    }

    /// True when `VK_NV_device_diagnostic_checkpoints` is loaded.
    pub fn supports_checkpoints(&self) -> bool {
        self.nv_fn.is_some()
    }

    /// True when `VK_EXT_device_fault` is loaded.
    pub fn supports_fault_info(&self) -> bool {
        self.fault_fn.is_some()
    }

    /// True when `VK_AMD_buffer_marker` is loaded.
    pub fn supports_buffer_markers(&self) -> bool {
        self.amd_fn.is_some()
    }

    /// Allocate an AMD marker buffer with the given slot capacity.
    /// Returns an error if `VK_AMD_buffer_marker` is not enabled.
    pub fn create_marker_buffer(&self, capacity: u32) -> Result<Arc<AmdMarkerBuffer>> {
        if self.amd_fn.is_none() {
            return Err(Error::FeatureNotEnabled("VK_AMD_buffer_marker"));
        }
        Ok(Arc::new(AmdMarkerBuffer::new(
            Arc::clone(&self.shared),
            capacity,
        )?))
    }

    /// Insert an NV checkpoint into a command buffer.
    ///
    /// The 64-bit `value` is opaque to the driver; choose any encoding
    /// that helps you correlate readbacks with code locations (a hash of
    /// the call site, a sequential counter, a packed pass+frame id...).
    /// On non-NV hardware this is a no-op.
    pub fn cmd_checkpoint(&self, rec: &CommandRecorder<'_>, value: u64) {
        if let Some(nv) = &self.nv_fn {
            unsafe {
                nv.cmd_set_checkpoint(rec.raw_buffer(), value as *const c_void);
            }
        }
    }

    /// Insert an AMD buffer marker.
    ///
    /// The marker writes the (slot + 1) value into the buffer's slot at
    /// the GPU pipeline stage boundary specified by `stage`. After
    /// `vkQueueWaitIdle` (or any synchronization), reading the buffer
    /// reveals exactly which markers reached the GPU.
    /// On non-AMD hardware this is a no-op.
    pub fn cmd_buffer_marker(
        &self,
        rec: &CommandRecorder<'_>,
        markers: &Arc<AmdMarkerBuffer>,
        label: &str,
        stage: vk::PipelineStageFlags,
    ) -> u32 {
        let slot = markers.next_slot.fetch_add(1, Ordering::Relaxed);
        if slot >= markers.capacity {
            // Buffer full; we still update labels with a "dropped" marker
            // so readback is consistent in length.
            markers.labels.lock().unwrap().push(MarkerLabel {
                label: format!("{label} (dropped: out of slots)"),
                stage,
            });
            return slot;
        }
        markers.labels.lock().unwrap().push(MarkerLabel {
            label: label.to_string(),
            stage,
        });

        if let Some(amd) = &self.amd_fn {
            unsafe {
                amd.cmd_write_buffer_marker(
                    rec.raw_buffer(),
                    stage,
                    markers.buffer,
                    (slot as u64) * 4,
                    slot + 1,
                );
            }
        }

        slot
    }

    /// Read back the most recent NV checkpoints for the given queue.
    ///
    /// Uses the raw `vkGetQueueCheckpointDataNV` function pointer for
    /// portability: ash's safe wrapper signature differs across patch
    /// versions of 0.38, but the FFI signature is stable. Performs the
    /// canonical two-call query (count, then sized fetch). Returns an
    /// empty vector when the extension is not loaded.
    pub fn collect_checkpoints(&self, queue: vk::Queue) -> Vec<CheckpointEntry> {
        let Some(nv) = &self.nv_fn else {
            return Vec::new();
        };
        unsafe {
            let device_fn = nv.fp();
            let mut count = 0_u32;
            (device_fn.get_queue_checkpoint_data_nv)(
                queue,
                &mut count,
                std::ptr::null_mut(),
            );
            if count == 0 {
                return Vec::new();
            }
            let mut buf: Vec<vk::CheckpointDataNV<'_>> =
                vec![vk::CheckpointDataNV::default(); count as usize];
            (device_fn.get_queue_checkpoint_data_nv)(
                queue,
                &mut count,
                buf.as_mut_ptr(),
            );
            buf.truncate(count as usize);
            buf.iter()
                .map(|c| CheckpointEntry {
                    stage: c.stage,
                    value: c.p_checkpoint_marker as usize as u64,
                })
                .collect()
        }
    }

    /// Query the device for `VK_EXT_device_fault` info.
    ///
    /// Per spec, only valid to call after `VK_ERROR_DEVICE_LOST`. Calling
    /// this on a healthy device is undefined; ignis still calls into the
    /// driver but returns whatever the driver returns. On most drivers
    /// the "no fault" reply is a description like "no fault recorded".
    /// Returns `None` when the extension is not loaded.
    ///
    /// Uses the raw `vkGetDeviceFaultInfoEXT` function pointer because
    /// ash 0.38 does not always generate a safe wrapper for this call.
    pub fn collect_fault_info(&self) -> Option<DeviceFaultInfo> {
        let fault_fn = self.fault_fn.as_ref()?;
        unsafe {
            let device_fn = fault_fn.fp();
            let device_handle = self.shared.device.handle();

            // First call: query counts.
            let mut counts = vk::DeviceFaultCountsEXT::default();
            let result = (device_fn.get_device_fault_info_ext)(
                device_handle,
                &mut counts,
                std::ptr::null_mut(),
            );
            if result != vk::Result::SUCCESS {
                return Some(DeviceFaultInfo {
                    description: format!(
                        "vkGetDeviceFaultInfoEXT counts query returned {:?}",
                        result
                    ),
                    ..Default::default()
                });
            }

            // Second call: fetch the data into pre-sized buffers.
            let mut addr_buf = vec![
                vk::DeviceFaultAddressInfoEXT::default();
                counts.address_info_count as usize
            ];
            let mut vendor_buf = vec![
                vk::DeviceFaultVendorInfoEXT::default();
                counts.vendor_info_count as usize
            ];
            let mut vendor_binary_buf = vec![0_u8; counts.vendor_binary_size as usize];

            let mut info = vk::DeviceFaultInfoEXT::default();
            info.p_address_infos = addr_buf.as_mut_ptr();
            info.p_vendor_infos = vendor_buf.as_mut_ptr();
            info.p_vendor_binary_data = vendor_binary_buf.as_mut_ptr().cast();

            let result = (device_fn.get_device_fault_info_ext)(
                device_handle,
                &mut counts,
                &mut info,
            );
            if result != vk::Result::SUCCESS {
                return Some(DeviceFaultInfo {
                    description: format!(
                        "vkGetDeviceFaultInfoEXT data query returned {:?}",
                        result
                    ),
                    ..Default::default()
                });
            }

            let description = c_array_to_string(info.description.as_ptr());
            let address_infos = addr_buf
                .into_iter()
                .map(|a| DeviceFaultAddressInfo {
                    address_type: a.address_type,
                    reported_address: a.reported_address,
                    address_precision: a.address_precision,
                })
                .collect();
            let vendor_infos = vendor_buf
                .into_iter()
                .map(|v| DeviceFaultVendorInfo {
                    description: c_array_to_string(v.description.as_ptr()),
                    vendor_fault_code: v.vendor_fault_code,
                    vendor_fault_data: v.vendor_fault_data,
                })
                .collect();

            Some(DeviceFaultInfo {
                description,
                address_infos,
                vendor_infos,
                vendor_binary: vendor_binary_buf,
            })
        }
    }

    /// Aggregate everything available: NV checkpoints (if a queue is
    /// supplied), EXT fault info, and AMD markers (if a buffer is
    /// supplied). Convenience for [`CrashReporter`].
    ///
    /// [`CrashReporter`]: super::crash_report::CrashReporter
    pub fn collect_all(
        &self,
        queue: Option<vk::Queue>,
        markers: Option<&Arc<AmdMarkerBuffer>>,
    ) -> DeviceFaultData {
        let checkpoints = queue
            .map(|q| self.collect_checkpoints(q))
            .unwrap_or_default();
        let fault_info = self.collect_fault_info();
        let buffer_markers = markers.map(|m| m.readback()).unwrap_or_default();
        DeviceFaultData {
            checkpoints,
            fault_info,
            buffer_markers,
            supports_checkpoints: self.supports_checkpoints(),
            supports_fault_info: self.supports_fault_info(),
            supports_buffer_markers: self.supports_buffer_markers(),
        }
    }

    /// Format collected data as a markdown section. Used by [`CrashReporter`]
    /// when generating the unified report.
    ///
    /// [`CrashReporter`]: super::crash_report::CrashReporter
    pub fn format_section(&self, data: &DeviceFaultData) -> String {
        format_device_fault_section(data)
    }
}

/// Aggregate of everything the recorder pulled at one point in time.
#[derive(Debug, Clone, Default)]
pub struct DeviceFaultData {
    /// NV per-queue checkpoint snapshot.
    pub checkpoints: Vec<CheckpointEntry>,
    /// EXT fault info, if the extension is present.
    pub fault_info: Option<DeviceFaultInfo>,
    /// AMD marker readback, if a buffer was supplied.
    pub buffer_markers: Vec<MarkerEntry>,
    /// True if NV checkpoints are loaded.
    pub supports_checkpoints: bool,
    /// True if EXT device fault is loaded.
    pub supports_fault_info: bool,
    /// True if AMD markers are loaded.
    pub supports_buffer_markers: bool,
}

fn probe_proc(entry: &ash::Entry, instance: &ash::Instance, name: &str) -> bool {
    let cname = match CString::new(name) {
        Ok(c) => c,
        Err(_) => return false,
    };
    // vkGetInstanceProcAddr returns non-null for both instance- and
    // device-level extension functions when the corresponding extension
    // is enabled, per Vulkan loader spec. Using only this entry-level
    // probe avoids depending on the exact field layout of EntryFnV1_0
    // versus InstanceFnV1_0 across ash versions.
    unsafe {
        entry
            .get_instance_proc_addr(instance.handle(), cname.as_ptr())
            .is_some()
    }
}

fn find_host_visible(props: &vk::PhysicalDeviceMemoryProperties, bits: u32) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if bits & (1 << i) == 0 {
            continue;
        }
        let f = props.memory_types[i as usize].property_flags;
        if f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) {
            return Some(i);
        }
    }
    None
}

unsafe fn c_array_to_string(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

/// Build the markdown section for one DeviceFaultData snapshot. Used both
/// directly and via `CrashReporter::generate`.
fn format_device_fault_section(data: &DeviceFaultData) -> String {
    use std::fmt::Write;
    let mut o = String::with_capacity(2048);
    let _ = writeln!(o, "## Device Fault Diagnostics\n");
    let _ = writeln!(
        o,
        "- VK_EXT_device_fault: **{}**",
        if data.supports_fault_info { "loaded" } else { "not loaded" }
    );
    let _ = writeln!(
        o,
        "- VK_NV_device_diagnostic_checkpoints: **{}**",
        if data.supports_checkpoints { "loaded" } else { "not loaded" }
    );
    let _ = writeln!(
        o,
        "- VK_AMD_buffer_marker: **{}**",
        if data.supports_buffer_markers { "loaded" } else { "not loaded" }
    );
    let _ = writeln!(o);

    if let Some(fi) = &data.fault_info {
        let _ = writeln!(o, "### Fault Info (VK_EXT_device_fault)\n");
        let _ = writeln!(o, "Description: `{}`\n", fi.description);
        if !fi.address_infos.is_empty() {
            let _ = writeln!(o, "**Address faults ({}):**\n", fi.address_infos.len());
            let _ = writeln!(o, "| Type | Address | Granularity |");
            let _ = writeln!(o, "|------|---------|-------------|");
            for a in &fi.address_infos {
                let _ = writeln!(
                    o,
                    "| {:?} | `{:#x}` | {} |",
                    a.address_type, a.reported_address, a.address_precision
                );
            }
            let _ = writeln!(o);
        }
        if !fi.vendor_infos.is_empty() {
            let _ = writeln!(o, "**Vendor info ({}):**\n", fi.vendor_infos.len());
            let _ = writeln!(o, "| Code | Data | Description |");
            let _ = writeln!(o, "|------|------|-------------|");
            for v in &fi.vendor_infos {
                let _ = writeln!(
                    o,
                    "| `{:#x}` | `{:#x}` | {} |",
                    v.vendor_fault_code, v.vendor_fault_data, v.description
                );
            }
            let _ = writeln!(o);
        }
        if !fi.vendor_binary.is_empty() {
            let _ = writeln!(
                o,
                "Vendor binary blob: {} bytes (opaque, attach to vendor support)\n",
                fi.vendor_binary.len()
            );
        }
    }

    if !data.checkpoints.is_empty() {
        let _ = writeln!(
            o,
            "### NV Checkpoints (last per stage, {} entries)\n",
            data.checkpoints.len()
        );
        let _ = writeln!(o, "| Pipeline Stage | Marker Value |");
        let _ = writeln!(o, "|----------------|--------------|");
        for c in &data.checkpoints {
            let _ = writeln!(o, "| {:?} | `{:#x}` |", c.stage, c.value);
        }
        let _ = writeln!(o);
    }

    if !data.buffer_markers.is_empty() {
        let fired = data.buffer_markers.iter().filter(|m| m.fired).count();
        let _ = writeln!(
            o,
            "### AMD Buffer Markers ({} fired / {} total)\n",
            fired,
            data.buffer_markers.len()
        );
        let _ = writeln!(o, "| Slot | Stage | Status | Label |");
        let _ = writeln!(o, "|------|-------|--------|-------|");
        for m in &data.buffer_markers {
            let status = if m.fired { "fired" } else { "**PENDING**" };
            let _ = writeln!(
                o,
                "| {} | {:?} | {} | {} |",
                m.slot, m.stage, status, m.label
            );
        }
        let _ = writeln!(o);
    }

    if data.checkpoints.is_empty()
        && data.buffer_markers.is_empty()
        && data.fault_info.is_none()
    {
        let _ = writeln!(
            o,
            "_No diagnostic data captured. Either no DEVICE_LOST occurred yet, \
             or no extensions are enabled, or no checkpoints/markers were \
             inserted before the fault._"
        );
    }

    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_entry_default_state_means_pending() {
        // Empty data should format as "no diagnostic data captured" rather
        // than panic.
        let data = DeviceFaultData::default();
        let section = format_device_fault_section(&data);
        assert!(section.contains("Device Fault Diagnostics"));
        assert!(section.contains("not loaded"));
        assert!(section.contains("No diagnostic data captured"));
    }

    #[test]
    fn formatter_renders_fired_and_pending_markers() {
        let data = DeviceFaultData {
            buffer_markers: vec![
                MarkerEntry {
                    slot: 0,
                    label: "geometry_done".into(),
                    stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    fired: true,
                    value: 1,
                },
                MarkerEntry {
                    slot: 1,
                    label: "lighting_done".into(),
                    stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    fired: false,
                    value: 0,
                },
            ],
            supports_buffer_markers: true,
            ..Default::default()
        };
        let s = format_device_fault_section(&data);
        assert!(s.contains("geometry_done"));
        assert!(s.contains("lighting_done"));
        assert!(s.contains("fired"));
        assert!(s.contains("**PENDING**"));
        assert!(s.contains("1 fired / 2 total"));
    }

    #[test]
    fn formatter_renders_checkpoints() {
        let data = DeviceFaultData {
            checkpoints: vec![
                CheckpointEntry {
                    stage: vk::PipelineStageFlags::VERTEX_SHADER,
                    value: 0xDEAD,
                },
                CheckpointEntry {
                    stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    value: 0xBEEF,
                },
            ],
            supports_checkpoints: true,
            ..Default::default()
        };
        let s = format_device_fault_section(&data);
        assert!(s.contains("0xdead"));
        assert!(s.contains("0xbeef"));
        assert!(s.contains("VERTEX_SHADER"));
        assert!(s.contains("FRAGMENT_SHADER"));
    }

    #[test]
    fn formatter_renders_fault_info_with_address_and_vendor() {
        let data = DeviceFaultData {
            fault_info: Some(DeviceFaultInfo {
                description: "Page fault on read".into(),
                address_infos: vec![DeviceFaultAddressInfo {
                    address_type: vk::DeviceFaultAddressTypeEXT::READ_INVALID,
                    reported_address: 0xCAFE_BABE,
                    address_precision: 4096,
                }],
                vendor_infos: vec![DeviceFaultVendorInfo {
                    description: "MMU translation".into(),
                    vendor_fault_code: 0x1234,
                    vendor_fault_data: 0x5678,
                }],
                vendor_binary: vec![1, 2, 3, 4],
            }),
            supports_fault_info: true,
            ..Default::default()
        };
        let s = format_device_fault_section(&data);
        assert!(s.contains("Page fault on read"));
        assert!(s.contains("0xcafebabe"));
        assert!(s.contains("4 bytes"));
        assert!(s.contains("MMU translation"));
        assert!(s.contains("0x1234"));
    }

    #[test]
    fn extension_support_flags_default_to_false() {
        let data = DeviceFaultData::default();
        assert!(!data.supports_checkpoints);
        assert!(!data.supports_fault_info);
        assert!(!data.supports_buffer_markers);
    }
}