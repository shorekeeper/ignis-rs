//! Debug window lifecycle and Vulkan plumbing.
//!
//! On Windows, this opens a native window via raw `user32` FFI, creates
//! a `VkSurfaceKHR` and swapchain on it, and runs a render loop on a
//! dedicated worker thread. On other platforms, the public API still
//! compiles but [`DebugWindowBuilder::open`] returns an error.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(target_os = "windows")]
use ash::vk;

use crate::error::{Error, Result};
#[cfg(target_os = "windows")]
use crate::resource_trace::ResourceTrace;
#[cfg(target_os = "windows")]
use crate::AllocationProfiler;
#[cfg(target_os = "windows")]
use crate::Ignis;

#[cfg(not(target_os = "windows"))]
use crate::resource_trace::ResourceTrace;
#[cfg(not(target_os = "windows"))]
use crate::AllocationProfiler;
#[cfg(not(target_os = "windows"))]
use crate::Ignis;

/// Builder for [`DebugWindow`].
pub struct DebugWindowBuilder {
    title: String,
    width: u32,
    height: u32,
    memory_source: Option<Arc<AllocationProfiler>>,
    trace_source: Option<Arc<ResourceTrace>>,
    refresh_hz: u32,
    timeline_window_ms: u64,
}

impl DebugWindowBuilder {
    fn new() -> Self {
        Self {
            title: "Ignis Debug".into(),
            width: 1280,
            height: 720,
            memory_source: None,
            trace_source: None,
            refresh_hz: 30,
            timeline_window_ms: 5_000,
        }
    }

    /// Set the window title.
    pub fn title(mut self, t: impl Into<String>) -> Self {
        self.title = t.into();
        self
    }

    /// Set initial window size in pixels. The window is resizable.
    pub fn size(mut self, w: u32, h: u32) -> Self {
        self.width = w.max(320);
        self.height = h.max(200);
        self
    }

    /// Attach an allocation profiler as the memory panel data source.
    pub fn memory_source(mut self, profiler: Arc<AllocationProfiler>) -> Self {
        self.memory_source = Some(profiler);
        self
    }

    /// Attach a resource trace as the timeline panel data source.
    pub fn trace_source(mut self, trace: Arc<ResourceTrace>) -> Self {
        self.trace_source = Some(trace);
        self
    }

    /// Target redraw frequency in Hz. Capped to a reasonable range.
    pub fn refresh_hz(mut self, hz: u32) -> Self {
        self.refresh_hz = hz.clamp(1, 240);
        self
    }

    /// Time window shown in the timeline panel, in milliseconds. Default 5s.
    pub fn timeline_window_ms(mut self, ms: u64) -> Self {
        self.timeline_window_ms = ms.max(50);
        self
    }

    /// Open the window.
    ///
    /// # Required Vulkan Extensions
    ///
    /// The Ignis context must have been created with surface and
    /// swapchain extensions enabled, since the debug window builds its
    /// own [`Swapchain`](crate::Swapchain) on top of a fresh
    /// `VkSurfaceKHR`. Specifically:
    ///
    /// - Instance: `VK_KHR_surface`, `VK_KHR_win32_surface`
    /// - Device: `VK_KHR_swapchain`
    ///
    /// Add them via `ManagedConfig`:
    ///
    /// ```rust,ignore
    /// let ctx = Ignis::managed(
    ///     ManagedConfig::new("app", vk::API_VERSION_1_2)
    ///         .instance_extension(ash::khr::surface::NAME)
    ///         .instance_extension(ash::khr::win32_surface::NAME)
    ///         .device_extension(ash::khr::swapchain::NAME),
    /// )?;
    /// ```
    ///
    /// If the extensions are not available, this method returns
    /// [`Error::FeatureNotEnabled`] with a clear message rather than
    /// panicking deep inside the swapchain machinery.
    #[cfg(target_os = "windows")]
    pub fn open(self, ignis: &Ignis) -> Result<DebugWindow> {
        windows::open(ignis, self)
    }

    /// Open the window.
    #[cfg(not(target_os = "windows"))]
    pub fn open(self, _ignis: &Ignis) -> Result<DebugWindow> {
        Err(Error::InvalidConfig(
            "debug-window: only Windows is implemented at the moment. \
             Linux/macOS support requires platform-specific window code \
             that has not been written yet. PRs welcome.",
        ))
    }
}

/// Live debug window handle.
///
/// Construct via [`DebugWindow::builder`]. Drop to close the window.
pub struct DebugWindow {
    inner: Option<Inner>,
}

struct Inner {
    shutdown: Arc<AtomicBool>,
    closed: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl DebugWindow {
    /// Create a new builder.
    pub fn builder() -> DebugWindowBuilder {
        DebugWindowBuilder::new()
    }

    /// Whether the user closed the window.
    pub fn is_closed(&self) -> bool {
        self.inner
            .as_ref()
            .map(|i| i.closed.load(Ordering::Relaxed))
            .unwrap_or(true)
    }

    /// Request the window to close. The worker thread will tear down on
    /// the next message pump cycle. The handle remains valid until
    /// dropped; further calls become no-ops once the worker exits.
    pub fn close(&self) {
        if let Some(i) = self.inner.as_ref() {
            i.shutdown.store(true, Ordering::Relaxed);
        }
    }
}

impl Drop for DebugWindow {
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            inner.shutdown.store(true, Ordering::Relaxed);
            if let Some(h) = inner.handle.take() {
                let _ = h.join();
            }
        }
    }
}

// ---- Windows implementation ---------------------------------------------

#[cfg(target_os = "windows")]
mod windows {
    use super::*;
    use std::ffi::c_void;
    use std::sync::Arc;

    use ash::vk::Handle;

    use crate::debug_window::panels;
    use crate::debug_window::raster::{palette, Framebuffer};

    type HWND = *mut c_void;
    type HINSTANCE = *mut c_void;
    type HCURSOR = *mut c_void;
    type HBRUSH = *mut c_void;
    type LRESULT = isize;
    type WPARAM = usize;
    type LPARAM = isize;

    const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
    const WS_VISIBLE: u32 = 0x1000_0000;
    const CW_USEDEFAULT: i32 = -2_147_483_648;
    const SW_SHOW: i32 = 5;
    const WM_CLOSE: u32 = 0x0010;
    const WM_DESTROY: u32 = 0x0002;
    const WM_QUIT: u32 = 0x0012;
    const WM_SIZE: u32 = 0x0005;
    const PM_REMOVE: u32 = 0x0001;
    const IDC_ARROW: *const u16 = 32512 as *const u16;
    const COLOR_WINDOW: i32 = 5;
    const CS_OWNDC: u32 = 0x0020;
    const CS_HREDRAW: u32 = 0x0002;
    const CS_VREDRAW: u32 = 0x0001;
    const GWLP_USERDATA: i32 = -21;

    type WndProc = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;

    #[repr(C)]
    struct WNDCLASSEXW {
        cb_size: u32,
        style: u32,
        lpfn_wnd_proc: Option<WndProc>,
        cb_cls_extra: i32,
        cb_wnd_extra: i32,
        h_instance: HINSTANCE,
        h_icon: *mut c_void,
        h_cursor: HCURSOR,
        hbr_background: HBRUSH,
        lpsz_menu_name: *const u16,
        lpsz_class_name: *const u16,
        h_icon_sm: *mut c_void,
    }

    #[repr(C)]
    struct POINT {
        x: i32,
        y: i32,
    }

    #[repr(C)]
    struct MSG {
        hwnd: HWND,
        message: u32,
        w_param: WPARAM,
        l_param: LPARAM,
        time: u32,
        pt: POINT,
    }

    #[link(name = "user32")]
    extern "system" {
        fn RegisterClassExW(c: *const WNDCLASSEXW) -> u16;
        fn CreateWindowExW(
            ex: u32,
            class: *const u16,
            title: *const u16,
            style: u32,
            x: i32,
            y: i32,
            w: i32,
            h: i32,
            parent: HWND,
            menu: *mut c_void,
            inst: HINSTANCE,
            param: *mut c_void,
        ) -> HWND;
        fn DefWindowProcW(h: HWND, m: u32, w: WPARAM, l: LPARAM) -> LRESULT;
        fn ShowWindow(h: HWND, s: i32) -> i32;
        fn PeekMessageW(m: *mut MSG, h: HWND, mn: u32, mx: u32, r: u32) -> i32;
        fn TranslateMessage(m: *const MSG) -> i32;
        fn DispatchMessageW(m: *const MSG) -> LRESULT;
        fn DestroyWindow(h: HWND) -> i32;
        fn PostQuitMessage(c: i32);
        fn LoadCursorW(i: HINSTANCE, n: *const u16) -> HCURSOR;
        fn SetWindowLongPtrW(h: HWND, idx: i32, v: isize) -> isize;
        fn GetWindowLongPtrW(h: HWND, idx: i32) -> isize;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleW(name: *const u16) -> HINSTANCE;
    }

    fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Per-window mutable state pointed to by GWLP_USERDATA.
    struct WindowState {
        closed: Arc<AtomicBool>,
        size: Arc<AtomicSize>,
    }

    /// Atomic (width, height) packed into a single u64.
    struct AtomicSize(std::sync::atomic::AtomicU64);
    impl AtomicSize {
        fn new(w: u32, h: u32) -> Self {
            Self(std::sync::atomic::AtomicU64::new(
                ((w as u64) << 32) | (h as u64),
            ))
        }
        fn load(&self) -> (u32, u32) {
            let v = self.0.load(Ordering::Relaxed);
            ((v >> 32) as u32, (v & 0xFFFF_FFFF) as u32)
        }
        fn store(&self, w: u32, h: u32) {
            self.0
                .store(((w as u64) << 32) | (h as u64), Ordering::Relaxed);
        }
    }

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        w: WPARAM,
        l: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_CLOSE => {
                let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
                if !ptr.is_null() {
                    (*ptr).closed.store(true, Ordering::Relaxed);
                }
                DestroyWindow(hwnd);
                0
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                0
            }
            WM_SIZE => {
                let w_px = (l & 0xFFFF) as u32;
                let h_px = ((l >> 16) & 0xFFFF) as u32;
                let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
                if !ptr.is_null() && w_px > 0 && h_px > 0 {
                    (*ptr).size.store(w_px, h_px);
                }
                0
            }
            _ => DefWindowProcW(hwnd, msg, w, l),
        }
    }

    /// Open the window and start the render thread.
    pub fn open(ignis: &Ignis, cfg: DebugWindowBuilder) -> Result<DebugWindow> {
        let shared = ignis.shared_state().clone();
        let gfx_queue = ignis.queue(crate::QueueType::Graphics)?;
        let queue_family = gfx_queue.family_index();
        let queue_index = gfx_queue.queue_index();

        let shutdown = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));

        let title_wide = to_wide(&cfg.title);
        let class_wide = to_wide("ignis_debug_window");

        let init_w = cfg.width;
        let init_h = cfg.height;
        let size = Arc::new(AtomicSize::new(init_w, init_h));

        let memory_source = cfg.memory_source.clone();
        let trace_source = cfg.trace_source.clone();
        let refresh_hz = cfg.refresh_hz;
        let timeline_window_ns = cfg.timeline_window_ms * 1_000_000;

        let t_shutdown = Arc::clone(&shutdown);
        let t_closed = Arc::clone(&closed);
        let t_size = Arc::clone(&size);

        let handle = std::thread::Builder::new()
            .name("ignis-debug-window".into())
            .spawn(move || {
                if let Err(e) = run_thread(
                    shared,
                    queue_family,
                    queue_index,
                    title_wide,
                    class_wide,
                    init_w,
                    init_h,
                    t_size,
                    t_shutdown,
                    t_closed.clone(),
                    memory_source,
                    trace_source,
                    refresh_hz,
                    timeline_window_ns,
                ) {
                    eprintln!("ignis debug-window thread error: {e}");
                    t_closed.store(true, Ordering::Relaxed);
                }
            })
            .map_err(|_| Error::InvalidConfig("failed to spawn debug-window thread"))?;

        Ok(DebugWindow {
            inner: Some(Inner {
                shutdown,
                closed,
                handle: Some(handle),
            }),
        })
    }

    /// Worker thread: create window + Vulkan resources, render loop, teardown.
    fn run_thread(
        shared: Arc<crate::device::SharedState>,
        queue_family: u32,
        queue_index: u32,
        title_wide: Vec<u16>,
        class_wide: Vec<u16>,
        init_w: u32,
        init_h: u32,
        size: Arc<AtomicSize>,
        shutdown: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
        memory_source: Option<Arc<AllocationProfiler>>,
        trace_source: Option<Arc<ResourceTrace>>,
        refresh_hz: u32,
        timeline_window_ns: u64,
    ) -> Result<()> {
        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSEXW {
                cb_size: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_OWNDC | CS_HREDRAW | CS_VREDRAW,
                lpfn_wnd_proc: Some(wnd_proc),
                cb_cls_extra: 0,
                cb_wnd_extra: 0,
                h_instance: hinstance,
                h_icon: std::ptr::null_mut(),
                h_cursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbr_background: (COLOR_WINDOW as usize + 1) as HBRUSH,
                lpsz_menu_name: std::ptr::null(),
                lpsz_class_name: class_wide.as_ptr(),
                h_icon_sm: std::ptr::null_mut(),
            };
            // Class is global per process; ignore "already registered" by
            // attempting registration and not returning on zero (other
            // tests may have already created it).
            let _ = RegisterClassExW(&wc);

            let hwnd = CreateWindowExW(
                0,
                class_wide.as_ptr(),
                title_wide.as_ptr(),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                init_w as i32,
                init_h as i32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return Err(Error::InvalidConfig("CreateWindowExW failed"));
            }

            // Attach window state via GWLP_USERDATA so wnd_proc can update
            // closed/size atomics from any thread.
            let state = Box::new(WindowState {
                closed: Arc::clone(&closed),
                size: Arc::clone(&size),
            });
            let state_ptr = Box::into_raw(state);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, state_ptr as isize);

            ShowWindow(hwnd, SW_SHOW);

            let result = render_loop(
                hwnd,
                hinstance,
                shared,
                queue_family,
                queue_index,
                size,
                shutdown,
                closed.clone(),
                memory_source,
                trace_source,
                refresh_hz,
                timeline_window_ns,
            );

            // Ensure window is destroyed if it survived.
            DestroyWindow(hwnd);
            // Reclaim WindowState.
            let _ = Box::from_raw(state_ptr);

            closed.store(true, Ordering::Relaxed);
            result
        }
    }

    fn render_loop(
        hwnd: HWND,
        hinstance: HINSTANCE,
        shared: Arc<crate::device::SharedState>,
        queue_family: u32,
        queue_index: u32,
        size: Arc<AtomicSize>,
        shutdown: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
        memory_source: Option<Arc<AllocationProfiler>>,
        trace_source: Option<Arc<ResourceTrace>>,
        refresh_hz: u32,
        timeline_window_ns: u64,
    ) -> Result<()> {
        // Build VkSurfaceKHR via VK_KHR_win32_surface. ash 0.38 panics
        // deep in its dispatch table when the function pointer is null
        // (extension not enabled at instance creation), so we probe
        // vkGetInstanceProcAddr ourselves first and produce a clean
        // diagnostic instead.
        let entry = unsafe { ash::Entry::load() }.map_err(|_| Error::LoadFailed)?;
        unsafe {
            let probe = std::ffi::CString::new("vkCreateWin32SurfaceKHR").unwrap();
            if entry
                .get_instance_proc_addr(shared.instance.handle(), probe.as_ptr())
                .is_none()
            {
                return Err(Error::FeatureNotEnabled(
                    "VK_KHR_surface, VK_KHR_win32_surface, and VK_KHR_swapchain must be \
                     enabled on the Ignis context. Add them via ManagedConfig: \
                     .instance_extension(ash::khr::surface::NAME) \
                     .instance_extension(ash::khr::win32_surface::NAME) \
                     .device_extension(ash::khr::swapchain::NAME)",
                ));
            }
            let probe = std::ffi::CString::new("vkCreateSwapchainKHR").unwrap();
            if shared
                .instance
                .get_device_proc_addr(shared.device.handle(), probe.as_ptr())
                .is_none()
            {
                return Err(Error::FeatureNotEnabled(
                    "VK_KHR_swapchain must be enabled on the Ignis device. Add via \
                     ManagedConfig::device_extension(ash::khr::swapchain::NAME)",
                ));
            }
        }

        let surface_fn = ash::khr::surface::Instance::new(&entry, &shared.instance);
        let win32_fn = ash::khr::win32_surface::Instance::new(&entry, &shared.instance);

        let surface = unsafe {
            win32_fn.create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(hinstance as isize)
                    .hwnd(hwnd as isize),
                None,
            )?
        };

        // Build swapchain helper (same path as animated_window example).
        let (init_w, init_h) = size.load();
        let swap_cfg = crate::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST,
            preferred_present_mode: vk::PresentModeKHR::FIFO,
            image_count: 3,
            ..Default::default()
        };
        let mut swap = crate::Swapchain::new(
            Arc::clone(&shared),
            surface,
            &swap_cfg,
            init_w,
            init_h,
        )?;

        // Per-frame infrastructure.
        let frames_in_flight = swap.image_count().min(2);
        let frame_sync = crate::FrameSync::new(Arc::clone(&shared), frames_in_flight)?;
        let mut pools = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            pools.push(crate::CommandPool::new(Arc::clone(&shared), queue_family)?);
        }

        // Raw queue handle (used only on this thread).
        let raw_queue = unsafe { shared.device.get_device_queue(queue_family, queue_index) };

        // Persistent CPU framebuffer + staging buffer; both grow as the
        // window resizes.
        let mut fb = Framebuffer::new(swap.extent().width, swap.extent().height);
        let mut staging = create_staging(&shared, fb.byte_len() as u64)?;

        let mut last_size = swap.extent();
        let frame_dur = Duration::from_micros(1_000_000 / (refresh_hz as u64).max(1));

        loop {
            if shutdown.load(Ordering::Relaxed) || closed.load(Ordering::Relaxed) {
                break;
            }

            // Pump messages.
            unsafe {
                let mut msg: MSG = std::mem::zeroed();
                while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                    if msg.message == WM_QUIT {
                        closed.store(true, Ordering::Relaxed);
                        break;
                    }
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }

            if closed.load(Ordering::Relaxed) {
                break;
            }

            // Resize handling: if user-reported size differs from swapchain.
            let (w, h) = size.load();
            if w > 0
                && h > 0
                && (w != swap.extent().width || h != swap.extent().height)
            {
                frame_sync.wait_all()?;
                swap.recreate(w, h)?;
                fb.resize(swap.extent().width, swap.extent().height);
                staging = create_staging(&shared, fb.byte_len() as u64)?;
                last_size = swap.extent();
            }
            let _ = last_size;

            // Render the frame into the CPU framebuffer.
            paint(&mut fb, &memory_source, &trace_source, timeline_window_ns);

            // Begin frame, acquire image, copy bitmap, present.
            let frame = frame_sync.begin_frame()?;
            let pool = &pools[frame.frame_index() as usize];
            pool.reset()?;

            let (image_idx, _) = match swap.acquire_next_image(
                u64::MAX,
                frame.image_available_semaphore(),
                vk::Fence::null(),
            ) {
                Ok(v) => v,
                Err(Error::SwapchainOutOfDate) => {
                    frame_sync.advance();
                    continue;
                }
                Err(e) => return Err(e),
            };

            // Upload bitmap to staging.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    fb.bytes().as_ptr(),
                    staging.mapped,
                    fb.byte_len(),
                );
            }

            // Record command buffer.
            let cmd = pool.allocate_primary()?;
            let rec = pool.begin_primary(cmd)?;
            let swap_image = swap.images()[image_idx as usize];

            // UNDEFINED -> TRANSFER_DST_OPTIMAL.
            let to_transfer = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(swap_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            rec.pipeline_barrier(
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_transfer),
            );

            rec.copy_buffer_to_image(
                staging.buffer,
                swap_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width: swap.extent().width,
                        height: swap.extent().height,
                        depth: 1,
                    },
                }],
            );

            // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR.
            let to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .image(swap_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            rec.pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_present),
            );

            let cmd = rec.end()?;

            // Submit.
            let cmds = [cmd];
            let waits = [frame.image_available_semaphore()];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let signals = [swap.render_complete_semaphore(image_idx)];
            let submit_info = [vk::SubmitInfo::default()
                .command_buffers(&cmds)
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .signal_semaphores(&signals)];
            unsafe {
                shared
                    .device
                    .queue_submit(raw_queue, &submit_info, frame.fence())?;
            }

            // Present.
            match swap.present(raw_queue, image_idx, &signals) {
                Ok(_) => {}
                Err(Error::SwapchainOutOfDate) => {
                    // Loop will recreate next iteration.
                }
                Err(e) => return Err(e),
            }
            frame_sync.advance();

            // Sleep to honor refresh_hz cap.
            std::thread::sleep(frame_dur);
        }

        // Teardown: wait idle, drop swapchain, destroy surface, drop staging.
        unsafe { shared.device.device_wait_idle().ok() };
        drop(swap);
        unsafe { surface_fn.destroy_surface(surface, None) };
        destroy_staging(&shared, &staging);
        Ok(())
    }

    /// Persistent host-visible staging buffer mapped for direct CPU writes.
    struct StagingBuffer {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        mapped: *mut u8,
        size: u64,
    }
    // SAFETY: pointer is valid for the buffer's lifetime; sole producer
    // is the worker thread.
    unsafe impl Send for StagingBuffer {}

    fn create_staging(
        shared: &Arc<crate::device::SharedState>,
        size: u64,
    ) -> Result<StagingBuffer> {
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { shared.device.create_buffer(&ci, None)? };
        let req = unsafe { shared.device.get_buffer_memory_requirements(buffer) };
        let mt = find_host_visible(&shared.memory_properties, req.memory_type_bits)
            .ok_or(Error::NoSuitableMemoryType)?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mt);
        let memory = unsafe { shared.device.allocate_memory(&alloc_info, None)? };
        unsafe { shared.device.bind_buffer_memory(buffer, memory, 0)? };
        let ptr = unsafe {
            shared
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
        }
        .cast::<u8>();
        Ok(StagingBuffer {
            buffer,
            memory,
            mapped: ptr,
            size,
        })
    }

    fn destroy_staging(shared: &Arc<crate::device::SharedState>, sb: &StagingBuffer) {
        unsafe {
            shared.device.unmap_memory(sb.memory);
            shared.device.destroy_buffer(sb.buffer, None);
            shared.device.free_memory(sb.memory, None);
        }
    }

    fn find_host_visible(
        props: &vk::PhysicalDeviceMemoryProperties,
        bits: u32,
    ) -> Option<u32> {
        for i in 0..props.memory_type_count {
            if bits & (1 << i) == 0 {
                continue;
            }
            let f = props.memory_types[i as usize].property_flags;
            if f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && f.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                return Some(i);
            }
        }
        None
    }

    /// Render the full frame: clear, draw memory panel on top, timeline below.
    fn paint(
        fb: &mut Framebuffer,
        memory: &Option<Arc<AllocationProfiler>>,
        trace: &Option<Arc<ResourceTrace>>,
        timeline_window_ns: u64,
    ) {
        fb.clear(palette::BG);

        let pad = 8_i32;
        let total_w = fb.width() as i32;
        let total_h = fb.height() as i32;

        match (memory, trace) {
            (Some(m), Some(t)) => {
                // Top half memory, bottom half timeline.
                let mid = total_h / 2;
                panels::render_memory_panel(
                    fb,
                    pad,
                    pad,
                    total_w - 2 * pad,
                    mid - pad,
                    m,
                );
                panels::render_timeline_panel(
                    fb,
                    pad,
                    mid + pad / 2,
                    total_w - 2 * pad,
                    total_h - mid - pad,
                    t,
                    timeline_window_ns,
                );
            }
            (Some(m), None) => panels::render_memory_panel(
                fb,
                pad,
                pad,
                total_w - 2 * pad,
                total_h - 2 * pad,
                m,
            ),
            (None, Some(t)) => panels::render_timeline_panel(
                fb,
                pad,
                pad,
                total_w - 2 * pad,
                total_h - 2 * pad,
                t,
                timeline_window_ns,
            ),
            (None, None) => {
                // Empty configuration. Show a single message.
                fb.text(
                    pad + 8,
                    pad + 8,
                    "DEBUG WINDOW: no data sources configured",
                    palette::TEXT,
                );
            }
        }
    }
}

// silences unused-import warnings in the cfg-disabled path
#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
fn _unused_atomics() {
    let _ = AtomicU32::new(0);
    let _ = Mutex::new(0_u32);
}