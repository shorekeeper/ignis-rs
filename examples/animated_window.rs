//! Self-contained Win32 graphical demo using ignis.
//!
//! Shows a window displaying an animated procedural pattern, generated
//! on the CPU each frame and copied into the swapchain image via
//! vkCmdCopyBufferToImage. No shaders or pipelines involved -  just
//! transfer commands - which keeps the example focused on queue
//! orchestration, synchronization, and swapchain lifecycle.
//!
//! Run with:
//! ```sh
//! cargo run --example animated_window --features full
//! ```
//!
//! Press Esc or close the window to exit.

#[cfg(not(feature = "full"))]
compile_error!("animated_window requires --features full");

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("animated_window is Windows-only in this example.");
    eprintln!("Port to Linux needs xlib/xcb + VK_KHR_xlib_surface.");
    std::process::exit(0);
}

#[cfg(target_os = "windows")]
use std::sync::atomic::Ordering;

#[cfg(target_os = "windows")]
use ash::vk;

#[cfg(target_os = "windows")]
fn main() {
    if let Err(e) = run() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}

// Win32 FFI 
#[cfg(target_os = "windows")]
mod win {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    pub type HWND = *mut c_void;
    pub type HINSTANCE = *mut c_void;
    pub type HICON = *mut c_void;
    pub type HCURSOR = *mut c_void;
    pub type HBRUSH = *mut c_void;
    pub type HMENU = *mut c_void;
    pub type LRESULT = isize;
    pub type WPARAM = usize;
    pub type LPARAM = isize;

    pub const WS_OVERLAPPEDWINDOW: u32 = 0x00CF0000;
    pub const WS_VISIBLE: u32 = 0x10000000;
    pub const CW_USEDEFAULT: i32 = -2147483648;
    pub const SW_SHOW: i32 = 5;
    pub const WM_CLOSE: u32 = 0x0010;
    pub const WM_DESTROY: u32 = 0x0002;
    pub const WM_QUIT: u32 = 0x0012;
    pub const WM_SIZE: u32 = 0x0005;
    pub const WM_KEYDOWN: u32 = 0x0100;
    pub const VK_ESCAPE: u32 = 0x1B;
    pub const PM_REMOVE: u32 = 0x0001;
    pub const IDC_ARROW: *const u16 = 32512 as *const u16;
    pub const COLOR_WINDOW: i32 = 5;
    pub const CS_OWNDC: u32 = 0x0020;
    pub const CS_HREDRAW: u32 = 0x0002;
    pub const CS_VREDRAW: u32 = 0x0001;

    pub type WndProc = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;

    #[repr(C)]
    pub struct WNDCLASSEXW {
        pub cb_size: u32,
        pub style: u32,
        pub lpfn_wnd_proc: Option<WndProc>,
        pub cb_cls_extra: i32,
        pub cb_wnd_extra: i32,
        pub h_instance: HINSTANCE,
        pub h_icon: HICON,
        pub h_cursor: HCURSOR,
        pub hbr_background: HBRUSH,
        pub lpsz_menu_name: *const u16,
        pub lpsz_class_name: *const u16,
        pub h_icon_sm: HICON,
    }

    #[repr(C)]
    pub struct POINT { pub x: i32, pub y: i32 }

    #[repr(C)]
    pub struct MSG {
        pub hwnd: HWND,
        pub message: u32,
        pub w_param: WPARAM,
        pub l_param: LPARAM,
        pub time: u32,
        pub pt: POINT,
    }

    #[link(name = "user32")]
    extern "system" {
        pub fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> u16;
        pub fn CreateWindowExW(
            dw_ex_style: u32, lp_class_name: *const u16, lp_window_name: *const u16,
            dw_style: u32, x: i32, y: i32, n_width: i32, n_height: i32,
            h_wnd_parent: HWND, h_menu: HMENU, h_instance: HINSTANCE,
            lp_param: *mut c_void,
        ) -> HWND;
        pub fn DefWindowProcW(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT;
        pub fn ShowWindow(hwnd: HWND, n_cmd_show: i32) -> i32;
        pub fn PeekMessageW(
            lp_msg: *mut MSG, hwnd: HWND,
            w_msg_filter_min: u32, w_msg_filter_max: u32, w_remove_msg: u32,
        ) -> i32;
        pub fn TranslateMessage(lp_msg: *const MSG) -> i32;
        pub fn DispatchMessageW(lp_msg: *const MSG) -> LRESULT;
        pub fn DestroyWindow(hwnd: HWND) -> i32;
        pub fn PostQuitMessage(n_exit_code: i32);
        pub fn LoadCursorW(h_instance: HINSTANCE, lp_cursor_name: *const u16) -> HCURSOR;
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetModuleHandleW(lp_module_name: *const u16) -> HINSTANCE;
    }

    pub fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    // Shared state populated by the WndProc callback.
    pub static WINDOW_CLOSED: AtomicBool = AtomicBool::new(false);
    pub static WINDOW_RESIZED: AtomicBool = AtomicBool::new(false);
    pub static WINDOW_WIDTH: AtomicU32 = AtomicU32::new(1024);
    pub static WINDOW_HEIGHT: AtomicU32 = AtomicU32::new(768);

    pub unsafe extern "system" fn wnd_proc(
        hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_CLOSE => {
                WINDOW_CLOSED.store(true, Ordering::SeqCst);
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
                if w_px > 0 && h_px > 0 {
                    WINDOW_WIDTH.store(w_px, Ordering::SeqCst);
                    WINDOW_HEIGHT.store(h_px, Ordering::SeqCst);
                    WINDOW_RESIZED.store(true, Ordering::SeqCst);
                }
                0
            }
            WM_KEYDOWN => {
                if w as u32 == VK_ESCAPE {
                    WINDOW_CLOSED.store(true, Ordering::SeqCst);
                    DestroyWindow(hwnd);
                }
                0
            }
            _ => DefWindowProcW(hwnd, msg, w, l),
        }
    }
}

// Entry point 
#[cfg(target_os = "windows")]
fn run() -> ignis::Result<()> {
    println!("=== ignis animated_window demo ===\n");

    // 1. Win32 window
    let (hwnd, hinstance) = create_window()?;
    println!("[1/6] window created: {}x{}",
        win::WINDOW_WIDTH.load(Ordering::SeqCst),
        win::WINDOW_HEIGHT.load(Ordering::SeqCst));

    // 2. Ignis context in MANAGED mode.
    //    Managed = ignis creates VkInstance, picks GPU, creates VkDevice.
    //    All of it is destroyed automatically when `ctx` is dropped.
    //    The instance_extension and device_extension calls are what make
    //    on-screen rendering possible; without them we couldn't create
    //    a surface or a swapchain.
    let ctx = ignis::Ignis::managed(
        ignis::ManagedConfig::new("ignis-animated", vk::API_VERSION_1_2)
            .enable_validation(cfg!(debug_assertions))
            .instance_extension(ash::khr::surface::NAME)
            .instance_extension(ash::khr::win32_surface::NAME)
            .device_extension(ash::khr::swapchain::NAME),
    )?;

    let dev_name = unsafe {
        std::ffi::CStr::from_ptr(ctx.device_properties().device_name.as_ptr())
    }
    .to_str()
    .unwrap_or("<unknown>");
    let api = ctx.device_properties().api_version;
    println!("[2/6] ignis context: {dev_name}  Vulkan {}.{}.{}",
        vk::api_version_major(api),
        vk::api_version_minor(api),
        vk::api_version_patch(api));

    // 3. Create VkSurfaceKHR manually.
    //    Ignis does not create or own surfaces - that lives entirely
    //    in user code. We load an ash::Entry just to construct the
    //    surface and win32_surface function loaders; this is cheap
    //    and idempotent since Vulkan is already loaded.
    let entry = unsafe { ash::Entry::load() }.map_err(|_| ignis::Error::LoadFailed)?;
    let surface_fn = ash::khr::surface::Instance::new(&entry, ctx.instance());
    let win32_surface_fn =
        ash::khr::win32_surface::Instance::new(&entry, ctx.instance());

    let surface = unsafe {
        win32_surface_fn.create_win32_surface(
            &vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(hinstance as isize)
                .hwnd(hwnd as isize),
            None,
        )?
    };
    println!("[3/6] VkSurfaceKHR created: {surface:?}");

    // 4. Swapchain.
    //    Default SwapchainConfig sets image_usage = COLOR_ATTACHMENT.
    //    We need TRANSFER_DST too because we'll copy our procedural
    //    buffer directly into the swapchain image. If you forget this,
    //    validation will flag it clearly (ignis forensic diagnostic
    //    decodes the VUID automatically on debug builds).
    let swap_cfg = ignis::SwapchainConfig {
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_DST,
        preferred_present_mode: vk::PresentModeKHR::FIFO,
        image_count: 3,
        ..Default::default()
    };

    let mut swap = ctx.create_swapchain(
        surface,
        &swap_cfg,
        win::WINDOW_WIDTH.load(Ordering::SeqCst),
        win::WINDOW_HEIGHT.load(Ordering::SeqCst),
    )?;
    println!("[4/6] swapchain {}x{}  format={:?}  images={}",
        swap.extent().width,
        swap.extent().height,
        swap.format().format,
        swap.image_count());

    // 5. Per-frame infrastructure.
    //    - One command pool per frame so they can be reset independently.
    //    - FrameSync gives us N-in-flight fences + two semaphores per slot
    //      (image_available and render_finished).
    //    - StagingRing gives us a fresh upload budget per frame slot.
    let gfx = ctx.queue(ignis::QueueType::Graphics)?;
    let frames_in_flight: u32 = 2;
    let frame_sync = ctx.create_frame_sync(frames_in_flight)?;

    let pools: Vec<ignis::CommandPool> = (0..frames_in_flight)
        .map(|_| ctx.create_command_pool(ignis::QueueType::Graphics))
        .collect::<ignis::Result<_>>()?;

    // Big enough for 2560x1440x4 = 14 MiB per frame, with slack.
    let mut staging = ctx.create_staging_ring(16 * 1024 * 1024, frames_in_flight)?;

    // Raw queue handle for vkQueuePresentKHR (ignis's submit helpers
    // use a locked wrapper, but present calls the raw handle directly).
    let raw_queue = unsafe {
        ctx.device()
            .get_device_queue(gfx.family_index(), gfx.queue_index())
    };
    println!("[5/6] frame infra: {frames_in_flight} in-flight, staging={} MiB/frame",
        staging.frame_capacity() / (1024 * 1024));

    println!("[6/6] entering render loop (press Esc to quit)\n");

    // 6. Render loop 
    let start = std::time::Instant::now();
    let mut frame_no: u32 = 0;
    let mut cpu_pattern: Vec<u8> = Vec::new();

    while !win::WINDOW_CLOSED.load(Ordering::SeqCst) {
        // Drain Win32 messages (non-blocking).
        pump_messages();

        if win::WINDOW_CLOSED.load(Ordering::SeqCst) {
            break;
        }

        // Resize: wait for GPU idle before recreating the swapchain.
        if win::WINDOW_RESIZED.load(Ordering::SeqCst) {
            let w = win::WINDOW_WIDTH.load(Ordering::SeqCst);
            let h = win::WINDOW_HEIGHT.load(Ordering::SeqCst);
            if w > 0 && h > 0 {
                frame_sync.wait_all()?;
                swap.recreate(w, h)?;
                println!("  -> resized to {w}x{h}");
            }
            win::WINDOW_RESIZED.store(false, Ordering::SeqCst);
            continue;
        }

        // Begin frame: blocks on this slot's fence, then resets it.
        // After this call, the CPU knows the GPU has finished the
        // command buffer submitted N frames ago that used this slot.
        let frame = frame_sync.begin_frame()?;
        let pool = &pools[frame.frame_index() as usize];
        pool.reset()?;

        // Acquire the next swapchain image.
        //   - image_available_semaphore will be signaled when the image
        //     is actually owned by us and safe to render to.
        //   - ERROR_OUT_OF_DATE_KHR means the surface changed under us
        //     (resize, DPI switch, etc). We flag for recreation.
        let (image_idx, _suboptimal) = match swap.acquire_next_image(
            u64::MAX,
            frame.image_available_semaphore(),
            vk::Fence::null(),
        ) {
            Ok(v) => v,
            Err(ignis::Error::SwapchainOutOfDate) => {
                win::WINDOW_RESIZED.store(true, Ordering::SeqCst);
                frame_sync.advance();
                continue;
            }
            Err(e) => return Err(e),
        };

        // Generate animated pattern on the CPU. Cheap enough at this
        // resolution; in a real app you would do this in a compute
        // shader so the CPU doesn't burn cycles on pixel math.
        let extent = swap.extent();
        let bytes = (extent.width * extent.height * 4) as usize;
        cpu_pattern.resize(bytes, 0);
        generate_pattern(
            &mut cpu_pattern,
            extent.width,
            extent.height,
            frame_no,
            swap.format().format,
        );

        // Upload via the staging ring. push() returns the (buffer,
        // offset, size) region we can use as the copy source.
        staging.begin_frame()?;
        let region = staging.push(&cpu_pattern)?;

        // Record the command buffer.
        let cmd = pool.allocate_primary()?;
        let rec = pool.begin_primary(cmd)?;

        let swap_image = swap.images()[image_idx as usize];

        // ResourceTracker handles the layout arithmetic. We declare
        // the initial layout as UNDEFINED (always legal, discards
        // previous contents, which is correct here since we're about
        // to overwrite every pixel). A new tracker per frame because
        // acquire_next_image resets image ownership.
        let mut tracker = ignis::ResourceTracker::new();
        tracker.track_image(
            swap_image,
            vk::ImageLayout::UNDEFINED,
            1, // mip levels
            1, // array layers
            vk::ImageAspectFlags::COLOR,
        );

        // UNDEFINED -> TRANSFER_DST_OPTIMAL
        if let Some(t) =
            tracker.transition_image(swap_image, ignis::ImageUsageContext::TransferDst)
        {
            rec.apply_image_transitions(&[t]);
        }

        // Copy CPU pattern into the swapchain image.
        rec.copy_buffer_to_image(
            region.buffer,
            swap_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset: region.offset,
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
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            }],
        );

        // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR
        if let Some(t) =
            tracker.transition_image(swap_image, ignis::ImageUsageContext::PresentSrc)
        {
            rec.apply_image_transitions(&[t]);
        }

        let cmd = rec.end()?;

        // Submit:
        //   - Wait on image_available at the TRANSFER stage because
        //     that is where we first touch the image.
        //   - Signal render_finished when work completes.
        //   - Signal the frame fence so FrameSync can block on it N
        //     frames later when this slot is recycled.
        let cmds = [cmd];
        let waits = [frame.image_available_semaphore()];
        let stages = [vk::PipelineStageFlags::TRANSFER];
        let signals = [frame.render_finished_semaphore()];
        let submits = [vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&signals)];
        unsafe { gfx.submit_raw(&submits, frame.fence())? };

        // Present after render_finished is signaled by the GPU.
        match swap.present(raw_queue, image_idx, &[frame.render_finished_semaphore()]) {
            Ok(_) => {}
            Err(ignis::Error::SwapchainOutOfDate) => {
                win::WINDOW_RESIZED.store(true, Ordering::SeqCst);
            }
            Err(e) => return Err(e),
        }

        frame_sync.advance();
        frame_no = frame_no.wrapping_add(1);

        // FPS report every 60 frames.
        if frame_no % 60 == 0 {
            let fps = f64::from(frame_no) / start.elapsed().as_secs_f64();
            println!("  frame {frame_no:>5}  fps={fps:>6.1}");
        }
    }

    // Shutdown
    // Wait for any in-flight work before tearing anything down. If
    // we destroy objects while the GPU is still reading them, the
    // hardened/validation stack will flag it loudly (and your driver
    // may hang or crash).
    frame_sync.wait_all()?;

    println!("\nrender loop exited after {frame_no} frames");

    // Explicit destruction order matters:
    //   1. Swapchain owns image views derived from surface + ties to
    //      the device.
    //   2. Surface must be destroyed BEFORE the ignis context drops
    //      the instance, otherwise the validation layer reports
    //      VUID-vkDestroyInstance-instance-00629 (live child object).
    //   3. Window can be destroyed any time after surface.
    //   4. ctx drops at scope end and tears down the device and
    //      instance. Its Drop impl calls vkDeviceWaitIdle once more
    //      as insurance.
    drop(swap);
    unsafe { surface_fn.destroy_surface(surface, None) };
    unsafe { win::DestroyWindow(hwnd) };
    drop(ctx);

    println!("cleanup complete");
    Ok(())
}

#[cfg(target_os = "windows")]
fn pump_messages() {
    let mut msg: win::MSG = unsafe { std::mem::zeroed() };
    while unsafe {
        win::PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, win::PM_REMOVE)
    } != 0
    {
        if msg.message == win::WM_QUIT {
            win::WINDOW_CLOSED.store(true, Ordering::SeqCst);
            break;
        }
        unsafe {
            win::TranslateMessage(&msg);
            win::DispatchMessageW(&msg);
        }
    }
}

#[cfg(target_os = "windows")]
fn create_window() -> ignis::Result<(win::HWND, win::HINSTANCE)> {
    let hinstance = unsafe { win::GetModuleHandleW(std::ptr::null()) };
    let class_name = win::to_wide("ignis_animated_demo");
    let title = win::to_wide("ignis - animated pattern (press Esc to exit)");

    let wc = win::WNDCLASSEXW {
        cb_size: std::mem::size_of::<win::WNDCLASSEXW>() as u32,
        style: win::CS_OWNDC | win::CS_HREDRAW | win::CS_VREDRAW,
        lpfn_wnd_proc: Some(win::wnd_proc),
        cb_cls_extra: 0,
        cb_wnd_extra: 0,
        h_instance: hinstance,
        h_icon: std::ptr::null_mut(),
        h_cursor: unsafe { win::LoadCursorW(std::ptr::null_mut(), win::IDC_ARROW) },
        hbr_background: (win::COLOR_WINDOW as usize + 1) as win::HBRUSH,
        lpsz_menu_name: std::ptr::null(),
        lpsz_class_name: class_name.as_ptr(),
        h_icon_sm: std::ptr::null_mut(),
    };

    if unsafe { win::RegisterClassExW(&wc) } == 0 {
        return Err(ignis::Error::InvalidConfig("RegisterClassExW failed"));
    }

    let hwnd = unsafe {
        win::CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            win::WS_OVERLAPPEDWINDOW | win::WS_VISIBLE,
            win::CW_USEDEFAULT,
            win::CW_USEDEFAULT,
            win::WINDOW_WIDTH.load(Ordering::SeqCst) as i32,
            win::WINDOW_HEIGHT.load(Ordering::SeqCst) as i32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            hinstance,
            std::ptr::null_mut(),
        )
    };
    if hwnd.is_null() {
        return Err(ignis::Error::InvalidConfig("CreateWindowExW failed"));
    }
    unsafe { win::ShowWindow(hwnd, win::SW_SHOW) };
    Ok((hwnd, hinstance))
}

/// Procedural animated pattern. Three channels shift at different rates
/// on top of an XOR base, producing plaid-like bands that scroll.
#[cfg(target_os = "windows")]
fn generate_pattern(
    buf: &mut [u8],
    w: u32,
    h: u32,
    frame: u32,
    format: vk::Format,
) {
    // Swapchain images on Windows are almost always BGRA8; check so we
    // don't end up with swapped red/blue channels.
    let is_bgra = matches!(
        format,
        vk::Format::B8G8R8A8_SRGB
            | vk::Format::B8G8R8A8_UNORM
            | vk::Format::B8G8R8A8_SNORM
    );

    let t = frame as i32;

    for y in 0..h {
        let row = y as i32;
        for x in 0..w {
            let col = x as i32;
            let idx = ((y * w + x) * 4) as usize;

            let r = (col.wrapping_add(t) & 0xFF) as u8;
            let g = (row.wrapping_add(t.wrapping_mul(2)) & 0xFF) as u8;
            let b_base = ((col ^ row).wrapping_add(t.wrapping_mul(3)) & 0xFF) as u8;
            let diag = ((col + row - t * 4) & 0xFF) as u8;
            let b = b_base.saturating_add(diag >> 2);

            if is_bgra {
                buf[idx]     = b;
                buf[idx + 1] = g;
                buf[idx + 2] = r;
                buf[idx + 3] = 255;
            } else {
                buf[idx]     = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
                buf[idx + 3] = 255;
            }
        }
    }
}

// External mode (commented reference)
//
// Everything above uses Ignis::managed(), which means ignis creates and
// owns the VkInstance/VkDevice. For integration with an engine that
// already has them (wgpu, vulkano, a custom engine), use external mode:
//
//     use ash::vk;
//
//     fn build_external(
//         instance: ash::Instance,
//         device: ash::Device,
//         physical: vk::PhysicalDevice,
//         gfx_queue: vk::Queue,
//         gfx_family_index: u32,
//     ) -> ignis::Result<ignis::Ignis> {
//         let ext = ignis::ExternalDeviceInfo {
//             instance,
//             device,
//             physical_device: physical,
//             queue_allocations: vec![ignis::QueueAllocation {
//                 family_index: gfx_family_index,
//                 queue_index: 0,
//                 handle: gfx_queue,
//                 capabilities: vk::QueueFlags::GRAPHICS
//                     | vk::QueueFlags::COMPUTE
//                     | vk::QueueFlags::TRANSFER,
//             }],
//             // Set true ONLY if the external device enabled
//             // VK_KHR_ray_tracing_pipeline and acceleration_structure.
//             enable_raytracing: false,
//         };
//         ignis::Ignis::external(ext)
//     }
//
// Important guarantees of external mode:
//
//   1. ignis NEVER destroys the instance or device. The external owner
//      retains full lifetime control.
//   2. ignis clones ash::Instance / ash::Device handles internally; all
//      the factory methods (create_buffer, create_image, create_pipeline
//      etc.) just work as in managed mode.
//   3. Swapchains and surfaces remain your responsibility in external
//      mode too - same as the managed case above.
//   4. You must enable all the device features/extensions ignis might
//      need (descriptor indexing for BindlessHeap, timeline semaphores
//      for async futures, etc.) at the time YOU create the device.
//      ignis will not and cannot enable them retroactively.
//
// Shared Arc<SharedState> is what makes interop painless - any object
// created through an ignis context (Buffer, Image, CommandPool, ...)
// keeps the device alive until the object itself is dropped, just like
// in managed mode. The only difference is whose Drop destroys the
// VkDevice at the very end - yours (external) or ignis's (managed).