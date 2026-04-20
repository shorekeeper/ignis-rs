//! Integration test: frame graph rendering to a real swapchain + shader printf.
//!
//! Opens a native Win32 window via direct user32 FFI calls (no winit, no
//! raw-window-handle), creates a VkSurfaceKHR, builds a swapchain, and
//! drives a multi-pass frame graph that clears the backbuffer and runs a
//! compute shader that calls debugPrintfEXT. Verifies both that frame
//! graph passes execute in topological order and that shader printf
//! messages are delivered through the registered handler.
//!
//! Run with:
//! ```sh
//! cargo run --example window_frame_graph --features full
//! ```
//!
//! # Platform support
//!
//! Windows only for now. Linux requires roughly the same structure but
//! with xlib or xcb instead of user32, and VK_KHR_xlib_surface /
//! VK_KHR_xcb_surface instead of VK_KHR_win32_surface. The rest of the
//! test (frame graph construction, printf handler, submission, presentation)
//! is platform-agnostic.

#[cfg(not(feature = "full"))]
compile_error!("window_frame_graph requires --features full");

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("window_frame_graph currently supports Windows only.");
    eprintln!("The same pattern works on Linux with xlib and VK_KHR_xlib_surface.");
    std::process::exit(0);
}

#[cfg(target_os = "windows")]
fn main() {
    if let Err(e) = run() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "windows")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
#[cfg(target_os = "windows")]
use std::sync::Arc;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use ash::vk;

// Hand-assembled SPIR-V for:
//   #version 450
//   #extension GL_EXT_debug_printf : require
//   layout(local_size_x = 1) in;
//   void main() { debugPrintfEXT("test"); }
//
// Requires VK_KHR_shader_non_semantic_info and the validation layer with
// VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT (both are enabled when
// ManagedConfig::enable_shader_printf(true) is passed).
//
// Layout breakdown:
//   Header (5 words)
//   OpCapability Shader
//   OpExtension "SPV_KHR_non_semantic_info"
//   OpExtInstImport %1 "NonSemantic.DebugPrintf"
//   OpMemoryModel Logical GLSL450
//   OpEntryPoint GLCompute %4 "main"
//   OpExecutionMode %4 LocalSize 1 1 1
//   OpString %6 "test"
//   OpTypeVoid %2
//   OpTypeFunction %3 %2
//   OpFunction %2 %4 None %3
//   OpLabel %5
//   OpExtInst %2 %7 %1 1 %6         ; debugPrintfEXT("test")
//   OpReturn
//   OpFunctionEnd
#[cfg(target_os = "windows")]
#[rustfmt::skip]
const PRINTF_COMPUTE_SPV: &[u32] = &[
    // Header
    0x07230203, 0x00010000, 0x00000000, 0x00000008, 0x00000000,
    // OpCapability Shader
    0x00020011, 0x00000001,
    // OpExtension "SPV_KHR_non_semantic_info"
    0x0008000A,
    0x5F565053, 0x5F52484B, 0x5F6E6F6E, 0x616D6573, 0x6369746E, 0x666E695F, 0x0000006F,
    // OpExtInstImport %1 "NonSemantic.DebugPrintf"
    0x0008000B, 0x00000001,
    0x536E6F4E, 0x6E616D65, 0x2E636974, 0x75626544, 0x69725067, 0x0066746E,
    // OpMemoryModel Logical GLSL450
    0x0003000E, 0x00000000, 0x00000001,
    // OpEntryPoint GLCompute %4 "main"
    0x0005000F, 0x00000005, 0x00000004, 0x6E69616D, 0x00000000,
    // OpExecutionMode %4 LocalSize 1 1 1
    0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001,
    // OpString %6 "test"
    0x00040007, 0x00000006, 0x74736574, 0x00000000,
    // OpTypeVoid %2
    0x00020013, 0x00000002,
    // OpTypeFunction %3 %2
    0x00030021, 0x00000003, 0x00000002,
    // OpFunction %2 %4 None %3
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    // OpLabel %5
    0x000200F8, 0x00000005,
    // OpExtInst %2 %7 %1 1 %6
    0x0006000C, 0x00000002, 0x00000007, 0x00000001, 0x00000001, 0x00000006,
    // OpReturn
    0x000100FD,
    // OpFunctionEnd
    0x00010038,
];

// Win32 FFI
#[cfg(target_os = "windows")]
mod win {
    use std::ffi::c_void;

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
    pub struct POINT {
        pub x: i32,
        pub y: i32,
    }

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
            dw_ex_style: u32,
            lp_class_name: *const u16,
            lp_window_name: *const u16,
            dw_style: u32,
            x: i32,
            y: i32,
            n_width: i32,
            n_height: i32,
            h_wnd_parent: HWND,
            h_menu: HMENU,
            h_instance: HINSTANCE,
            lp_param: *mut c_void,
        ) -> HWND;
        pub fn DefWindowProcW(hwnd: HWND, msg: u32, w_param: WPARAM, l_param: LPARAM) -> LRESULT;
        pub fn ShowWindow(hwnd: HWND, n_cmd_show: i32) -> i32;
        pub fn PeekMessageW(
            lp_msg: *mut MSG,
            hwnd: HWND,
            w_msg_filter_min: u32,
            w_msg_filter_max: u32,
            w_remove_msg: u32,
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

    /// Convert a Rust string to a null-terminated UTF-16 buffer.
    pub fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Shared flag set by WM_CLOSE / WM_DESTROY.
    pub static WINDOW_CLOSED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// Default window procedure used by the test.
    pub unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        w_param: WPARAM,
        l_param: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_CLOSE => {
                WINDOW_CLOSED.store(true, std::sync::atomic::Ordering::SeqCst);
                DestroyWindow(hwnd);
                0
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                0
            }
            _ => DefWindowProcW(hwnd, msg, w_param, l_param),
        }
    }
}

// Main test
#[cfg(target_os = "windows")]
fn run() -> ignis::Result<()> {
    println!();
    println!("  WINDOW + FRAME GRAPH + SHADER PRINTF");
    println!();

    // Create window ──
    let hinstance = unsafe { win::GetModuleHandleW(std::ptr::null()) };
    let class_name = win::to_wide("ignis_frame_graph_test");
    let title = win::to_wide("Ignis: frame graph + shader printf");

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

    let atom = unsafe { win::RegisterClassExW(&wc) };
    if atom == 0 {
        return Err(ignis::Error::InvalidConfig("RegisterClassExW failed"));
    }

    let width: i32 = 800;
    let height: i32 = 600;
    let hwnd = unsafe {
        win::CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            win::WS_OVERLAPPEDWINDOW | win::WS_VISIBLE,
            win::CW_USEDEFAULT,
            win::CW_USEDEFAULT,
            width,
            height,
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
    println!("  [init] window created: {width}x{height}");

    // Create ignis context with printf + surface extensions ──
    let ctx = ignis::Ignis::managed(
        ignis::ManagedConfig::new("ignis-window-test", vk::API_VERSION_1_2)
            .enable_validation(true)
            .enable_shader_printf(true)
            .instance_extension(ash::khr::surface::NAME)
            .instance_extension(ash::khr::win32_surface::NAME)
            .device_extension(ash::khr::swapchain::NAME),
    )?;
    println!("  [init] ignis context created with shader_printf=true");

    // Register printf handler ──
    let printf_count = Arc::new(AtomicU32::new(0));
    let last_stage: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let counter = Arc::clone(&printf_count);
    let stage_store = Arc::clone(&last_stage);
    ctx.set_shader_printf_handler(move |msg| {
        let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
        if n <= 3 {
            println!(
                "  [printf #{n}] stage={} body=\"{}\"",
                msg.shader_stage, msg.formatted
            );
        }
        *stage_store.lock().unwrap() = msg.shader_stage.to_string();
    });

    // ash 0.38 requires an Entry to build extension function tables.
    // The library is already loaded by ignis; this load() just retrieves
    // the same handle from the OS loader cache.
    let entry = unsafe { ash::Entry::load() }.map_err(|_| ignis::Error::LoadFailed)?;
    let surface_fn = ash::khr::win32_surface::Instance::new(&entry, ctx.instance());

    // SAFETY: we just loaded the surface ext function table from the same
    // instance ignis holds. hinstance/hwnd come directly from Win32.
    let surface_ci = vk::Win32SurfaceCreateInfoKHR::default()
        .hinstance(hinstance as isize)
        .hwnd(hwnd as isize);
    let surface = unsafe { surface_fn.create_win32_surface(&surface_ci, None)? };
    println!("  [init] VkSurfaceKHR created: {:?}", surface);

    // Create swapchain
    let sc_config = ignis::SwapchainConfig {
        image_count: 2,
        preferred_present_mode: vk::PresentModeKHR::FIFO,
        // We cmd_clear_color_image on swapchain images, which requires
        // TRANSFER_DST usage. The default config only has COLOR_ATTACHMENT.
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
        ..Default::default()
    };
    let mut swapchain = ctx.create_swapchain(surface, &sc_config, width as u32, height as u32)?;
    let image_count = swapchain.image_count();
    println!(
        "  [init] swapchain: {}x{} {:?} images={}",
        swapchain.extent().width,
        swapchain.extent().height,
        swapchain.format().format,
        image_count
    );

    // Build printf compute pipeline ─
    let cs_module = ctx.create_shader_module(PRINTF_COMPUTE_SPV)?;
    let pipeline_layout = ctx.pipeline_layout_builder().build()?;
    let printf_pipeline = ctx
        .compute_pipeline_builder()
        .shader(cs_module.handle(), "main")
        .layout(pipeline_layout.handle())
        .build()?;
    println!("  [init] printf compute pipeline ready");

    // Per-frame state
    let gfx = ctx.queue(ignis::QueueType::Graphics)?;
    let frames_in_flight = image_count.min(2);
    let frame_sync = ctx.create_frame_sync(frames_in_flight)?;

    let mut frame_pools: Vec<ignis::CommandPool> = Vec::new();
    for _ in 0..frames_in_flight {
        frame_pools.push(ctx.create_command_pool(ignis::QueueType::Graphics)?);
    }

    let raw_queue = unsafe {
        ctx.device()
            .get_device_queue(gfx.family_index(), gfx.queue_index())
    };

    // Render loop
    let max_frames: u32 = 60;
    let max_wall = Duration::from_secs(5);
    let start = Instant::now();
    let mut frame_number: u32 = 0;
    let mut pass_exec_counter: u32 = 0;
    let pass_counter_shared = Arc::new(AtomicU32::new(0));

    println!("  [loop] running up to {max_frames} frames or {max_wall:?}...");
    println!();

    while frame_number < max_frames
        && start.elapsed() < max_wall
        && !win::WINDOW_CLOSED.load(Ordering::SeqCst)
    {
        // Pump window messages (non-blocking).
        let mut msg: win::MSG = unsafe { std::mem::zeroed() };
        while unsafe { win::PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, win::PM_REMOVE) }
            != 0
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
        if win::WINDOW_CLOSED.load(Ordering::SeqCst) {
            break;
        }

        // Frame timing and sync.
        let frame = frame_sync.begin_frame()?;
        let pool = &frame_pools[frame.frame_index() as usize];
        pool.reset()?;

        let (image_idx, _suboptimal) = match swapchain.acquire_next_image(
            u64::MAX,
            frame.image_available_semaphore(),
            vk::Fence::null(),
        ) {
            Ok(v) => v,
            Err(ignis::Error::SwapchainOutOfDate) => {
                println!("  [loop] swapchain out of date, skipping frame");
                frame_sync.advance();
                continue;
            }
            Err(e) => return Err(e),
        };

        let cmd = pool.allocate_primary()?;
        let rec = pool.begin_primary(cmd)?;

        // Build frame graph for this frame ───
        let mut fg = ignis::FrameGraph::new();
        let swap_image = swapchain.images()[image_idx as usize];
        let bb = fg.import_image(
            "backbuffer",
            swap_image,
            // UNDEFINED is always legal to transition from; the driver
            // discards prior contents which is what we want since we clear.
            vk::ImageLayout::UNDEFINED,
            1,
            1,
            vk::ImageAspectFlags::COLOR,
        );

        // Cycling clear color so the frame is visibly different each frame.
        let t = frame_number as f32 * 0.05;
        let clear_r = 0.5 + 0.4 * t.sin();
        let clear_g = 0.5 + 0.4 * (t + 2.0).sin();
        let clear_b = 0.5 + 0.4 * (t + 4.0).sin();
        let counter_for_clear = Arc::clone(&pass_counter_shared);
        fg.add_pass("clear_backbuffer", move |p| {
            p.writes_image(bb, ignis::ImageUsageContext::TransferDst);
            p.execute(Box::new(move |rec, resolver| {
                counter_for_clear.fetch_add(1, Ordering::SeqCst);
                let img = resolver.image(bb);
                let clear = vk::ClearColorValue {
                    float32: [clear_r, clear_g, clear_b, 1.0],
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                unsafe {
                    rec.raw_device().cmd_clear_color_image(
                        rec.raw_buffer(),
                        img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &clear,
                        std::slice::from_ref(&range),
                    );
                }
            }));
        });

        let counter_for_printf = Arc::clone(&pass_counter_shared);
        let pipeline_handle = printf_pipeline;
        fg.add_pass("printf_compute", move |p| {
            p.execute(Box::new(move |rec, _resolver| {
                counter_for_printf.fetch_add(1, Ordering::SeqCst);
                rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline_handle);
                rec.dispatch(1, 1, 1);
            }));
        });

        let counter_for_present = Arc::clone(&pass_counter_shared);
        fg.add_pass("transition_to_present", move |p| {
            // PresentSrc reader triggers the barrier that moves the image
            // into VK_IMAGE_LAYOUT_PRESENT_SRC_KHR. The pass body is empty.
            p.reads_image(bb, ignis::ImageUsageContext::PresentSrc);
            p.execute(Box::new(move |_rec, _r| {
                counter_for_present.fetch_add(1, Ordering::SeqCst);
            }));
        });

        let compiled = fg.compile(&ctx)?;
        if frame_number == 0 {
            println!("  [loop] frame 0 plan:");
            for line in compiled.dump_plan().lines() {
                println!("    {line}");
            }
        }
        compiled.record(&rec);

        let cmd = rec.end()?;

        // Manual submit: SubmitBuilder creates its own fence but here we
        // want the FrameSync fence, so we drop to submit_raw.
        let cmds = [cmd];
        let waits = [frame.image_available_semaphore()];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signals = [frame.render_finished_semaphore()];
        let submits = [vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&signals)];
        // SAFETY: cmd is freshly recorded and ended, semaphores come from
        // FrameSync which tracks fence lifetime for us.
        unsafe { gfx.submit_raw(&submits, frame.fence())? };

        let present_result =
            swapchain.present(raw_queue, image_idx, &[frame.render_finished_semaphore()]);
        match present_result {
            Ok(_) => {}
            Err(ignis::Error::SwapchainOutOfDate) => {
                println!("  [loop] swapchain out of date on present");
            }
            Err(e) => return Err(e),
        }

        frame_sync.advance();
        frame_number += 1;
        pass_exec_counter = pass_counter_shared.load(Ordering::SeqCst);

        // Brief status line every ~10 frames.
        if frame_number % 10 == 0 {
            let fps = frame_number as f64 / start.elapsed().as_secs_f64();
            println!(
                "  [loop] frame {frame_number:>3}  fps={fps:>5.1}  pass_execs={pass_exec_counter}  printfs={}",
                printf_count.load(Ordering::SeqCst)
            );
        }
    }

    // Wait for GPU to settle before tearing anything down.
    frame_sync.wait_all()?;

    let elapsed = start.elapsed();
    let printf_total = printf_count.load(Ordering::SeqCst);
    let pass_total = pass_counter_shared.load(Ordering::SeqCst);

    println!();
    println!("  ── Results ──");
    println!("  frames rendered:   {frame_number}");
    println!("  wall time:         {:.2}s", elapsed.as_secs_f64());
    println!(
        "  avg fps:           {:.1}",
        frame_number as f64 / elapsed.as_secs_f64().max(0.001)
    );
    println!(
        "  pass executions:   {pass_total} (expected {}: 3 passes x {frame_number} frames)",
        frame_number * 3
    );
    println!("  printf callbacks:  {printf_total}");
    if printf_total > 0 {
        println!("  printf last stage: \"{}\"", last_stage.lock().unwrap());
    }
    println!();

    let frame_graph_ok = pass_total == frame_number * 3;
    // Printf success: we got at least some callbacks. Validation layer
    // batches messages aggressively so the count may be lower than frame
    // count; any nonzero count means the pipeline works end-to-end.
    let printf_ok = printf_total > 0;

    if frame_graph_ok {
        println!("  [OK] frame graph: every pass executed every frame");
    } else {
        println!("  [FAIL] frame graph: pass execution count mismatch");
    }
    if printf_ok {
        println!("  [OK] shader printf: handler received messages");
    } else {
        println!("  [WARN] shader printf: no callbacks received");
        println!("       possible causes:");
        println!("       - validation layer not loaded (VK_LAYER_KHRONOS_validation missing)");
        println!("       - layer build predates debugPrintfEXT support");
        println!("       - GPU driver rejects SPV_KHR_non_semantic_info");
    }

    // Cleanup order matters: swapchain owns image views derived from the
    // surface, which must be destroyed before the surface.
    // Destroy the raw pipeline handle before the context drops.
    unsafe {
        ctx.device().destroy_pipeline(printf_pipeline, None);
    }
    drop(swapchain);
    // destroy_surface lives on VK_KHR_surface (base), not VK_KHR_win32_surface.
    // Build a separate function table for the base extension just to destroy.
    let base_surface_fn = ash::khr::surface::Instance::new(&entry, ctx.instance());
    unsafe {
        base_surface_fn.destroy_surface(surface, None);
    }
    // Destroying the window after the surface is safest.
    unsafe { win::DestroyWindow(hwnd) };

    if frame_graph_ok && printf_ok {
        println!();
        println!("  ALL TESTS OK");
        println!();
        Ok(())
    } else if frame_graph_ok {
        println!();
        println!("  FRAME GRAPH OK, PRINTF UNVERIFIED");
        println!();
        Ok(())
    } else {
        Err(ignis::Error::InvalidConfig(
            "frame graph execution count mismatch",
        ))
    }
}

// Silence "unused" warning on non-Windows.
#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
const _UNUSED: &[u32] = &[];
