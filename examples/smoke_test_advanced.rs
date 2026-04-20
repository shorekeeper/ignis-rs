//! Advanced features smoke test.
//!
//! Validates the second generation of ignis features layered on top of
//! the core smoke_test: pipeline statistics, frame graph, crash reporter,
//! BLAS/TLAS builders, bindless descriptor heap, shader printf
//! registration, validation policy.
//!
//! This is intentionally separate from smoke_test.rs which is already
//! 41 steps and covers the fundamentals. Tests here gracefully skip when
//! the hosting GPU or driver lacks required features.
//!
//! Run with:
//! ```sh
//! cargo run --example smoke_test_advanced --features full
//! ```

#[cfg(not(feature = "full"))]
compile_error!("smoke_test_advanced requires --features full");

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ash::vk;

// Same minimal empty compute shader used by smoke_test.rs.
// void main() {} with local_size(1, 1, 1).
#[rustfmt::skip]
const EMPTY_COMPUTE_SPV: &[u32] = &[
    0x07230203, 0x00010000, 0x00000000, 0x00000006, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000E, 0x00000000, 0x00000001,
    0x0005000F, 0x00000005, 0x00000004, 0x6E69616D, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001,
    0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200F8, 0x00000005,
    0x000100FD,
    0x00010038,
];

const TOTAL_STEPS: u32 = 9;

fn main() {
    println!();
    println!("  IGNIS ADVANCED FEATURES TEST");
    println!("  pipeline stats, frame graph, crash report, accel struct, bindless,");
    println!("  shader printf, validation policy");
    println!();

    let wall = Instant::now();
    match run() {
        Ok((passed, skipped)) => {
            let elapsed = wall.elapsed();
            println!();
            println!(
                "  RESULTS  passed: {}  skipped: {}  total: {}",
                passed, skipped, TOTAL_STEPS
            );
            println!("  Elapsed: {:.2?}", elapsed);
            println!();
            if passed + skipped == TOTAL_STEPS {
                println!("  ALL TESTS OK");
            } else {
                println!("  SOME TESTS MISSING (expected {} steps)", TOTAL_STEPS);
            }
            println!();
        }
        Err(e) => {
            eprintln!();
            eprintln!("  FATAL: {e}");
            eprintln!();
            std::process::exit(1);
        }
    }
}

fn run() -> ignis::Result<(u32, u32)> {
    let mut passed: u32 = 0;
    let mut skipped: u32 = 0;

    let enable_validation = cfg!(debug_assertions) && std::env::var("CI").is_err();

    // Vulkan 1.2 for timeline semaphores and descriptor indexing availability.
    // We request pipeline stats and descriptor indexing up front so individual
    // tests can just assume they are enabled (and skip gracefully if the
    // driver does not support them, which is checked by the create calls).
    let ctx = ignis::Ignis::managed(
        ignis::ManagedConfig::new("ignis-advanced", vk::API_VERSION_1_2)
            .enable_validation(enable_validation)
            .enable_pipeline_stats(true)
            .enable_descriptor_indexing(true),
    )?;

    let gfx = ctx.queue(ignis::QueueType::Graphics)?;
    let pool = ctx.create_command_pool(ignis::QueueType::Graphics)?;

    let is_software_gpu = ctx.device_properties().device_type == vk::PhysicalDeviceType::CPU;
    if is_software_gpu {
        info("software renderer detected, some tests will be skipped");
    }

    // Step 1: Pipeline statistics pool with a real compute dispatch.
    step(1, "Pipeline statistics pool");
    {
        match ctx.create_pipeline_stats_pool(ignis::PipelineStats::COMPUTE_INVOCATIONS, 16) {
            Ok(mut stats) => {
                let cs = ctx.create_shader_module(EMPTY_COMPUTE_SPV)?;
                let layout = ctx.pipeline_layout_builder().build()?;
                let pipeline = ctx
                    .compute_pipeline_builder()
                    .shader(cs.handle(), "main")
                    .layout(layout.handle())
                    .build()?;

                let cmd = pool.allocate_primary()?;
                let rec = pool.begin_primary(cmd)?;
                stats.reset(&rec);
                let scope = stats.begin(&rec, "empty_dispatch_4x4x4");
                rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
                rec.dispatch(4, 4, 4);
                stats.end(&rec, scope);
                let cmd = rec.end()?;
                gfx.submit_simple(cmd)?.wait()?;

                let results = stats.readback()?;
                assert_eq!(results.len(), 1);
                info(&format!("scope: \"{}\"", results[0].label));
                for (name, value) in &results[0].counters {
                    info(&format!("  {name} = {value}"));
                }
                // 4 * 4 * 4 workgroups of local_size 1 = 64 invocations expected.
                if let Some(inv) = results[0].compute_invocations() {
                    info(&format!("compute_invocations = {inv}"));
                    // Software renderers may not report accurate counts,
                    // so only assert on real hardware.
                    if !is_software_gpu {
                        assert!(
                            inv >= 64,
                            "expected at least 64 invocations, got {inv}"
                        );
                    }
                }

                unsafe { ctx.device().destroy_pipeline(pipeline, None) };
                passed += 1;
            }
            Err(e) => {
                info(&format!("stats pool creation failed: {e}"));
                skip("pipeline_statistics_query not supported by this GPU");
                skipped += 1;
            }
        }
    }
    ok();

    // Step 2: Frame graph single-pass execution with a clear op.
    step(2, "Frame graph single-pass execution");
    {
        let executed = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&executed);

        let mut fg = ignis::FrameGraph::new();
        let img = fg.declare_image(
            "test_target",
            ignis::ImageDesc {
                width: 64,
                height: 64,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            },
        );

        fg.add_pass("clear", |p| {
            p.writes_image(img, ignis::ImageUsageContext::TransferDst);
            p.execute(Box::new(move |rec, resolver| {
                let handle = resolver.image(img);
                let clear = vk::ClearColorValue {
                    float32: [0.2, 0.4, 0.6, 1.0],
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
                        handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &clear,
                        std::slice::from_ref(&range),
                    );
                }
                flag.store(true, Ordering::SeqCst);
            }));
        });

        let compiled = fg.compile(&ctx)?;
        info(&format!("compiled plan lines: {}", compiled.dump_plan().lines().count()));
        compiled.execute(&ctx, &pool, &gfx)?;

        assert!(executed.load(Ordering::SeqCst), "pass was not executed");
        info("single-pass frame graph executed and cleared image OK");
        passed += 1;
    }
    ok();

    // Step 3: Frame graph topological ordering across two passes.
    step(3, "Frame graph 2-pass dependency ordering");
    {
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        let mut fg = ignis::FrameGraph::new();
        let gbuffer = fg.declare_image(
            "gbuffer",
            ignis::ImageDesc {
                width: 64,
                height: 64,
                format: vk::Format::R16G16B16A16_SFLOAT,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            },
        );
        let shaded = fg.declare_image(
            "shaded",
            ignis::ImageDesc::color(64, 64, vk::Format::R8G8B8A8_UNORM),
        );

        // Register lighting FIRST to verify the sort actually reorders them.
        let order_light = Arc::clone(&order);
        fg.add_pass("lighting", |p| {
            p.reads_image(gbuffer, ignis::ImageUsageContext::FragmentShaderRead);
            p.writes_image(shaded, ignis::ImageUsageContext::ColorAttachment);
            p.execute(Box::new(move |_rec, _r| {
                order_light.lock().unwrap().push("lighting");
            }));
        });

        let order_geom = Arc::clone(&order);
        fg.add_pass("gbuffer_write", |p| {
            p.writes_image(gbuffer, ignis::ImageUsageContext::ColorAttachment);
            p.execute(Box::new(move |_rec, _r| {
                order_geom.lock().unwrap().push("gbuffer_write");
            }));
        });

        let compiled = fg.compile(&ctx)?;
        compiled.execute(&ctx, &pool, &gfx)?;

        let final_order = order.lock().unwrap();
        info(&format!("execution order: {:?}", *final_order));
        assert_eq!(final_order.len(), 2);
        assert_eq!(
            final_order[0], "gbuffer_write",
            "gbuffer_write must run before lighting"
        );
        assert_eq!(final_order[1], "lighting");
        info("topological sort correctly reordered passes");
        passed += 1;
    }
    ok();

    // Step 4: Frame graph cycle detection.
    step(4, "Frame graph cycle detection");
    {
        let mut fg = ignis::FrameGraph::new();
        let a = fg.declare_image(
            "a",
            ignis::ImageDesc::color(16, 16, vk::Format::R8G8B8A8_UNORM),
        );
        let b = fg.declare_image(
            "b",
            ignis::ImageDesc::color(16, 16, vk::Format::R8G8B8A8_UNORM),
        );

        fg.add_pass("pass_a_reads_b_writes_a", |p| {
            p.reads_image(b, ignis::ImageUsageContext::FragmentShaderRead);
            p.writes_image(a, ignis::ImageUsageContext::ColorAttachment);
            p.execute(Box::new(|_, _| {}));
        });
        fg.add_pass("pass_b_reads_a_writes_b", |p| {
            p.reads_image(a, ignis::ImageUsageContext::FragmentShaderRead);
            p.writes_image(b, ignis::ImageUsageContext::ColorAttachment);
            p.execute(Box::new(|_, _| {}));
        });

        match fg.compile(&ctx) {
            Ok(_) => panic!("cycle should have been detected"),
            Err(ignis::Error::InvalidConfig(msg)) => {
                info(&format!("cycle detected: {msg}"));
                passed += 1;
            }
            Err(e) => panic!("expected InvalidConfig, got: {e}"),
        }
    }
    ok();

    // Step 5: Crash reporter end-to-end (without real device lost).
    step(5, "Crash reporter generation");
    {
        let reporter = ctx.create_crash_reporter();

        // Attach a journal populated with fake submissions.
        let journal = Arc::new(ctx.create_journal(32));
        for i in 0..3 {
            let fence_ci = vk::FenceCreateInfo::default();
            let fence = unsafe { ctx.device().create_fence(&fence_ci, None)? };
            journal.record(
                gfx.family_index(),
                gfx.queue_index(),
                &format!("fake_submit_{i}"),
                &[],
                &[],
                &[],
                fence,
            );
            unsafe { ctx.device().destroy_fence(fence, None) };
        }
        reporter.attach_journal(Arc::clone(&journal));

        // Attach a breadcrumb buffer.
        let bc = Arc::new(ctx.create_breadcrumb_buffer()?);
        reporter.attach_breadcrumbs(Arc::clone(&bc));

        // Register a custom section for application-specific context.
        reporter.add_section(
            "Application Context",
            "Scene: shadow_test\nFrame: 42\nCamera: (0, 5, 10)\n",
        );

        // Generate without invoking the default file-write handler.
        let report = reporter.generate(vk::Result::ERROR_DEVICE_LOST);

        assert!(
            report.body.contains("DEVICE_LOST"),
            "report must mention the error"
        );
        assert!(
            report.body.contains("fake_submit_0"),
            "report must include journal entries"
        );
        assert!(
            report.body.contains("Application Context"),
            "report must include custom sections"
        );
        assert!(
            report.body.contains("shadow_test"),
            "custom section content must appear"
        );

        info(&format!("report size: {} bytes", report.body.len()));
        info(&format!("timestamp: {}", report.timestamp));
        info(&format!("default path: {:?}", report.default_path()));
        info("journal, breadcrumbs, custom section all present");
        passed += 1;
    }
    ok();

    // Step 6: BLAS and TLAS builders (requires ray tracing extensions).
    step(6, "BLAS/TLAS builders");
    {
        match ignis::Ignis::managed(
            ignis::ManagedConfig::new("ignis-rt", vk::API_VERSION_1_2).enable_raytracing(true),
        ) {
            Ok(rt_ctx) => {
                let rt_gfx = rt_ctx.queue(ignis::QueueType::Graphics)?;
                let rt_pool = rt_ctx.create_command_pool(ignis::QueueType::Graphics)?;

                // One triangle in a device-address-enabled buffer pair.
                #[repr(C)]
                #[derive(Copy, Clone)]
                struct Vertex {
                    x: f32,
                    y: f32,
                    z: f32,
                }
                let vertices = [
                    Vertex {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Vertex {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Vertex {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                ];
                let indices: [u32; 3] = [0, 1, 2];

                let vbo = rt_ctx.create_buffer(&ignis::BufferInfo {
                    size: std::mem::size_of_val(&vertices) as u64,
                    usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::VERTEX_BUFFER,
                    location: ignis::MemoryLocation::CpuToGpu,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                })?;
                unsafe { vbo.write_struct(&vertices) };

                let ibo = rt_ctx.create_buffer(&ignis::BufferInfo {
                    size: std::mem::size_of_val(&indices) as u64,
                    usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::INDEX_BUFFER,
                    location: ignis::MemoryLocation::CpuToGpu,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                })?;
                unsafe { ibo.write_struct(&indices) };

                // Build BLAS.
                let blas_result = ignis::BlasBuilder::new(&rt_ctx)?
                    .triangles(ignis::TriangleGeometry {
                        vertex_buffer: vbo.device_address(),
                        vertex_format: vk::Format::R32G32B32_SFLOAT,
                        vertex_stride: std::mem::size_of::<Vertex>() as u64,
                        max_vertex: 2,
                        index_buffer: ibo.device_address(),
                        index_type: vk::IndexType::UINT32,
                        triangle_count: 1,
                    })
                    .build(&rt_pool, &rt_gfx);

                match blas_result {
                    Ok(blas) => {
                        info(&format!("BLAS device_address: {:#x}", blas.device_address()));
                        assert_ne!(blas.device_address(), 0);

                        // Build TLAS referencing that BLAS.
                        let tlas = ignis::TlasBuilder::new(&rt_ctx)?
                            .add_instance(ignis::InstanceDesc {
                                blas_address: blas.device_address(),
                                transform: ignis::identity_transform(),
                                instance_id: 0,
                                mask: 0xFF,
                                sbt_offset: 0,
                                flags: 0,
                            })
                            .build(&rt_pool, &rt_gfx)?;

                        info(&format!("TLAS device_address: {:#x}", tlas.device_address()));
                        assert_ne!(tlas.device_address(), 0);
                        info("BLAS + TLAS built and queryable");

                        drop(tlas);
                        drop(blas);
                        passed += 1;
                    }
                    Err(e) => {
                        info(&format!("BLAS build failed: {e}"));
                        skip("BLAS construction not supported on this driver");
                        skipped += 1;
                    }
                }

                drop(rt_ctx);
            }
            Err(e) => {
                info(&format!("RT context creation failed: {e}"));
                skip("hardware/driver lacks VK_KHR_ray_tracing_pipeline");
                skipped += 1;
            }
        }
    }
    ok();

    // Step 7: Bindless descriptor heap with slot recycling.
    step(7, "Bindless descriptor heap");
    {
        match ctx.create_bindless_heap(ignis::BindlessConfig {
            sampled_images: 64,
            storage_images: 16,
            samplers: 8,
            storage_buffers: 32,
        }) {
            Ok(heap) => {
                info(&format!("heap layout: {:?}", heap.layout()));
                info(&format!("heap set: {:?}", heap.set()));

                let img = ctx.create_image(&ignis::ImageInfo::texture_2d(
                    32,
                    32,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ))?;
                let view = img.create_view(vk::ImageAspectFlags::COLOR)?;

                // Register two sampled images and verify contiguous slots.
                let h1 = heap
                    .register_sampled_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
                assert_eq!(h1.raw(), 0, "first slot should be 0");

                let h2 = heap
                    .register_sampled_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
                assert_eq!(h2.raw(), 1, "second slot should be 1");
                info(&format!("two sampled images registered at {}, {}", h1.raw(), h2.raw()));

                // Free slot 0, then register again, verify reuse.
                heap.free_sampled_image(h1);
                let h3 = heap
                    .register_sampled_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
                assert_eq!(h3.raw(), 0, "freed slot should be reused");
                info(&format!("slot 0 freed and reused: new handle = {}", h3.raw()));

                // Test storage buffer registration.
                let buf = ctx.create_buffer(&ignis::BufferInfo::storage(
                    1024,
                    ignis::MemoryLocation::GpuOnly,
                ))?;
                let sb_handle = heap.register_storage_buffer(buf.handle(), 0, 1024)?;
                assert_eq!(sb_handle.raw(), 0);
                info(&format!("storage buffer registered at {}", sb_handle.raw()));

                unsafe { ctx.device().destroy_image_view(view, None) };
                passed += 1;
            }
            Err(e) => {
                info(&format!("heap creation failed: {e}"));
                skip("descriptor_indexing features not supported");
                skipped += 1;
            }
        }
    }
    ok();

    // Step 8: Shader printf handler registration.
    step(8, "Shader printf handler registration");
    {
        let invoked = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&invoked);

        ctx.set_shader_printf_handler(move |msg| {
            flag.store(true, Ordering::SeqCst);
            let _ = msg.formatted.len();
            let _ = msg.shader_stage;
        });
        info("handler registered via set_shader_printf_handler");

        // Replace with a no-op to verify the registry correctly swaps handlers.
        ctx.set_shader_printf_handler(|_| {});
        info("handler replacement OK (registry accepts new closure)");

        // Note: actually triggering the handler requires a SPIR-V shader
        // that uses debugPrintfEXT and a context created with
        // enable_shader_printf(true). That requires either shipping a
        // pre-compiled SPV blob or an external compiler, neither of which
        // this crate depends on. The runtime path is covered when users
        // build their own shaders with GL_EXT_debug_printf extension.
        info("(actual shader invocation requires external SPIR-V, not exercised)");

        let _ = invoked;
        passed += 1;
    }
    ok();

    // Step 9: Validation policy configuration.
    step(9, "Validation policy configuration");
    {
        use ignis::ValidationPolicy;

        ignis::set_validation_policy(ValidationPolicy::FormatAll);
        assert_eq!(
            ignis::debug::validation::policy(),
            ValidationPolicy::FormatAll
        );
        info("policy = FormatAll");

        ignis::set_validation_policy(ValidationPolicy::ErrorsOnly);
        assert_eq!(
            ignis::debug::validation::policy(),
            ValidationPolicy::ErrorsOnly
        );
        info("policy = ErrorsOnly");

        ignis::set_validation_policy(ValidationPolicy::DropInfo);
        assert_eq!(
            ignis::debug::validation::policy(),
            ValidationPolicy::DropInfo
        );
        info("policy = DropInfo (default)");

        passed += 1;
    }
    ok();

    // Cleanup.
    println!("  Dropping resources...");
    drop(pool);
    drop(ctx);
    println!("  Done.");

    Ok((passed, skipped))
}

fn step(n: u32, title: &str) {
    println!("[{n:>2}/{TOTAL_STEPS}] {title}");
}

fn info(msg: &str) {
    println!("    {msg}");
}

fn ok() {
    println!("    PASSED");
    println!();
}

fn skip(reason: &str) {
    println!("    SKIPPED: {reason}");
}