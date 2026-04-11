//! Mipmap generation utility via blit chain.
//!
//! Records a series of `vkCmdBlitImage` commands with proper layout
//! transitions for each mip level. Uses the [`ResourceTracker`] for
//! automatic barrier computation.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ash::vk;
//! # fn example(rec: &CommandRecorder, tracker: &mut ResourceTracker,
//! #            image: vk::Image) {
//! ignis::tracking::mipmap::generate_mipmaps(
//!     rec, tracker, image,
//!     vk::Format::R8G8B8A8_UNORM,
//!     64, 64, 7, // width, height, mip_levels
//!     vk::Filter::LINEAR,
//! );
//! # }
//! ```

use ash::vk;

use crate::command::CommandRecorder;
use super::tracker::{ImageUsageContext, ResourceTracker};

/// Record mipmap generation commands via blit chain.
///
/// The image must already be tracked and its mip 0 must contain the
/// source data in `TRANSFER_SRC_OPTIMAL` layout. All other mip levels
/// should be in `UNDEFINED` or `TRANSFER_DST_OPTIMAL`.
///
/// After this function, all mip levels will be in `TRANSFER_SRC_OPTIMAL`.
/// Transition to the desired final layout afterwards.
pub fn generate_mipmaps(
    rec: &CommandRecorder<'_>,
    tracker: &mut ResourceTracker,
    image: vk::Image,
    _format: vk::Format,
    mut width: u32,
    mut height: u32,
    mip_levels: u32,
    filter: vk::Filter,
) {
    // Mip 0 should already be TRANSFER_SRC.
    for mip in 1..mip_levels {
        // Transition mip `mip` to TRANSFER_DST.
        if let Some(t) = tracker.transition_mip(image, mip, ImageUsageContext::TransferDst) {
            rec.apply_image_transitions(&[t]);
        }

        let src_w = width;
        let src_h = height;
        let dst_w = (width / 2).max(1);
        let dst_h = (height / 2).max(1);

        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: mip - 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src_w as i32,
                    y: src_h as i32,
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: mip,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst_w as i32,
                    y: dst_h as i32,
                    z: 1,
                },
            ],
        };

        unsafe {
            rec.device.cmd_blit_image(
                rec.buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                filter,
            );
        }

        // Transition mip `mip` to TRANSFER_SRC for next iteration.
        if let Some(t) = tracker.transition_mip(image, mip, ImageUsageContext::TransferSrc) {
            rec.apply_image_transitions(&[t]);
        }

        width = dst_w;
        height = dst_h;
    }
}