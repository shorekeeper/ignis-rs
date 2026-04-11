//! Vulkan format utilities and compute dispatch helpers.
//!
//! Provides pure-Rust utility functions for querying format properties
//! without Vulkan API calls. These are compile-time-constant lookups
//! covering the most common Vulkan formats.
//!
//! # Dispatch Helpers
//!
//! The [`dispatch_size`] function eliminates the ubiquitous
//! `(count + local_size - 1) / local_size` pattern.
//!
//! # Example
//!
//! ```rust
//! use ignis::format;
//! use ash::vk;
//!
//! let bpp = format::format_byte_size(vk::Format::R8G8B8A8_UNORM);
//! assert_eq!(bpp, Some(4));
//!
//! let groups = format::dispatch_size(1000, 64);
//! assert_eq!(groups, 16); // ceil(1000/64)
//! ```

use ash::vk;

/// Returns the byte size per pixel (or per texel block for compressed formats).
///
/// Returns `None` for formats not in the lookup table.
pub fn format_byte_size(format: vk::Format) -> Option<u32> {
    match format {
        // 1-byte formats.
        vk::Format::R8_UNORM
        | vk::Format::R8_SNORM
        | vk::Format::R8_UINT
        | vk::Format::R8_SINT
        | vk::Format::R8_SRGB => Some(1),

        // 2-byte formats.
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT
        | vk::Format::D16_UNORM
        | vk::Format::R5G6B5_UNORM_PACK16
        | vk::Format::A1R5G5B5_UNORM_PACK16 => Some(2),

        // 3-byte formats.
        vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SNORM
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8_SINT
        | vk::Format::R8G8B8_SRGB
        | vk::Format::B8G8R8_UNORM
        | vk::Format::B8G8R8_SRGB => Some(3),

        // 4-byte formats.
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A8B8G8R8_SRGB_PACK32
        | vk::Format::A2R10G10B10_UNORM_PACK32
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R32_SFLOAT
        | vk::Format::B10G11R11_UFLOAT_PACK32
        | vk::Format::E5B9G9R9_UFLOAT_PACK32
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::X8_D24_UNORM_PACK32 => Some(4),

        // 5-byte formats.
        vk::Format::D32_SFLOAT_S8_UINT => Some(5),

        // 6-byte formats.
        vk::Format::R16G16B16_SFLOAT | vk::Format::R16G16B16_UINT => Some(6),

        // 8-byte formats.
        vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R64_SFLOAT => Some(8),

        // 12-byte formats.
        vk::Format::R32G32B32_UINT
        | vk::Format::R32G32B32_SINT
        | vk::Format::R32G32B32_SFLOAT => Some(12),

        // 16-byte formats.
        vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R64G64_SFLOAT => Some(16),

        // Compressed: bytes per 4×4 block.
        vk::Format::BC1_RGB_UNORM_BLOCK
        | vk::Format::BC1_RGB_SRGB_BLOCK
        | vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK
        | vk::Format::ETC2_R8G8B8_UNORM_BLOCK
        | vk::Format::ETC2_R8G8B8_SRGB_BLOCK => Some(8),

        vk::Format::BC2_UNORM_BLOCK
        | vk::Format::BC2_SRGB_BLOCK
        | vk::Format::BC3_UNORM_BLOCK
        | vk::Format::BC3_SRGB_BLOCK
        | vk::Format::BC5_UNORM_BLOCK
        | vk::Format::BC5_SNORM_BLOCK
        | vk::Format::BC6H_UFLOAT_BLOCK
        | vk::Format::BC6H_SFLOAT_BLOCK
        | vk::Format::BC7_UNORM_BLOCK
        | vk::Format::BC7_SRGB_BLOCK
        | vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK
        | vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK
        | vk::Format::ASTC_4X4_UNORM_BLOCK
        | vk::Format::ASTC_4X4_SRGB_BLOCK => Some(16),

        _ => None,
    }
}

/// Returns the appropriate `ImageAspectFlags` for the given format.
///
/// - Depth formats → `DEPTH`
/// - Depth+stencil formats → `DEPTH | STENCIL`
/// - Stencil-only formats → `STENCIL`
/// - Everything else → `COLOR`
pub fn format_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// Returns `true` if the format has a depth component.
pub fn is_depth_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM
            | vk::Format::D32_SFLOAT
            | vk::Format::X8_D24_UNORM_PACK32
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// Returns `true` if the format has a stencil component.
pub fn is_stencil_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// Returns `true` if the format is a block-compressed format (BC, ETC, ASTC).
pub fn is_compressed_format(format: vk::Format) -> bool {
    let raw = format.as_raw();
    // BC1..BC7: 131..145, ETC2: 147..156, ASTC: 157..184
    (131..=184).contains(&raw)
}

/// Returns the texel block dimensions for compressed formats.
///
/// For uncompressed formats returns `(1, 1)`.
pub fn format_block_extent(format: vk::Format) -> (u32, u32) {
    match format {
        f if !is_compressed_format(f) => (1, 1),
        // BC1-BC7, ETC2 are all 4×4 blocks.
        vk::Format::ASTC_5X4_UNORM_BLOCK | vk::Format::ASTC_5X4_SRGB_BLOCK => (5, 4),
        vk::Format::ASTC_5X5_UNORM_BLOCK | vk::Format::ASTC_5X5_SRGB_BLOCK => (5, 5),
        vk::Format::ASTC_6X5_UNORM_BLOCK | vk::Format::ASTC_6X5_SRGB_BLOCK => (6, 5),
        vk::Format::ASTC_6X6_UNORM_BLOCK | vk::Format::ASTC_6X6_SRGB_BLOCK => (6, 6),
        vk::Format::ASTC_8X5_UNORM_BLOCK | vk::Format::ASTC_8X5_SRGB_BLOCK => (8, 5),
        vk::Format::ASTC_8X6_UNORM_BLOCK | vk::Format::ASTC_8X6_SRGB_BLOCK => (8, 6),
        vk::Format::ASTC_8X8_UNORM_BLOCK | vk::Format::ASTC_8X8_SRGB_BLOCK => (8, 8),
        vk::Format::ASTC_10X5_UNORM_BLOCK | vk::Format::ASTC_10X5_SRGB_BLOCK => (10, 5),
        vk::Format::ASTC_10X6_UNORM_BLOCK | vk::Format::ASTC_10X6_SRGB_BLOCK => (10, 6),
        vk::Format::ASTC_10X8_UNORM_BLOCK | vk::Format::ASTC_10X8_SRGB_BLOCK => (10, 8),
        vk::Format::ASTC_10X10_UNORM_BLOCK | vk::Format::ASTC_10X10_SRGB_BLOCK => (10, 10),
        vk::Format::ASTC_12X10_UNORM_BLOCK | vk::Format::ASTC_12X10_SRGB_BLOCK => (12, 10),
        vk::Format::ASTC_12X12_UNORM_BLOCK | vk::Format::ASTC_12X12_SRGB_BLOCK => (12, 12),
        _ => (4, 4), // BC1-BC7, ETC2 default
    }
}

/// Compute dispatch group count: `ceil(work_items / local_size)`.
///
/// Eliminates the `(count + local_size - 1) / local_size` pattern.
///
/// # Panics
///
/// Panics if `local_size` is zero.
///
/// # Example
///
/// ```rust
/// use ignis::format::dispatch_size;
/// assert_eq!(dispatch_size(1000, 64), 16);
/// assert_eq!(dispatch_size(64, 64), 1);
/// assert_eq!(dispatch_size(0, 64), 0);
/// ```
#[inline]
pub fn dispatch_size(work_items: u32, local_size: u32) -> u32 {
    assert_ne!(local_size, 0, "local_size must be non-zero");
    (work_items + local_size - 1) / local_size
}

/// Compute 3D dispatch group counts.
///
/// # Panics
///
/// Panics if any `local_size` component is zero.
#[inline]
pub fn dispatch_size_3d(work: [u32; 3], local: [u32; 3]) -> [u32; 3] {
    [
        dispatch_size(work[0], local[0]),
        dispatch_size(work[1], local[1]),
        dispatch_size(work[2], local[2]),
    ]
}

/// Compute the number of mip levels for a 2D image of the given size.
///
/// Returns `floor(log2(max(width, height))) + 1`.
#[inline]
pub fn mip_levels_for_size(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim == 0 {
        return 1;
    }
    u32::BITS - max_dim.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_byte_size() {
        assert_eq!(format_byte_size(vk::Format::R8G8B8A8_UNORM), Some(4));
        assert_eq!(format_byte_size(vk::Format::R32G32B32A32_SFLOAT), Some(16));
        assert_eq!(format_byte_size(vk::Format::D32_SFLOAT), Some(4));
        assert_eq!(format_byte_size(vk::Format::R16_SFLOAT), Some(2));
    }

    #[test]
    fn test_dispatch_size() {
        assert_eq!(dispatch_size(1000, 64), 16);
        assert_eq!(dispatch_size(64, 64), 1);
        assert_eq!(dispatch_size(65, 64), 2);
        assert_eq!(dispatch_size(0, 64), 0);
    }

    #[test]
    fn test_mip_levels() {
        assert_eq!(mip_levels_for_size(256, 256), 9);
        assert_eq!(mip_levels_for_size(1, 1), 1);
        assert_eq!(mip_levels_for_size(1024, 512), 11);
    }

    #[test]
    fn test_aspect_mask() {
        assert_eq!(
            format_aspect_mask(vk::Format::R8G8B8A8_UNORM),
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            format_aspect_mask(vk::Format::D32_SFLOAT),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            format_aspect_mask(vk::Format::D24_UNORM_S8_UINT),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
    }
}