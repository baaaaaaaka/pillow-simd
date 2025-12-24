/* decode_api.h - Direct decode API for PyTorch integration
 *
 * This API allows decoding BMP images directly into user-provided memory,
 * avoiding intermediate copies and enabling efficient PyTorch tensor population.
 *
 * Copyright (c) 2024
 */

#ifndef DECODE_API_H
#define DECODE_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decode a cropped region of a BMP file directly to CHW float32 format.
 *
 * This function performs:
 * - BMP header parsing
 * - Partial file reading (only needed rows)
 * - BGR -> RGB channel reordering
 * - HWC -> CHW layout conversion
 * - uint8 -> float32 conversion
 * - Optional normalization (divide by 255)
 * - Optional alpha channel dropping
 *
 * All operations are fused to minimize memory bandwidth usage.
 *
 * @param filename      Path to the BMP file
 * @param x0, y0        Top-left corner of crop region (inclusive)
 * @param x1, y1        Bottom-right corner of crop region (exclusive)
 * @param out_channels  Number of output channels (1, 3, or 4)
 * @param out           Pointer to output buffer (CHW float32)
 * @param stride_c      Stride between channels (in float elements)
 * @param stride_y      Stride between rows (in float elements)
 * @param stride_x      Stride between columns (in float elements)
 * @param normalize_01  If 1, multiply by (1/255.0); if 0, no normalization
 * @param drop_alpha    If 1, ignore alpha channel even if present; if 0, allow alpha
 *
 * @return 0 on success, negative error code on failure:
 *         -1: Failed to open file
 *         -2: Invalid BMP header
 *         -3: Unsupported BMP format (RLE compression, etc.)
 *         -4: Invalid crop coordinates
 *         -5: Unsupported bit depth
 *         -6: Memory allocation failed
 *         -7: File read error
 */
int decode_crop_to_chw_f32(
    const char* filename,
    int x0, int y0, int x1, int y1,      /* crop box: [x0, x1), [y0, y1) */
    int out_channels,                    /* 1, 3, or 4 */
    float* out,
    int64_t stride_c,
    int64_t stride_y,
    int64_t stride_x,
    int normalize_01,                    /* 1: write *= (1/255), 0: no normalize */
    int drop_alpha                       /* 1: input has A but only write RGB; 0: allow A */
);

/**
 * Get image dimensions without fully decoding.
 *
 * @param filename  Path to the BMP file
 * @param width     Output: image width
 * @param height    Output: image height
 * @param channels  Output: number of channels (1, 3, or 4)
 *
 * @return 0 on success, negative error code on failure
 */
int bmp_get_info(
    const char* filename,
    int* width,
    int* height,
    int* channels
);

#ifdef __cplusplus
}
#endif

#endif /* DECODE_API_H */

