/*
 * BmpDirectDecode.c - Direct BMP decoding to CHW float32 format
 *
 * This module provides efficient BMP decoding directly to PyTorch tensor memory,
 * minimizing memory bandwidth by fusing:
 * - File I/O (partial read)
 * - BGR -> RGB conversion
 * - HWC -> CHW conversion
 * - uint8 -> float32 conversion
 * - Optional normalization
 *
 * Design goals:
 * - Minimize memory traffic (read once, write once)
 * - Stay within L1/L2 cache as much as possible
 * - Use SIMD when beneficial
 *
 * Copyright (c) 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Memory mapping for zero-copy file access
 * 
 * NOTE: mmap can be slower on network filesystems (Lustre, NFS, CIFS).
 * Set environment variable PILLOW_NO_MMAP=1 to disable mmap and use fread instead.
 */
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define MMAP_AVAILABLE 1
#else
#define MMAP_AVAILABLE 0
#endif

/* Runtime check for mmap - disabled by default for network filesystem compatibility
 * Set PILLOW_USE_MMAP=1 to enable mmap (faster on local filesystems)
 * mmap can be slower on Lustre, NFS, CIFS due to page fault latency */
static int use_mmap_checked = 0;
static int use_mmap_value = 0;  /* Default: OFF (fread) for Lustre/NFS compatibility */

static int should_use_mmap(void) {
#if !MMAP_AVAILABLE
    return 0;
#else
    if (!use_mmap_checked) {
        const char* env = getenv("PILLOW_USE_MMAP");
        if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
            use_mmap_value = 1;  /* Enable mmap only if explicitly requested */
        }
        use_mmap_checked = 1;
    }
    return use_mmap_value;
#endif
}

/* SIMD headers for optimized processing */
#ifdef __SSE2__
#include <emmintrin.h>  /* SSE2 */
#endif
#ifdef __SSSE3__
#include <tmmintrin.h>  /* SSSE3: pshufb */
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>  /* SSE4.1: cvtepu8_epi32 */
#endif
#ifdef __AVX2__
#include <immintrin.h>  /* AVX2 */
#endif

#include "decode_api.h"

/* Error codes */
#define ERR_OPEN_FILE      -1
#define ERR_INVALID_HEADER -2
#define ERR_UNSUPPORTED    -3
#define ERR_INVALID_CROP   -4
#define ERR_UNSUPPORTED_DEPTH -5
#define ERR_MEMORY         -6
#define ERR_READ           -7

/* BMP compression types */
#define BMP_RAW       0
#define BMP_RLE8      1
#define BMP_RLE4      2
#define BMP_BITFIELDS 3

/* Helper macros for reading little-endian values */
#define READ_U16_LE(buf) ((uint16_t)(buf)[0] | ((uint16_t)(buf)[1] << 8))
#define READ_U32_LE(buf) ((uint32_t)(buf)[0] | ((uint32_t)(buf)[1] << 8) | \
                          ((uint32_t)(buf)[2] << 16) | ((uint32_t)(buf)[3] << 24))
#define READ_I32_LE(buf) ((int32_t)READ_U32_LE(buf))

/* BMP file info structure */
typedef struct {
    int width;
    int height;
    int bits;
    int compression;
    int direction;      /* -1 = bottom-up (normal), 1 = top-down */
    int data_offset;
    int stride;         /* bytes per row including padding */
    int channels;       /* derived: 1, 3, or 4 */
    int bytes_per_pixel;
    
    /* For BITFIELDS */
    uint32_t r_mask, g_mask, b_mask, a_mask;
} BmpInfo;

/* Precomputed normalization factor */
static const float INV_255 = 1.0f / 255.0f;

/**
 * Parse BMP header and extract relevant info.
 * Returns 0 on success, negative error code on failure.
 */
static int
parse_bmp_header(FILE* fp, BmpInfo* info) {
    uint8_t header[138];  /* Max header size we need */
    size_t bytes_read;
    uint32_t header_size;
    int32_t height_raw;
    
    /* Read file header (14 bytes) + first 4 bytes of DIB header */
    bytes_read = fread(header, 1, 18, fp);
    if (bytes_read < 18) {
        return ERR_INVALID_HEADER;
    }
    
    /* Check BMP signature */
    if (header[0] != 'B' || header[1] != 'M') {
        return ERR_INVALID_HEADER;
    }
    
    /* Get data offset from file header */
    info->data_offset = READ_U32_LE(header + 10);
    
    /* Get DIB header size */
    header_size = READ_U32_LE(header + 14);
    
    /* Read rest of DIB header */
    if (header_size > 4) {
        size_t remaining = header_size - 4;
        if (remaining > 120) remaining = 120;  /* Cap at our buffer size */
        bytes_read = fread(header + 18, 1, remaining, fp);
        if (bytes_read < remaining) {
            return ERR_INVALID_HEADER;
        }
    }
    
    /* Parse based on header type */
    if (header_size == 12) {
        /* OS/2 v1 header */
        info->width = READ_U16_LE(header + 18);
        info->height = READ_U16_LE(header + 20);
        info->bits = READ_U16_LE(header + 24);
        info->compression = BMP_RAW;
        info->direction = -1;
    } else if (header_size >= 40) {
        /* Windows v3+ header */
        info->width = READ_I32_LE(header + 18);
        height_raw = READ_I32_LE(header + 22);
        info->bits = READ_U16_LE(header + 28);
        info->compression = READ_U32_LE(header + 30);
        
        /* Handle negative height (top-down) */
        if (height_raw < 0) {
            info->height = -height_raw;
            info->direction = 1;
        } else {
            info->height = height_raw;
            info->direction = -1;
        }
        
        /* Parse BITFIELDS masks if needed */
        if (info->compression == BMP_BITFIELDS && header_size >= 56) {
            info->r_mask = READ_U32_LE(header + 54);
            info->g_mask = READ_U32_LE(header + 58);
            info->b_mask = READ_U32_LE(header + 62);
            info->a_mask = (header_size >= 60) ? READ_U32_LE(header + 66) : 0;
        }
    } else {
        return ERR_INVALID_HEADER;
    }
    
    /* Validate dimensions */
    if (info->width <= 0 || info->height <= 0) {
        return ERR_INVALID_HEADER;
    }
    
    /* Calculate derived values */
    info->bytes_per_pixel = (info->bits + 7) / 8;
    info->stride = ((info->width * info->bits + 31) / 32) * 4;
    
    /* Determine channel count */
    switch (info->bits) {
        case 8:
            info->channels = 1;
            break;
        case 24:
            info->channels = 3;
            break;
        case 32:
            info->channels = 4;
            break;
        default:
            return ERR_UNSUPPORTED_DEPTH;
    }
    
    /* Check for unsupported compression */
    if (info->compression == BMP_RLE8 || info->compression == BMP_RLE4) {
        return ERR_UNSUPPORTED;
    }
    
    return 0;
}

#if MMAP_AVAILABLE
/**
 * Parse BMP header from memory-mapped data.
 * This is a zero-copy version that reads directly from mapped memory.
 */
static int
parse_bmp_header_from_memory(const uint8_t* data, size_t data_size, BmpInfo* info) {
    uint32_t header_size;
    int32_t height_raw;
    size_t min_header_size;
    
    /* Need at least file header (14 bytes) + min DIB header (12 bytes) */
    if (data_size < 26) {
        return ERR_INVALID_HEADER;
    }
    
    /* Check BMP signature */
    if (data[0] != 'B' || data[1] != 'M') {
        return ERR_INVALID_HEADER;
    }
    
    /* Get data offset from file header */
    info->data_offset = READ_U32_LE(data + 10);
    
    /* Get DIB header size */
    header_size = READ_U32_LE(data + 14);
    
    /* Validate we have enough data for the header */
    min_header_size = 14 + header_size;
    if (min_header_size > 138) min_header_size = 138;
    if (data_size < min_header_size) {
        return ERR_INVALID_HEADER;
    }
    
    /* Parse based on header type */
    if (header_size == 12) {
        /* OS/2 v1 header */
        info->width = READ_U16_LE(data + 18);
        info->height = READ_U16_LE(data + 20);
        info->bits = READ_U16_LE(data + 24);
        info->compression = BMP_RAW;
        info->direction = -1;
    } else if (header_size >= 40) {
        /* Windows v3+ header */
        info->width = READ_I32_LE(data + 18);
        height_raw = READ_I32_LE(data + 22);
        info->bits = READ_U16_LE(data + 28);
        info->compression = READ_U32_LE(data + 30);
        
        /* Handle negative height (top-down) */
        if (height_raw < 0) {
            info->height = -height_raw;
            info->direction = 1;
        } else {
            info->height = height_raw;
            info->direction = -1;
        }
        
        /* Parse BITFIELDS masks if needed */
        if (info->compression == BMP_BITFIELDS && header_size >= 56) {
            info->r_mask = READ_U32_LE(data + 54);
            info->g_mask = READ_U32_LE(data + 58);
            info->b_mask = READ_U32_LE(data + 62);
            info->a_mask = (header_size >= 60) ? READ_U32_LE(data + 66) : 0;
        }
    } else {
        return ERR_INVALID_HEADER;
    }
    
    /* Validate dimensions */
    if (info->width <= 0 || info->height <= 0) {
        return ERR_INVALID_HEADER;
    }
    
    /* Calculate derived values */
    info->bytes_per_pixel = (info->bits + 7) / 8;
    info->stride = ((info->width * info->bits + 31) / 32) * 4;
    
    /* Determine channel count */
    switch (info->bits) {
        case 8:
            info->channels = 1;
            break;
        case 24:
            info->channels = 3;
            break;
        case 32:
            info->channels = 4;
            break;
        default:
            return ERR_UNSUPPORTED_DEPTH;
    }
    
    /* Check for unsupported compression */
    if (info->compression == BMP_RLE8 || info->compression == BMP_RLE4) {
        return ERR_UNSUPPORTED;
    }
    
    return 0;
}
#endif

/**
 * Process a single row of 24-bit BGR data to CHW float32 RGB.
 * 
 * OPTIMIZATION STRATEGY:
 * 1. Use streaming stores (_mm_stream_ps) to bypass cache on writes
 *    - Avoids "Read-for-Ownership" which wastes 64 bytes per 4-byte write
 *    - Keeps cache available for input data
 * 2. Process 4 pixels at a time with SSE (16 pixels with AVX2)
 * 3. All conversions happen in registers, not memory
 */

/*
 * =============================================================================
 * AVX2 Optimized Version (8 pixels per iteration)
 * =============================================================================
 * 
 * Key optimizations:
 * 1. SIMD load: Single 256-bit load gets ~10 pixels
 * 2. Shuffle deinterleave: pshufb extracts B/G/R in parallel
 * 3. Batch uint8→float: cvtepi32_ps converts 4/8 values at once
 * 4. Streaming stores: Bypasses cache, no Read-for-Ownership penalty
 * 
 * Memory layout for BGR24:
 *   Bytes: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5 G5 R5 B6 G6 R6 B7 G7 R7
 *   Index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23
 */

#ifdef __AVX2__
static void
process_row_bgr24_to_chw_f32_avx2(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    int width,
    float scale
) {
    int x = 0;
    __m256 vscale = _mm256_set1_ps(scale);
    
    /* Shuffle masks for extracting R/G/B from 4 packed BGR pixels
     * Input:  B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 X X X X
     * Output: R0 00 00 00 R1 00 00 00 R2 00 00 00 R3 00 00 00 (as 4 int32s)
     * Note: -1 (0x80) means zero that byte */
    const __m128i r_shuf = _mm_setr_epi8(
        2, -1, -1, -1,   /* R0 → int32[0] */
        5, -1, -1, -1,   /* R1 → int32[1] */
        8, -1, -1, -1,   /* R2 → int32[2] */
        11, -1, -1, -1   /* R3 → int32[3] */
    );
    const __m128i g_shuf = _mm_setr_epi8(
        1, -1, -1, -1,
        4, -1, -1, -1,
        7, -1, -1, -1,
        10, -1, -1, -1
    );
    const __m128i b_shuf = _mm_setr_epi8(
        0, -1, -1, -1,
        3, -1, -1, -1,
        6, -1, -1, -1,
        9, -1, -1, -1
    );
    
    /* Process 8 pixels per iteration (24 bytes input → 96 bytes output) */
    for (; x <= width - 8; x += 8) {
        /* Load pixels 0-3 (bytes 0-15, we use 0-11) */
        __m128i p0 = _mm_loadu_si128((const __m128i*)src);
        /* Load pixels 4-7 (bytes 12-27, we use 12-23) */
        __m128i p1 = _mm_loadu_si128((const __m128i*)(src + 12));
        src += 24;
        
        /* Shuffle to deinterleave BGR → separate R, G, B as int32 arrays
         * This is the KEY optimization: one instruction extracts 4 values */
        __m128i r0_i32 = _mm_shuffle_epi8(p0, r_shuf);
        __m128i g0_i32 = _mm_shuffle_epi8(p0, g_shuf);
        __m128i b0_i32 = _mm_shuffle_epi8(p0, b_shuf);
        
        __m128i r1_i32 = _mm_shuffle_epi8(p1, r_shuf);
        __m128i g1_i32 = _mm_shuffle_epi8(p1, g_shuf);
        __m128i b1_i32 = _mm_shuffle_epi8(p1, b_shuf);
        
        /* Convert int32 → float (batch conversion, very efficient) */
        __m128 r0_f = _mm_cvtepi32_ps(r0_i32);
        __m128 r1_f = _mm_cvtepi32_ps(r1_i32);
        __m128 g0_f = _mm_cvtepi32_ps(g0_i32);
        __m128 g1_f = _mm_cvtepi32_ps(g1_i32);
        __m128 b0_f = _mm_cvtepi32_ps(b0_i32);
        __m128 b1_f = _mm_cvtepi32_ps(b1_i32);
        
        /* Combine two 128-bit results into 256-bit */
        __m256 r8 = _mm256_insertf128_ps(_mm256_castps128_ps256(r0_f), r1_f, 1);
        __m256 g8 = _mm256_insertf128_ps(_mm256_castps128_ps256(g0_f), g1_f, 1);
        __m256 b8 = _mm256_insertf128_ps(_mm256_castps128_ps256(b0_f), b1_f, 1);
        
        /* Scale (normalize to 0-1 if scale = 1/255) */
        r8 = _mm256_mul_ps(r8, vscale);
        g8 = _mm256_mul_ps(g8, vscale);
        b8 = _mm256_mul_ps(b8, vscale);
        
        /* Store - use streaming if aligned, regular store otherwise
         * Note: _mm256_stream_ps requires 32-byte alignment */
        _mm256_storeu_ps(out_r + x, r8);
        _mm256_storeu_ps(out_g + x, g8);
        _mm256_storeu_ps(out_b + x, b8);
    }
    
    /* Handle remaining pixels (scalar) */
    for (; x < width; x++) {
        out_r[x] = (float)src[2] * scale;
        out_g[x] = (float)src[1] * scale;
        out_b[x] = (float)src[0] * scale;
        src += 3;
    }
}
#endif /* __AVX2__ */

/*
 * =============================================================================
 * SSSE3 Optimized Version (4 pixels per iteration)
 * =============================================================================
 * Fallback for CPUs without AVX2 but with SSSE3 (most x86-64 CPUs since 2006)
 */
#ifdef __SSSE3__
static void
process_row_bgr24_to_chw_f32_ssse3(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    int width,
    float scale
) {
    int x = 0;
    __m128 vscale = _mm_set1_ps(scale);
    
    /* Shuffle masks */
    const __m128i r_shuf = _mm_setr_epi8(2,-1,-1,-1, 5,-1,-1,-1, 8,-1,-1,-1, 11,-1,-1,-1);
    const __m128i g_shuf = _mm_setr_epi8(1,-1,-1,-1, 4,-1,-1,-1, 7,-1,-1,-1, 10,-1,-1,-1);
    const __m128i b_shuf = _mm_setr_epi8(0,-1,-1,-1, 3,-1,-1,-1, 6,-1,-1,-1, 9,-1,-1,-1);
    
    /* Process 4 pixels per iteration */
    for (; x <= width - 4; x += 4) {
        /* Load 16 bytes (contains 4 pixels = 12 bytes) */
        __m128i packed = _mm_loadu_si128((const __m128i*)src);
        src += 12;
        
        /* Shuffle to extract as int32 arrays */
        __m128i r_i32 = _mm_shuffle_epi8(packed, r_shuf);
        __m128i g_i32 = _mm_shuffle_epi8(packed, g_shuf);
        __m128i b_i32 = _mm_shuffle_epi8(packed, b_shuf);
        
        /* Convert to float and scale */
        __m128 r4 = _mm_mul_ps(_mm_cvtepi32_ps(r_i32), vscale);
        __m128 g4 = _mm_mul_ps(_mm_cvtepi32_ps(g_i32), vscale);
        __m128 b4 = _mm_mul_ps(_mm_cvtepi32_ps(b_i32), vscale);
        
        /* Store (use storeu for unaligned safety) */
        _mm_storeu_ps(out_r + x, r4);
        _mm_storeu_ps(out_g + x, g4);
        _mm_storeu_ps(out_b + x, b4);
    }
    
    /* Handle remaining pixels */
    for (; x < width; x++) {
        out_r[x] = (float)src[2] * scale;
        out_g[x] = (float)src[1] * scale;
        out_b[x] = (float)src[0] * scale;
        src += 3;
    }
}
#endif /* __SSSE3__ */

/*
 * =============================================================================
 * SSE2 Fallback Version
 * =============================================================================
 */
#ifdef __SSE2__
static void
process_row_bgr24_to_chw_f32_sse2(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    int width,
    float scale
) {
    int x = 0;
    __m128 vscale = _mm_set1_ps(scale);
    
    /* Process 4 pixels at a time - scalar loads, SIMD processing */
    for (; x <= width - 4; x += 4) {
        /* Scalar loads (no pshufb in SSE2) */
        __m128 r4 = _mm_set_ps((float)src[11], (float)src[8], (float)src[5], (float)src[2]);
        __m128 g4 = _mm_set_ps((float)src[10], (float)src[7], (float)src[4], (float)src[1]);
        __m128 b4 = _mm_set_ps((float)src[9],  (float)src[6], (float)src[3], (float)src[0]);
        src += 12;
        
        r4 = _mm_mul_ps(r4, vscale);
        g4 = _mm_mul_ps(g4, vscale);
        b4 = _mm_mul_ps(b4, vscale);
        
        _mm_storeu_ps(out_r + x, r4);
        _mm_storeu_ps(out_g + x, g4);
        _mm_storeu_ps(out_b + x, b4);
    }
    
    for (; x < width; x++) {
        out_r[x] = (float)src[2] * scale;
        out_g[x] = (float)src[1] * scale;
        out_b[x] = (float)src[0] * scale;
        src += 3;
    }
}
#endif /* __SSE2__ */

/*
 * =============================================================================
 * Dispatcher - automatically selects best implementation
 * =============================================================================
 * Priority: AVX2 > SSSE3 > SSE2 > Scalar
 */
static void
process_row_bgr24_to_chw_f32(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    int width,
    int64_t stride_x,
    float scale
) {
    /* For contiguous output (stride_x == 1), use SIMD optimized path */
    if (stride_x == 1) {
#ifdef __AVX2__
        process_row_bgr24_to_chw_f32_avx2(src, out_r, out_g, out_b, width, scale);
        return;
#elif defined(__SSSE3__)
        process_row_bgr24_to_chw_f32_ssse3(src, out_r, out_g, out_b, width, scale);
        return;
#elif defined(__SSE2__)
        process_row_bgr24_to_chw_f32_sse2(src, out_r, out_g, out_b, width, scale);
        return;
#endif
    }
    
    /* Fallback for non-contiguous stride or non-SIMD systems */
    {
        int x;
        for (x = 0; x < width; x++) {
            out_r[x * stride_x] = (float)src[2] * scale;
            out_g[x * stride_x] = (float)src[1] * scale;
            out_b[x * stride_x] = (float)src[0] * scale;
            src += 3;
        }
    }
}

/*
 * =============================================================================
 * 32-bit BGRX/BGRA Processing
 * =============================================================================
 * Easier than BGR24 because 4 bytes aligns perfectly with SIMD
 */

#ifdef __AVX2__
static void
process_row_bgrx32_to_chw_f32_avx2(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    float* __restrict out_a,
    int width,
    float scale,
    int write_alpha
) {
    int x = 0;
    __m256 vscale = _mm256_set1_ps(scale);
    
    /* Shuffle masks for BGRX (4 bytes per pixel, 4 pixels per 128-bit) */
    const __m128i r_shuf = _mm_setr_epi8(2,-1,-1,-1, 6,-1,-1,-1, 10,-1,-1,-1, 14,-1,-1,-1);
    const __m128i g_shuf = _mm_setr_epi8(1,-1,-1,-1, 5,-1,-1,-1, 9,-1,-1,-1, 13,-1,-1,-1);
    const __m128i b_shuf = _mm_setr_epi8(0,-1,-1,-1, 4,-1,-1,-1, 8,-1,-1,-1, 12,-1,-1,-1);
    const __m128i a_shuf = _mm_setr_epi8(3,-1,-1,-1, 7,-1,-1,-1, 11,-1,-1,-1, 15,-1,-1,-1);
    
    /* Process 8 pixels per iteration (32 bytes input) */
    for (; x <= width - 8; x += 8) {
        /* Load 8 pixels = 32 bytes */
        __m128i p0 = _mm_loadu_si128((const __m128i*)src);        /* pixels 0-3 */
        __m128i p1 = _mm_loadu_si128((const __m128i*)(src + 16)); /* pixels 4-7 */
        src += 32;
        
        /* Shuffle and convert */
        __m128 r0 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p0, r_shuf));
        __m128 r1 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p1, r_shuf));
        __m128 g0 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p0, g_shuf));
        __m128 g1 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p1, g_shuf));
        __m128 b0 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p0, b_shuf));
        __m128 b1 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p1, b_shuf));
        
        /* Combine and scale */
        __m256 r8 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(r0), r1, 1), vscale);
        __m256 g8 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(g0), g1, 1), vscale);
        __m256 b8 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(b0), b1, 1), vscale);
        
        _mm256_storeu_ps(out_r + x, r8);
        _mm256_storeu_ps(out_g + x, g8);
        _mm256_storeu_ps(out_b + x, b8);
        
        if (write_alpha && out_a) {
            __m128 a0 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p0, a_shuf));
            __m128 a1 = _mm_cvtepi32_ps(_mm_shuffle_epi8(p1, a_shuf));
            __m256 a8 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(a0), a1, 1), vscale);
            _mm256_storeu_ps(out_a + x, a8);
        }
    }
    
    /* Remaining pixels */
    for (; x < width; x++) {
        out_r[x] = (float)src[2] * scale;
        out_g[x] = (float)src[1] * scale;
        out_b[x] = (float)src[0] * scale;
        if (write_alpha && out_a) out_a[x] = (float)src[3] * scale;
        src += 4;
    }
}
#endif /* __AVX2__ */

#ifdef __SSSE3__
static void
process_row_bgrx32_to_chw_f32_ssse3(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    float* __restrict out_a,
    int width,
    float scale,
    int write_alpha
) {
    int x = 0;
    __m128 vscale = _mm_set1_ps(scale);
    
    const __m128i r_shuf = _mm_setr_epi8(2,-1,-1,-1, 6,-1,-1,-1, 10,-1,-1,-1, 14,-1,-1,-1);
    const __m128i g_shuf = _mm_setr_epi8(1,-1,-1,-1, 5,-1,-1,-1, 9,-1,-1,-1, 13,-1,-1,-1);
    const __m128i b_shuf = _mm_setr_epi8(0,-1,-1,-1, 4,-1,-1,-1, 8,-1,-1,-1, 12,-1,-1,-1);
    const __m128i a_shuf = _mm_setr_epi8(3,-1,-1,-1, 7,-1,-1,-1, 11,-1,-1,-1, 15,-1,-1,-1);
    
    for (; x <= width - 4; x += 4) {
        __m128i packed = _mm_loadu_si128((const __m128i*)src);
        src += 16;
        
        __m128 r4 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_shuffle_epi8(packed, r_shuf)), vscale);
        __m128 g4 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_shuffle_epi8(packed, g_shuf)), vscale);
        __m128 b4 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_shuffle_epi8(packed, b_shuf)), vscale);
        
        _mm_storeu_ps(out_r + x, r4);
        _mm_storeu_ps(out_g + x, g4);
        _mm_storeu_ps(out_b + x, b4);
        
        if (write_alpha && out_a) {
            __m128 a4 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_shuffle_epi8(packed, a_shuf)), vscale);
            _mm_storeu_ps(out_a + x, a4);
        }
    }
    
    for (; x < width; x++) {
        out_r[x] = (float)src[2] * scale;
        out_g[x] = (float)src[1] * scale;
        out_b[x] = (float)src[0] * scale;
        if (write_alpha && out_a) out_a[x] = (float)src[3] * scale;
        src += 4;
    }
}
#endif /* __SSSE3__ */

/* Dispatcher for 32-bit */
static void
process_row_bgrx32_to_chw_f32(
    const uint8_t* __restrict src,
    float* __restrict out_r,
    float* __restrict out_g,
    float* __restrict out_b,
    float* __restrict out_a,
    int width,
    int64_t stride_x,
    float scale,
    int write_alpha
) {
    if (stride_x == 1) {
#ifdef __AVX2__
        process_row_bgrx32_to_chw_f32_avx2(src, out_r, out_g, out_b, out_a, width, scale, write_alpha);
        return;
#elif defined(__SSSE3__)
        process_row_bgrx32_to_chw_f32_ssse3(src, out_r, out_g, out_b, out_a, width, scale, write_alpha);
        return;
#endif
    }
    
    /* Scalar fallback */
    {
        int x;
        for (x = 0; x < width; x++) {
            out_r[x * stride_x] = (float)src[2] * scale;
            out_g[x * stride_x] = (float)src[1] * scale;
            out_b[x * stride_x] = (float)src[0] * scale;
            if (write_alpha && out_a) {
                out_a[x * stride_x] = (float)src[3] * scale;
            }
            src += 4;
        }
    }
}

/**
 * Process a single row of 8-bit grayscale data to CHW float32.
 */
static void
process_row_gray8_to_chw_f32(
    const uint8_t* __restrict src,
    float* __restrict out,
    int width,
    int64_t stride_x,
    float scale
) {
    int x;
    
    for (x = 0; x < width; x++) {
        out[x * stride_x] = (float)src[x] * scale;
    }
}

/**
 * Process all rows from a buffer - optimized version for bulk processing.
 * Handles both top-down and bottom-up BMPs by adjusting the read order.
 */
static void
process_all_rows(
    const uint8_t* data,
    int stride,
    int bits,
    int bytes_per_pixel,
    int x0,
    int crop_width,
    int crop_height,
    int direction,  /* 1 = top-down (data[0] is first output row), -1 = bottom-up (data[0] is last output row) */
    int out_channels,
    float* out,
    int64_t stride_c,
    int64_t stride_y,
    int64_t stride_x,
    float scale,
    int drop_alpha
) {
    int out_row;
    
    for (out_row = 0; out_row < crop_height; out_row++) {
        const uint8_t* src;
        float* dst_base;
        int data_row;
        
        /* For bottom-up BMP, the first row in data buffer is the last output row */
        if (direction == -1) {
            data_row = crop_height - 1 - out_row;
        } else {
            data_row = out_row;
        }
        
        /* Point to start of crop region within the row */
        src = data + data_row * stride + x0 * bytes_per_pixel;
        
        /* Calculate output position */
        dst_base = out + out_row * stride_y;
        
        /* Process based on bit depth */
        switch (bits) {
            case 24:
                if (out_channels >= 3) {
                    process_row_bgr24_to_chw_f32(
                        src,
                        dst_base,
                        dst_base + stride_c,
                        dst_base + 2 * stride_c,
                        crop_width,
                        stride_x,
                        scale
                    );
                } else if (out_channels == 1) {
                    int x;
                    for (x = 0; x < crop_width; x++) {
                        float r = (float)src[x*3 + 2] * scale;
                        float g = (float)src[x*3 + 1] * scale;
                        float b = (float)src[x*3 + 0] * scale;
                        dst_base[x * stride_x] = (r + g + b) / 3.0f;
                    }
                }
                break;
                
            case 32:
                {
                    int write_alpha = (out_channels == 4 && !drop_alpha);
                    float* out_a = write_alpha ? (dst_base + 3 * stride_c) : NULL;
                    
                    if (out_channels >= 3) {
                        process_row_bgrx32_to_chw_f32(
                            src,
                            dst_base,
                            dst_base + stride_c,
                            dst_base + 2 * stride_c,
                            out_a,
                            crop_width,
                            stride_x,
                            scale,
                            write_alpha
                        );
                    } else if (out_channels == 1) {
                        int x;
                        for (x = 0; x < crop_width; x++) {
                            float r = (float)src[x*4 + 2] * scale;
                            float g = (float)src[x*4 + 1] * scale;
                            float b = (float)src[x*4 + 0] * scale;
                            dst_base[x * stride_x] = (r + g + b) / 3.0f;
                        }
                    }
                }
                break;
                
            case 8:
                if (out_channels >= 1) {
                    process_row_gray8_to_chw_f32(
                        src,
                        dst_base,
                        crop_width,
                        stride_x,
                        scale
                    );
                    
                    if (out_channels >= 3) {
                        int x;
                        for (x = 0; x < crop_width; x++) {
                            float val = dst_base[x * stride_x];
                            dst_base[stride_c + x * stride_x] = val;
                            dst_base[2 * stride_c + x * stride_x] = val;
                        }
                    }
                }
                break;
        }
    }
    
    /* Memory fence to ensure all streaming stores are visible
     * This is CRITICAL when using _mm_stream_ps */
#ifdef __SSE2__
    _mm_sfence();
#endif
}

/**
 * Main decode function.
 * 
 * Optimization Strategy:
 * 1. Use mmap() instead of malloc+fread to eliminate intermediate buffer copy
 *    - Data goes directly: File → Page Cache → CPU → Tensor
 *    - No extra memory allocation for bulk_buffer
 *    - Kernel handles prefetching automatically
 * 
 * 2. For platforms without mmap (Windows), fall back to fread
 * 
 * Memory flow comparison:
 *   Old: File → PageCache → bulk_buffer(write) → bulk_buffer(read) → Tensor
 *   New: File → PageCache → mmap(read only) → Tensor
 *   Savings: Eliminates ~3MB write + read of intermediate buffer
 */
int
decode_crop_to_chw_f32(
    const char* filename,
    int x0, int y0, int x1, int y1,
    int out_channels,
    float* out,
    int64_t stride_c,
    int64_t stride_y,
    int64_t stride_x,
    int normalize_01,
    int drop_alpha
) {
    BmpInfo info;
    int result;
    int crop_width, crop_height;
    int file_start_row;
    size_t data_offset;
    float scale;
    
    /* Variables for both paths */
    FILE* fp = NULL;
    uint8_t* bulk_buffer = NULL;
    long file_offset;
    size_t bytes_to_read;
    
#if MMAP_AVAILABLE
    /* Variables for mmap path */
    int fd = -1;
    struct stat st;
    uint8_t* mapped = NULL;
    size_t mapped_size = 0;
    const uint8_t* data_ptr;
    
    /* ================================================================
     * RUNTIME CHOICE: mmap vs fread
     * 
     * mmap is faster on local filesystems but can be slower on network
     * filesystems (Lustre, NFS, CIFS). Set PILLOW_NO_MMAP=1 to disable.
     * ================================================================ */
    if (should_use_mmap()) {
        /* MMAP path */
        fd = open(filename, O_RDONLY);
        if (fd < 0) {
            return ERR_OPEN_FILE;
        }
        
        if (fstat(fd, &st) < 0) {
            close(fd);
            return ERR_OPEN_FILE;
        }
        mapped_size = st.st_size;
        
        mapped = (uint8_t*)mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            close(fd);
            return ERR_MEMORY;
        }
        
        madvise(mapped, mapped_size, MADV_SEQUENTIAL);
        close(fd);
        
        result = parse_bmp_header_from_memory(mapped, mapped_size, &info);
        if (result < 0) {
            munmap(mapped, mapped_size);
            return result;
        }
        
        if (x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0) {
            munmap(mapped, mapped_size);
            return ERR_INVALID_CROP;
        }
        if (x1 > info.width || y1 > info.height) {
            munmap(mapped, mapped_size);
            return ERR_INVALID_CROP;
        }
        
        crop_width = x1 - x0;
        crop_height = y1 - y0;
        
        if (out_channels < 1 || out_channels > 4) {
            munmap(mapped, mapped_size);
            return ERR_INVALID_CROP;
        }
        
        scale = normalize_01 ? INV_255 : 1.0f;
        
        if (info.direction == -1) {
            file_start_row = info.height - y1;
        } else {
            file_start_row = y0;
        }
        data_offset = info.data_offset + (size_t)file_start_row * info.stride;
        
        if (data_offset + (size_t)crop_height * info.stride > mapped_size) {
            munmap(mapped, mapped_size);
            return ERR_READ;
        }
        
        data_ptr = mapped + data_offset;
        
        process_all_rows(
            data_ptr,
            info.stride,
            info.bits,
            info.bytes_per_pixel,
            x0,
            crop_width,
            crop_height,
            info.direction,
            out_channels,
            out,
            stride_c,
            stride_y,
            stride_x,
            scale,
            drop_alpha
        );
        
        munmap(mapped, mapped_size);
        return 0;
    }
#endif
    
    /* ================================================================
     * FREAD path (used when mmap disabled or on Windows)
     * Better for network filesystems like Lustre, NFS, CIFS
     * ================================================================ */
    fp = fopen(filename, "rb");
    if (!fp) {
        return ERR_OPEN_FILE;
    }
    
    result = parse_bmp_header(fp, &info);
    if (result < 0) {
        fclose(fp);
        return result;
    }
    
    if (x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0) {
        fclose(fp);
        return ERR_INVALID_CROP;
    }
    if (x1 > info.width || y1 > info.height) {
        fclose(fp);
        return ERR_INVALID_CROP;
    }
    
    crop_width = x1 - x0;
    crop_height = y1 - y0;
    
    if (out_channels < 1 || out_channels > 4) {
        fclose(fp);
        return ERR_INVALID_CROP;
    }
    
    bytes_to_read = (size_t)crop_height * info.stride;
    bulk_buffer = (uint8_t*)malloc(bytes_to_read);
    if (!bulk_buffer) {
        fclose(fp);
        return ERR_MEMORY;
    }
    
    scale = normalize_01 ? INV_255 : 1.0f;
    
    if (info.direction == -1) {
        file_start_row = info.height - y1;
    } else {
        file_start_row = y0;
    }
    
    file_offset = info.data_offset + (long)file_start_row * info.stride;
    if (fseek(fp, file_offset, SEEK_SET) != 0) {
        free(bulk_buffer);
        fclose(fp);
        return ERR_READ;
    }
    
    if (fread(bulk_buffer, 1, bytes_to_read, fp) != bytes_to_read) {
        free(bulk_buffer);
        fclose(fp);
        return ERR_READ;
    }
    
    fclose(fp);
    
    process_all_rows(
        bulk_buffer,
        info.stride,
        info.bits,
        info.bytes_per_pixel,
        x0,
        crop_width,
        crop_height,
        info.direction,
        out_channels,
        out,
        stride_c,
        stride_y,
        stride_x,
        scale,
        drop_alpha
    );
    
    free(bulk_buffer);
    return 0;
}

/**
 * Get BMP image info without fully decoding.
 */
int
bmp_get_info(
    const char* filename,
    int* width,
    int* height,
    int* channels
) {
    FILE* fp;
    BmpInfo info;
    int result;
    
    fp = fopen(filename, "rb");
    if (!fp) {
        return ERR_OPEN_FILE;
    }
    
    result = parse_bmp_header(fp, &info);
    fclose(fp);
    
    if (result < 0) {
        return result;
    }
    
    if (width) *width = info.width;
    if (height) *height = info.height;
    if (channels) *channels = info.channels;
    
    return 0;
}

