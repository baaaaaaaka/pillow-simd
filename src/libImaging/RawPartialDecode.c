/*
 * The Python Imaging Library.
 *
 * Decoder for raw (uncompressed) image data with partial/region support.
 * This decoder supports skipping columns for horizontal cropping optimization.
 *
 * Copyright (c) 2024
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"
#include "RawPartial.h"

int
ImagingRawPartialDecode(Imaging im, ImagingCodecState state, UINT8 *buf, Py_ssize_t bytes) {
    /*
     * Partial raw decoder with column skip support.
     * 
     * This decoder differs from the standard raw decoder in that it can:
     * 1. Skip bytes at the start of each line (skip_left) for left crop
     * 2. Only decode a portion of each line (decode_width) for width crop
     * 3. Skip remaining bytes (skip_right) to reach the next line
     *
     * This allows efficient partial loading where only the needed columns
     * are decoded, not the entire row.
     */
    
    enum { LINE = 1, SKIP_LEFT, SKIP_RIGHT };
    RAWPARTIALSTATE *rawstate = state->context;

    UINT8 *ptr;

    if (state->state == 0) {
        /* Initialize context variables */

        /* Calculate bytes per decoded line based on actual decode width */
        /* Note: state->xsize should be set to the crop width, not full image width */
        state->bytes = (state->xsize * state->bits + 7) / 8;
        
        /* Calculate skip_right: stride - skip_left - bytes_to_decode */
        if (rawstate->stride) {
            rawstate->skip_right = rawstate->stride - rawstate->skip_left - state->bytes;
            if (rawstate->skip_right < 0) {
                state->errcode = IMAGING_CODEC_CONFIG;
                return -1;
            }
        } else {
            rawstate->skip_right = 0;
        }

        /* Check image orientation */
        if (state->ystep < 0) {
            state->y = state->ysize - 1;
            state->ystep = -1;
        } else {
            state->ystep = 1;
        }

        state->state = SKIP_LEFT;
    }

    ptr = buf;

    for (;;) {
        /* State: Skip left bytes (for left crop) */
        if (state->state == SKIP_LEFT) {
            if (rawstate->skip_left > 0) {
                if (bytes < rawstate->skip_left) {
                    return ptr - buf;
                }
                ptr += rawstate->skip_left;
                bytes -= rawstate->skip_left;
            }
            state->state = LINE;
        }

        /* State: Decode line data */
        if (state->state == LINE) {
            if (bytes < state->bytes) {
                return ptr - buf;
            }

            /* Unpack data - write to output at correct position */
            state->shuffle(
                (UINT8 *)im->image[state->y + state->yoff] + state->xoff * im->pixelsize,
                ptr,
                state->xsize);

            ptr += state->bytes;
            bytes -= state->bytes;

            state->y += state->ystep;

            if (state->y < 0 || state->y >= state->ysize) {
                /* End of file (errcode = 0) */
                return -1;
            }

            state->state = SKIP_RIGHT;
        }

        /* State: Skip right bytes (remaining bytes to reach next line) */
        if (state->state == SKIP_RIGHT) {
            if (rawstate->skip_right > 0) {
                if (bytes < rawstate->skip_right) {
                    return ptr - buf;
                }
                ptr += rawstate->skip_right;
                bytes -= rawstate->skip_right;
            }
            state->state = SKIP_LEFT;
        }
    }
}

