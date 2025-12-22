/* RawPartial.h - Header for partial raw decoding with column skip support */

#ifndef RAWPARTIAL_H
#define RAWPARTIAL_H

typedef struct {
    /* CONFIGURATION */

    /* Distance between lines in source (0=no padding) */
    int stride;

    /* Number of bytes to skip at the start of each line (for left crop) */
    int skip_left;

    /* Number of bytes to decode per line (for width crop) */
    int decode_width;

    /* PRIVATE (initialized by decoder) */

    /* Padding to skip after decoded bytes */
    int skip_right;

} RAWPARTIALSTATE;

#endif /* RAWPARTIAL_H */

