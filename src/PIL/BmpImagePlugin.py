#
# The Python Imaging Library.
# $Id$
#
# BMP file handler
#
# Windows (and OS/2) native bitmap storage format.
#
# history:
# 1995-09-01 fl   Created
# 1996-04-30 fl   Added save
# 1997-08-27 fl   Fixed save of 1-bit images
# 1998-03-06 fl   Load P images as L where possible
# 1998-07-03 fl   Load P images as 1 where possible
# 1998-12-29 fl   Handle small palettes
# 2002-12-30 fl   Fixed load of 1-bit palette images
# 2003-04-21 fl   Fixed load of 1-bit monochrome images
# 2003-04-23 fl   Added limited support for BI_BITFIELDS compression
#
# Copyright (c) 1997-2003 by Secret Labs AB
# Copyright (c) 1995-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#


import os

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32

#
# --------------------------------------------------------------------
# Read BMP file

BIT2MODE = {
    # bits => mode, rawmode
    1: ("P", "P;1"),
    4: ("P", "P;4"),
    8: ("P", "P"),
    16: ("RGB", "BGR;15"),
    24: ("RGB", "BGR"),
    32: ("RGB", "BGRX"),
}


def _accept(prefix):
    return prefix[:2] == b"BM"


def _dib_accept(prefix):
    return i32(prefix) in [12, 40, 64, 108, 124]


# =============================================================================
# Image plugin for the Windows BMP format.
# =============================================================================
class BmpImageFile(ImageFile.ImageFile):
    """Image plugin for the Windows Bitmap format (BMP)"""

    # ------------------------------------------------------------- Description
    format_description = "Windows Bitmap"
    format = "BMP"

    # -------------------------------------------------- BMP Compression values
    COMPRESSIONS = {"RAW": 0, "RLE8": 1, "RLE4": 2, "BITFIELDS": 3, "JPEG": 4, "PNG": 5}
    for k, v in COMPRESSIONS.items():
        vars()[k] = v

    # Class-level flag to enable/disable partial loading optimization
    # Set to False to use traditional full-load behavior for benchmarking
    ENABLE_PARTIAL_LOAD = True

    def __init__(self, fp=None, filename=None):
        # Store file info for partial loading
        self._bmp_info = None
        self._data_offset = None
        self._raw_mode = None
        self._stride = None
        super().__init__(fp, filename)

    def _bitmap(self, header=0, offset=0):
        """Read relevant info about the BMP"""
        read, seek = self.fp.read, self.fp.seek
        if header:
            seek(header)
        # read bmp header size @offset 14 (this is part of the header size)
        file_info = {"header_size": i32(read(4)), "direction": -1}

        # -------------------- If requested, read header at a specific position
        # read the rest of the bmp header, without its size
        header_data = ImageFile._safe_read(self.fp, file_info["header_size"] - 4)

        # -------------------------------------------------- IBM OS/2 Bitmap v1
        # ----- This format has different offsets because of width/height types
        if file_info["header_size"] == 12:
            file_info["width"] = i16(header_data, 0)
            file_info["height"] = i16(header_data, 2)
            file_info["planes"] = i16(header_data, 4)
            file_info["bits"] = i16(header_data, 6)
            file_info["compression"] = self.RAW
            file_info["palette_padding"] = 3

        # --------------------------------------------- Windows Bitmap v2 to v5
        # v3, OS/2 v2, v4, v5
        elif file_info["header_size"] in (40, 64, 108, 124):
            file_info["y_flip"] = header_data[7] == 0xFF
            file_info["direction"] = 1 if file_info["y_flip"] else -1
            file_info["width"] = i32(header_data, 0)
            file_info["height"] = (
                i32(header_data, 4)
                if not file_info["y_flip"]
                else 2**32 - i32(header_data, 4)
            )
            file_info["planes"] = i16(header_data, 8)
            file_info["bits"] = i16(header_data, 10)
            file_info["compression"] = i32(header_data, 12)
            # byte size of pixel data
            file_info["data_size"] = i32(header_data, 16)
            file_info["pixels_per_meter"] = (
                i32(header_data, 20),
                i32(header_data, 24),
            )
            file_info["colors"] = i32(header_data, 28)
            file_info["palette_padding"] = 4
            self.info["dpi"] = tuple(x / 39.3701 for x in file_info["pixels_per_meter"])
            if file_info["compression"] == self.BITFIELDS:
                if len(header_data) >= 52:
                    for idx, mask in enumerate(
                        ["r_mask", "g_mask", "b_mask", "a_mask"]
                    ):
                        file_info[mask] = i32(header_data, 36 + idx * 4)
                else:
                    # 40 byte headers only have the three components in the
                    # bitfields masks, ref:
                    # https://msdn.microsoft.com/en-us/library/windows/desktop/dd183376(v=vs.85).aspx
                    # See also
                    # https://github.com/python-pillow/Pillow/issues/1293
                    # There is a 4th component in the RGBQuad, in the alpha
                    # location, but it is listed as a reserved component,
                    # and it is not generally an alpha channel
                    file_info["a_mask"] = 0x0
                    for mask in ["r_mask", "g_mask", "b_mask"]:
                        file_info[mask] = i32(read(4))
                file_info["rgb_mask"] = (
                    file_info["r_mask"],
                    file_info["g_mask"],
                    file_info["b_mask"],
                )
                file_info["rgba_mask"] = (
                    file_info["r_mask"],
                    file_info["g_mask"],
                    file_info["b_mask"],
                    file_info["a_mask"],
                )
        else:
            msg = f"Unsupported BMP header type ({file_info['header_size']})"
            raise OSError(msg)

        # ------------------ Special case : header is reported 40, which
        # ---------------------- is shorter than real size for bpp >= 16
        self._size = file_info["width"], file_info["height"]

        # ------- If color count was not found in the header, compute from bits
        file_info["colors"] = (
            file_info["colors"]
            if file_info.get("colors", 0)
            else (1 << file_info["bits"])
        )
        if offset == 14 + file_info["header_size"] and file_info["bits"] <= 8:
            offset += 4 * file_info["colors"]

        # ---------------------- Check bit depth for unusual unsupported values
        self.mode, raw_mode = BIT2MODE.get(file_info["bits"], (None, None))
        if self.mode is None:
            msg = f"Unsupported BMP pixel depth ({file_info['bits']})"
            raise OSError(msg)

        # ---------------- Process BMP with Bitfields compression (not palette)
        decoder_name = "raw"
        if file_info["compression"] == self.BITFIELDS:
            SUPPORTED = {
                32: [
                    (0xFF0000, 0xFF00, 0xFF, 0x0),
                    (0xFF000000, 0xFF0000, 0xFF00, 0x0),
                    (0xFF000000, 0xFF0000, 0xFF00, 0xFF),
                    (0xFF, 0xFF00, 0xFF0000, 0xFF000000),
                    (0xFF0000, 0xFF00, 0xFF, 0xFF000000),
                    (0x0, 0x0, 0x0, 0x0),
                ],
                24: [(0xFF0000, 0xFF00, 0xFF)],
                16: [(0xF800, 0x7E0, 0x1F), (0x7C00, 0x3E0, 0x1F)],
            }
            MASK_MODES = {
                (32, (0xFF0000, 0xFF00, 0xFF, 0x0)): "BGRX",
                (32, (0xFF000000, 0xFF0000, 0xFF00, 0x0)): "XBGR",
                (32, (0xFF000000, 0xFF0000, 0xFF00, 0xFF)): "ABGR",
                (32, (0xFF, 0xFF00, 0xFF0000, 0xFF000000)): "RGBA",
                (32, (0xFF0000, 0xFF00, 0xFF, 0xFF000000)): "BGRA",
                (32, (0x0, 0x0, 0x0, 0x0)): "BGRA",
                (24, (0xFF0000, 0xFF00, 0xFF)): "BGR",
                (16, (0xF800, 0x7E0, 0x1F)): "BGR;16",
                (16, (0x7C00, 0x3E0, 0x1F)): "BGR;15",
            }
            if file_info["bits"] in SUPPORTED:
                if (
                    file_info["bits"] == 32
                    and file_info["rgba_mask"] in SUPPORTED[file_info["bits"]]
                ):
                    raw_mode = MASK_MODES[(file_info["bits"], file_info["rgba_mask"])]
                    self.mode = "RGBA" if "A" in raw_mode else self.mode
                elif (
                    file_info["bits"] in (24, 16)
                    and file_info["rgb_mask"] in SUPPORTED[file_info["bits"]]
                ):
                    raw_mode = MASK_MODES[(file_info["bits"], file_info["rgb_mask"])]
                else:
                    msg = "Unsupported BMP bitfields layout"
                    raise OSError(msg)
            else:
                msg = "Unsupported BMP bitfields layout"
                raise OSError(msg)
        elif file_info["compression"] == self.RAW:
            if file_info["bits"] == 32 and header == 22:  # 32-bit .cur offset
                raw_mode, self.mode = "BGRA", "RGBA"
        elif file_info["compression"] in (self.RLE8, self.RLE4):
            decoder_name = "bmp_rle"
        else:
            msg = f"Unsupported BMP compression ({file_info['compression']})"
            raise OSError(msg)

        # --------------- Once the header is processed, process the palette/LUT
        if self.mode == "P":  # Paletted for 1, 4 and 8 bit images
            # ---------------------------------------------------- 1-bit images
            if not (0 < file_info["colors"] <= 65536):
                msg = f"Unsupported BMP Palette size ({file_info['colors']})"
                raise OSError(msg)
            else:
                padding = file_info["palette_padding"]
                palette = read(padding * file_info["colors"])
                greyscale = True
                indices = (
                    (0, 255)
                    if file_info["colors"] == 2
                    else list(range(file_info["colors"]))
                )

                # ----------------- Check if greyscale and ignore palette if so
                for ind, val in enumerate(indices):
                    rgb = palette[ind * padding : ind * padding + 3]
                    if rgb != o8(val) * 3:
                        greyscale = False

                # ------- If all colors are grey, white or black, ditch palette
                if greyscale:
                    self.mode = "1" if file_info["colors"] == 2 else "L"
                    raw_mode = self.mode
                else:
                    self.mode = "P"
                    self.palette = ImagePalette.raw(
                        "BGRX" if padding == 4 else "BGR", palette
                    )

        # ---------------------------- Finally set the tile data for the plugin
        self.info["compression"] = file_info["compression"]
        
        # Calculate stride (bytes per row including padding)
        stride = ((file_info["width"] * file_info["bits"] + 31) >> 3) & (~3)
        
        args = [raw_mode]
        if decoder_name == "bmp_rle":
            args.append(file_info["compression"] == self.RLE4)
        else:
            args.append(stride)
        args.append(file_info["direction"])
        
        # Store info for partial loading
        self._bmp_info = file_info
        self._data_offset = offset or self.fp.tell()
        self._raw_mode = raw_mode
        self._stride = stride
        self._decoder_name = decoder_name
        
        self.tile = [
            (
                decoder_name,
                (0, 0, file_info["width"], file_info["height"]),
                self._data_offset,
                tuple(args),
            )
        ]

    def _open(self):
        """Open file, check magic number and read header"""
        # read 14 bytes: magic number, filesize, reserved, header final offset
        head_data = self.fp.read(14)
        # choke if the file does not have the required magic bytes
        if not _accept(head_data):
            msg = "Not a BMP file"
            raise SyntaxError(msg)
        # read the start position of the BMP image data (u32)
        offset = i32(head_data, 10)
        # load bitmap information (offset=raster info)
        self._bitmap(offset=offset)

    def load_region(self, box):
        """
        Load only a region of the image, optimized for uncompressed BMP files.
        
        This method provides significant performance improvements for large BMP files
        when only a small region is needed, by:
        - Reducing disk I/O (skipping unneeded rows)
        - Reducing memory usage (allocating only the crop region)
        - Reducing decode time (only decoding needed rows)
        
        The implementation directly uses the C-layer raw decoder for maximum efficiency.
        
        :param box: A 4-tuple (left, upper, right, lower) defining the region.
        :returns: An Image object containing only the requested region.
        :raises ValueError: If the box is invalid.
        :raises OSError: If the image uses RLE compression (not supported for partial load).
        
        Example usage::
        
            with Image.open("large_image.bmp") as img:
                # Only loads the specified region from disk
                region = img.load_region((100, 100, 500, 500))
        """
        left, upper, right, lower = box
        
        # Validate box
        if left < 0 or upper < 0:
            msg = "Box coordinates must be non-negative"
            raise ValueError(msg)
        if right <= left or lower <= upper:
            msg = "Invalid box: right must be > left, lower must be > upper"
            raise ValueError(msg)
        if right > self.size[0] or lower > self.size[1]:
            msg = f"Box {box} exceeds image size {self.size}"
            raise ValueError(msg)
        
        # Check if partial loading is supported
        if self._bmp_info is None:
            msg = "Image info not available for partial loading"
            raise OSError(msg)
        
        compression = self._bmp_info.get("compression", -1)
        if compression in (self.RLE4, self.RLE8):
            # RLE compression requires sequential decoding, fall back to full load + crop
            self.load()
            return self.crop(box)
        
        # For uncompressed BMP, we can do partial loading
        crop_width = right - left
        crop_height = lower - upper
        
        # BMP stores rows from bottom to top (direction = -1) or top to bottom (direction = 1)
        direction = self._bmp_info.get("direction", -1)
        img_height = self._bmp_info["height"]
        img_width = self._bmp_info["width"]
        
        # Calculate which rows we need to read from the file
        if direction == -1:  # Bottom-up (most common)
            # Row 0 in file = bottom row of image (y = height - 1)
            # We want rows from 'upper' to 'lower-1' in image coordinates
            # In file: row (height - lower) to row (height - upper - 1)
            file_start_row = img_height - lower
        else:  # Top-down
            file_start_row = upper
        
        # Calculate file offset to skip unneeded rows
        row_offset = self._data_offset + file_start_row * self._stride
        
        # Create a temporary image to hold the full-width rows we need
        # This allows us to use C-layer decoding directly
        temp_im = Image.new(self.mode, (img_width, crop_height))
        
        # Copy palette if exists
        if self.mode == "P" and self.palette:
            temp_im.putpalette(self.palette)
        
        # Set up the tile for C-layer decoding
        # tile format: (decoder_name, extents, offset, args)
        # extents: (x0, y0, x1, y1) - region in the OUTPUT image to write to
        # offset: position in file to start reading
        # args: (rawmode, stride, direction)
        tile = [(
            "raw",
            (0, 0, img_width, crop_height),  # Write to full width of temp image
            row_offset,                       # Start reading from calculated offset
            (self._raw_mode, self._stride, direction)
        )]
        
        # Use ImageFile's tile-based loading mechanism (calls C decoder)
        self.fp.seek(row_offset)
        decoder = Image._getdecoder(
            self.mode, "raw", (self._raw_mode, self._stride, direction)
        )
        decoder.setimage(temp_im.im, (0, 0, img_width, crop_height))
        
        # Read and decode - let C layer handle the decoding
        bytes_to_read = crop_height * self._stride
        raw_data = ImageFile._safe_read(self.fp, bytes_to_read)
        decoder.decode(raw_data)
        decoder.cleanup()
        
        # Now crop horizontally using C-layer crop (im.crop is implemented in C)
        if left == 0 and right == img_width:
            # No horizontal crop needed
            return temp_im
        else:
            # Use C-layer crop for horizontal extraction
            return temp_im.crop((left, 0, right, crop_height))

    def load_region_c_optimized(self, box):
        """
        Load only a region of the image using C-layer partial decoding.
        
        This is a more optimized version that uses a custom C decoder
        (raw_partial) to decode only the needed columns directly,
        avoiding the need for a second crop operation.
        
        This method provides additional performance improvements over
        load_region() by:
        - Decoding only the needed columns (not full rows)
        - Eliminating the Python bytes object overhead
        - Eliminating the secondary crop operation
        
        :param box: A 4-tuple (left, upper, right, lower) defining the region.
        :returns: An Image object containing only the requested region.
        :raises ValueError: If the box is invalid.
        :raises OSError: If the image uses RLE compression or unsupported bit depth.
        
        Example usage::
        
            with Image.open("large_image.bmp") as img:
                # Uses C-layer optimized partial decoding
                region = img.load_region_c_optimized((100, 100, 500, 500))
        """
        left, upper, right, lower = box
        
        # Validate box
        if left < 0 or upper < 0:
            msg = "Box coordinates must be non-negative"
            raise ValueError(msg)
        if right <= left or lower <= upper:
            msg = "Invalid box: right must be > left, lower must be > upper"
            raise ValueError(msg)
        if right > self.size[0] or lower > self.size[1]:
            msg = f"Box {box} exceeds image size {self.size}"
            raise ValueError(msg)
        
        # Check if partial loading is supported
        if self._bmp_info is None:
            msg = "Image info not available for partial loading"
            raise OSError(msg)
        
        compression = self._bmp_info.get("compression", -1)
        if compression in (self.RLE4, self.RLE8):
            # RLE compression requires sequential decoding, fall back
            return self.load_region(box)
        
        bits = self._bmp_info["bits"]
        # C-layer partial decoding only works well for 8+ bit images
        if bits < 8:
            # Fall back to Python implementation for sub-byte pixels
            return self.load_region(box)
        
        # For uncompressed BMP with 8+ bits, use C-layer partial decoding
        crop_width = right - left
        crop_height = lower - upper
        
        # BMP stores rows from bottom to top (direction = -1) or top to bottom (direction = 1)
        direction = self._bmp_info.get("direction", -1)
        img_height = self._bmp_info["height"]
        
        # Calculate which rows we need to read from the file
        if direction == -1:  # Bottom-up (most common)
            file_start_row = img_height - lower
        else:  # Top-down
            file_start_row = upper
        
        # Calculate file offset to skip unneeded rows
        row_offset = self._data_offset + file_start_row * self._stride
        
        # Calculate bytes to skip for left crop
        bytes_per_pixel = bits // 8
        skip_left_bytes = left * bytes_per_pixel
        
        # Create output image with exact crop dimensions
        out_im = Image.new(self.mode, (crop_width, crop_height))
        
        # Copy palette if exists
        if self.mode == "P" and self.palette:
            out_im.putpalette(self.palette)
        
        # Seek to starting position
        self.fp.seek(row_offset)
        
        # Use the new raw_partial decoder
        # Args: mode, rawmode, stride, ystep, skip_left
        decoder = Image._getdecoder(
            self.mode, "raw_partial", 
            (self._raw_mode, self._stride, direction, skip_left_bytes)
        )
        decoder.setimage(out_im.im, (0, 0, crop_width, crop_height))
        
        # Read and decode - still need to read full rows but decoder skips columns
        bytes_to_read = crop_height * self._stride
        raw_data = ImageFile._safe_read(self.fp, bytes_to_read)
        decoder.decode(raw_data)
        decoder.cleanup()
        
        return out_im

    def crop(self, box=None, *, use_partial_load=None):
        """
        Returns a rectangular region from this image.
        
        This is an optimized version that uses partial loading for uncompressed BMP files.
        For RLE-compressed files, it falls back to the standard crop behavior.
        
        :param box: The crop rectangle, as a (left, upper, right, lower)-tuple.
        :param use_partial_load: Override the default partial loading behavior.
            - None (default): Use class-level ENABLE_PARTIAL_LOAD setting
            - True: Force partial loading (will raise if not supported)
            - False: Force traditional full-load + crop
        :rtype: :py:class:`~PIL.Image.Image`
        :returns: An :py:class:`~PIL.Image.Image` object.
        
        Example::
        
            # Use default behavior (auto-detect)
            region = img.crop((100, 100, 500, 500))
            
            # Force traditional method (for benchmarking)
            region = img.crop((100, 100, 500, 500), use_partial_load=False)
            
            # Force optimized method
            region = img.crop((100, 100, 500, 500), use_partial_load=True)
            
            # Or disable globally for benchmarking:
            BmpImageFile.ENABLE_PARTIAL_LOAD = False
        """
        if box is None:
            return self.copy()
        
        # Determine whether to use partial loading
        if use_partial_load is None:
            use_partial_load = self.ENABLE_PARTIAL_LOAD
        
        # Check if we can use optimized partial loading
        if (
            use_partial_load
            and self._bmp_info is not None
            and self._bmp_info.get("compression", -1) not in (self.RLE4, self.RLE8)
            and self.fp is not None
            and not getattr(self, '_loaded', False)
        ):
            try:
                # Use C-optimized version for best performance
                return self.load_region_c_optimized(box)
            except (OSError, ValueError):
                if use_partial_load is True:
                    # User explicitly requested partial load, re-raise
                    raise
                # Fall back to standard crop if partial load fails
                pass
        
        # Standard crop behavior
        self.load()
        self._loaded = True
        return self._new(self._crop(self.im, box))

    def load(self):
        """Load image data based on tile list"""
        result = super().load()
        self._loaded = True
        return result


class BmpRleDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        rle4 = self.args[1]
        data = bytearray()
        x = 0
        while len(data) < self.state.xsize * self.state.ysize:
            pixels = self.fd.read(1)
            byte = self.fd.read(1)
            if not pixels or not byte:
                break
            num_pixels = pixels[0]
            if num_pixels:
                # encoded mode
                if x + num_pixels > self.state.xsize:
                    # Too much data for row
                    num_pixels = max(0, self.state.xsize - x)
                if rle4:
                    first_pixel = o8(byte[0] >> 4)
                    second_pixel = o8(byte[0] & 0x0F)
                    for index in range(num_pixels):
                        if index % 2 == 0:
                            data += first_pixel
                        else:
                            data += second_pixel
                else:
                    data += byte * num_pixels
                x += num_pixels
            else:
                if byte[0] == 0:
                    # end of line
                    while len(data) % self.state.xsize != 0:
                        data += b"\x00"
                    x = 0
                elif byte[0] == 1:
                    # end of bitmap
                    break
                elif byte[0] == 2:
                    # delta
                    bytes_read = self.fd.read(2)
                    if len(bytes_read) < 2:
                        break
                    right, up = self.fd.read(2)
                    data += b"\x00" * (right + up * self.state.xsize)
                    x = len(data) % self.state.xsize
                else:
                    # absolute mode
                    if rle4:
                        # 2 pixels per byte
                        byte_count = byte[0] // 2
                        bytes_read = self.fd.read(byte_count)
                        for byte_read in bytes_read:
                            data += o8(byte_read >> 4)
                            data += o8(byte_read & 0x0F)
                    else:
                        byte_count = byte[0]
                        bytes_read = self.fd.read(byte_count)
                        data += bytes_read
                    if len(bytes_read) < byte_count:
                        break
                    x += byte[0]

                    # align to 16-bit word boundary
                    if self.fd.tell() % 2 != 0:
                        self.fd.seek(1, os.SEEK_CUR)
        rawmode = "L" if self.mode == "L" else "P"
        self.set_as_raw(bytes(data), (rawmode, 0, self.args[-1]))
        return -1, 0


# =============================================================================
# Image plugin for the DIB format (BMP alias)
# =============================================================================
class DibImageFile(BmpImageFile):
    format = "DIB"
    format_description = "Windows Bitmap"

    def _open(self):
        self._bitmap()


#
# --------------------------------------------------------------------
# Write BMP file


SAVE = {
    "1": ("1", 1, 2),
    "L": ("L", 8, 256),
    "P": ("P", 8, 256),
    "RGB": ("BGR", 24, 0),
    "RGBA": ("BGRA", 32, 0),
}


def _dib_save(im, fp, filename):
    _save(im, fp, filename, False)


def _save(im, fp, filename, bitmap_header=True):
    try:
        rawmode, bits, colors = SAVE[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as BMP"
        raise OSError(msg) from e

    info = im.encoderinfo

    dpi = info.get("dpi", (96, 96))

    # 1 meter == 39.3701 inches
    ppm = tuple(map(lambda x: int(x * 39.3701 + 0.5), dpi))

    stride = ((im.size[0] * bits + 7) // 8 + 3) & (~3)
    header = 40  # or 64 for OS/2 version 2
    image = stride * im.size[1]

    if im.mode == "1":
        palette = b"".join(o8(i) * 4 for i in (0, 255))
    elif im.mode == "L":
        palette = b"".join(o8(i) * 4 for i in range(256))
    elif im.mode == "P":
        palette = im.im.getpalette("RGB", "BGRX")
        colors = len(palette) // 4
    else:
        palette = None

    # bitmap header
    if bitmap_header:
        offset = 14 + header + colors * 4
        file_size = offset + image
        if file_size > 2**32 - 1:
            msg = "File size is too large for the BMP format"
            raise ValueError(msg)
        fp.write(
            b"BM"  # file type (magic)
            + o32(file_size)  # file size
            + o32(0)  # reserved
            + o32(offset)  # image data offset
        )

    # bitmap info header
    fp.write(
        o32(header)  # info header size
        + o32(im.size[0])  # width
        + o32(im.size[1])  # height
        + o16(1)  # planes
        + o16(bits)  # depth
        + o32(0)  # compression (0=uncompressed)
        + o32(image)  # size of bitmap
        + o32(ppm[0])  # resolution
        + o32(ppm[1])  # resolution
        + o32(colors)  # colors used
        + o32(colors)  # colors important
    )

    fp.write(b"\0" * (header - 40))  # padding (for OS/2 format)

    if palette:
        fp.write(palette)

    ImageFile._save(im, fp, [("raw", (0, 0) + im.size, 0, (rawmode, stride, -1))])


#
# --------------------------------------------------------------------
# Direct Decode API for PyTorch Integration
#

def decode_bmp_to_tensor(
    filename,
    box=None,
    out_tensor=None,
    normalize=True,
    drop_alpha=True,
    out_channels=3
):
    """
    Decode a BMP image directly into a PyTorch tensor without intermediate copies.
    
    This function provides maximum performance for loading BMP images into PyTorch
    by performing all operations (crop, normalize, CHW conversion) in a single pass
    through the data, minimizing memory bandwidth usage.
    
    :param filename: Path to the BMP file.
    :param box: Optional crop box as (left, upper, right, lower). If None, loads full image.
    :param out_tensor: Optional pre-allocated PyTorch tensor (CHW, float32).
                       If None, a new tensor will be created.
    :param normalize: If True, normalize pixel values to [0, 1] range.
    :param drop_alpha: If True, ignore alpha channel even if present in the image.
    :param out_channels: Number of output channels (1, 3, or 4). Default is 3.
    :returns: PyTorch tensor with shape (C, H, W) and dtype float32.
    :raises RuntimeError: If the file cannot be decoded.
    :raises ImportError: If PyTorch is not available.
    
    Example::
    
        import torch
        from PIL.BmpImagePlugin import decode_bmp_to_tensor
        
        # Load full image
        tensor = decode_bmp_to_tensor("image.bmp")
        
        # Load cropped region directly into pre-allocated tensor
        out = torch.empty(3, 512, 512, dtype=torch.float32)
        decode_bmp_to_tensor("image.bmp", box=(100, 100, 612, 612), out_tensor=out)
        
        # Load with custom settings
        tensor = decode_bmp_to_tensor(
            "image.bmp",
            box=(0, 0, 256, 256),
            normalize=False,  # Keep values in [0, 255]
            out_channels=1    # Convert to grayscale
        )
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for decode_bmp_to_tensor")
    
    import numpy as np
    from PIL import _imaging, Image
    
    # Try fast C path first (supports RGB, RGBA, Grayscale, and Palette with AVX2 Gather)
    # Fallback to Pillow if unsupported format (e.g., RLE compression)
    try:
        return _decode_bmp_to_tensor_fast(
            filename, box, out_tensor, normalize, drop_alpha, out_channels, torch, _imaging
        )
    except RuntimeError as e:
        # Check if this is an "unsupported format" error that we can fallback from
        # vs a genuine error (file not found, invalid crop) that should propagate
        error_msg = str(e).lower()
        if 'unsupported' in error_msg or 'compression' in error_msg:
            # Fallback to slower but safer Pillow path for RLE, etc.
            return _decode_bmp_to_tensor_fallback(
                filename, box, out_tensor, normalize, drop_alpha, out_channels, torch, np
            )
        else:
            # Re-raise other errors (file not found, invalid crop, etc.)
            raise


def _decode_bmp_to_tensor_fast(filename, box, out_tensor, normalize, drop_alpha, out_channels, torch, _imaging):
    """Fast C-based decoding path."""
    # OPTIMIZATION: Only call bmp_get_info when absolutely necessary
    # (when box is None and we need the full image dimensions)
    # This avoids an extra file open/read/close cycle on Lustre
    
    if box is not None:
        # Box is provided, we know the crop dimensions without reading the file
        x0, y0, x1, y1 = box
        crop_width = x1 - x0
        crop_height = y1 - y0
        # For out_channels, default to 3 (most common case)
        # The C layer will handle the actual channel conversion
        if out_channels is None:
            out_channels = 3
    else:
        # No box, need to get full image dimensions
        width, height, channels = _imaging.bmp_get_info(filename)
        x0, y0, x1, y1 = 0, 0, width, height
        crop_width = width
        crop_height = height
        if out_channels is None:
            out_channels = 3 if drop_alpha else min(channels, 4)
    
    # Create or validate output tensor
    if out_tensor is None:
        out_tensor = torch.empty(out_channels, crop_height, crop_width, dtype=torch.float32)
    else:
        # Validate shape
        if out_tensor.dim() != 3:
            raise ValueError(f"out_tensor must be 3D, got {out_tensor.dim()}D")
        if out_tensor.shape[0] < out_channels:
            raise ValueError(f"out_tensor has {out_tensor.shape[0]} channels, need {out_channels}")
        if out_tensor.shape[1] < crop_height or out_tensor.shape[2] < crop_width:
            raise ValueError(f"out_tensor too small: {out_tensor.shape[1:]} < ({crop_height}, {crop_width})")
        if out_tensor.dtype != torch.float32:
            raise ValueError(f"out_tensor must be float32, got {out_tensor.dtype}")
        if out_tensor.device.type != 'cpu':
            raise ValueError(f"out_tensor must be on CPU, got {out_tensor.device}")
        if not out_tensor.is_contiguous():
            raise ValueError("out_tensor must be contiguous")
    
    # Get data pointer and strides
    out_ptr = out_tensor.data_ptr()
    stride_c = out_tensor.stride(0)
    stride_y = out_tensor.stride(1)
    stride_x = out_tensor.stride(2)
    
    # Call C function (may raise RuntimeError for unsupported formats)
    _imaging.bmp_decode_to_chw(
        filename,
        x0, y0, x1, y1,
        out_channels,
        out_ptr,
        stride_c, stride_y, stride_x,
        1 if normalize else 0,
        1 if drop_alpha else 0
    )
    
    return out_tensor


def _decode_bmp_to_tensor_fallback(filename, box, out_tensor, normalize, drop_alpha, out_channels, torch, np):
    """Fallback path using Pillow for unsupported formats (RLE, etc.)."""
    from PIL import Image
    
    with Image.open(filename) as img:
        # Apply crop if specified
        if box is not None:
            img = img.crop(box)
        
        # Convert to RGB/RGBA if needed
        if img.mode == 'P':
            img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
        elif img.mode == 'L':
            pass  # Keep grayscale
        elif img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        # Convert to numpy array
        arr = np.array(img, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            arr = arr / 255.0
        
        # Handle grayscale
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        
        # Transpose to CHW
        arr = arr.transpose(2, 0, 1)
        
        # Handle channels
        if out_channels is None:
            out_channels = 3 if drop_alpha else arr.shape[0]
        
        if drop_alpha and arr.shape[0] == 4:
            arr = arr[:3]
        
        # Ensure correct number of channels
        if arr.shape[0] < out_channels:
            # Expand grayscale to RGB if needed
            if arr.shape[0] == 1 and out_channels >= 3:
                arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] > out_channels:
            arr = arr[:out_channels]
        
        # Create or fill output tensor
        if out_tensor is None:
            return torch.from_numpy(arr.copy())
        else:
            # Validate and copy to existing tensor
            if out_tensor.dim() != 3:
                raise ValueError(f"out_tensor must be 3D, got {out_tensor.dim()}D")
            if out_tensor.dtype != torch.float32:
                raise ValueError(f"out_tensor must be float32, got {out_tensor.dtype}")
            if out_tensor.device.type != 'cpu':
                raise ValueError(f"out_tensor must be on CPU, got {out_tensor.device}")
            
            h, w = arr.shape[1], arr.shape[2]
            c = min(arr.shape[0], out_tensor.shape[0])
            out_tensor[:c, :h, :w] = torch.from_numpy(arr[:c])
            return out_tensor


def get_bmp_info(filename):
    """
    Get BMP image dimensions without loading the image data.
    
    :param filename: Path to the BMP file.
    :returns: Tuple of (width, height, channels).
    :raises RuntimeError: If the file cannot be read.
    
    Example::
    
        from PIL.BmpImagePlugin import get_bmp_info
        width, height, channels = get_bmp_info("image.bmp")
    """
    from PIL import _imaging
    return _imaging.bmp_get_info(filename)


#
# --------------------------------------------------------------------
# Registry


Image.register_open(BmpImageFile.format, BmpImageFile, _accept)
Image.register_save(BmpImageFile.format, _save)

Image.register_extension(BmpImageFile.format, ".bmp")

Image.register_mime(BmpImageFile.format, "image/bmp")

Image.register_decoder("bmp_rle", BmpRleDecoder)

Image.register_open(DibImageFile.format, DibImageFile, _dib_accept)
Image.register_save(DibImageFile.format, _dib_save)

Image.register_extension(DibImageFile.format, ".dib")

Image.register_mime(DibImageFile.format, "image/bmp")
