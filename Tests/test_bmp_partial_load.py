"""
Tests for BMP partial loading optimization.

This module tests the load_region() and load_region_c_optimized() methods
added to BmpImageFile for efficient partial BMP loading.
"""

import os
import tempfile
import pytest
import numpy as np
from PIL import Image
from PIL.BmpImagePlugin import BmpImageFile


class TestBmpPartialLoad:
    """Test suite for BMP partial loading functionality."""

    @pytest.fixture
    def temp_bmp_rgb(self):
        """Create a temporary 24-bit RGB BMP file."""
        data = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        img = Image.fromarray(data, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'BMP')
        yield filepath
        os.unlink(filepath)

    @pytest.fixture
    def temp_bmp_rgba(self):
        """Create a temporary 32-bit RGBA BMP file."""
        data = np.arange(100 * 100 * 4, dtype=np.uint8).reshape(100, 100, 4)
        img = Image.fromarray(data, mode='RGBA')
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'BMP')
        yield filepath
        os.unlink(filepath)

    @pytest.fixture
    def temp_bmp_large(self):
        """Create a larger temporary BMP file for realistic testing."""
        data = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        img = Image.fromarray(data, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'BMP')
        yield filepath
        os.unlink(filepath)

    @pytest.fixture
    def temp_bmp_palette(self):
        """Create a temporary 8-bit palette BMP file."""
        data = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        img = Image.fromarray(data, mode='L').convert('P')
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'BMP')
        yield filepath
        os.unlink(filepath)

    # ========================================================================
    # Basic functionality tests
    # ========================================================================

    def test_load_region_basic(self, temp_bmp_rgb):
        """Test basic load_region functionality."""
        with Image.open(temp_bmp_rgb) as img:
            region = img.load_region((10, 10, 50, 50))
            
            assert region.size == (40, 40)
            assert region.mode == 'RGB'

    def test_load_region_c_optimized_basic(self, temp_bmp_rgb):
        """Test basic load_region_c_optimized functionality."""
        with Image.open(temp_bmp_rgb) as img:
            region = img.load_region_c_optimized((10, 10, 50, 50))
            
            assert region.size == (40, 40)
            assert region.mode == 'RGB'

    def test_load_region_full_image(self, temp_bmp_rgb):
        """Test load_region with full image bounds."""
        with Image.open(temp_bmp_rgb) as img:
            region = img.load_region((0, 0, 100, 100))
            
            assert region.size == (100, 100)

    def test_load_region_c_optimized_full_image(self, temp_bmp_rgb):
        """Test load_region_c_optimized with full image bounds."""
        with Image.open(temp_bmp_rgb) as img:
            region = img.load_region_c_optimized((0, 0, 100, 100))
            
            assert region.size == (100, 100)

    # ========================================================================
    # Correctness tests - compare with traditional method
    # ========================================================================

    def test_load_region_correctness(self, temp_bmp_rgb):
        """Verify load_region produces identical results to traditional crop."""
        crop_box = (10, 20, 60, 80)
        
        # Traditional method
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        # load_region
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_c_optimized_correctness(self, temp_bmp_rgb):
        """Verify load_region_c_optimized produces identical results."""
        crop_box = (10, 20, 60, 80)
        
        # Traditional method
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        # load_region_c_optimized
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_correctness_random_crops(self, temp_bmp_large):
        """Test correctness with multiple random crop boxes."""
        import random
        random.seed(42)
        
        for _ in range(20):
            left = random.randint(0, 400)
            upper = random.randint(0, 400)
            right = left + random.randint(10, 100)
            lower = upper + random.randint(10, 100)
            crop_box = (left, upper, min(right, 500), min(lower, 500))
            
            # Traditional
            with Image.open(temp_bmp_large) as img:
                img.load()
                expected = np.array(img._new(img._crop(img.im, crop_box)))
            
            # load_region
            with Image.open(temp_bmp_large) as img:
                result1 = np.array(img.load_region(crop_box))
            
            # load_region_c_optimized
            with Image.open(temp_bmp_large) as img:
                result2 = np.array(img.load_region_c_optimized(crop_box))
            
            np.testing.assert_array_equal(result1, expected, 
                err_msg=f"load_region failed for box {crop_box}")
            np.testing.assert_array_equal(result2, expected,
                err_msg=f"load_region_c_optimized failed for box {crop_box}")

    # ========================================================================
    # Edge cases
    # ========================================================================

    def test_load_region_edge_left(self, temp_bmp_rgb):
        """Test crop at left edge."""
        crop_box = (0, 10, 30, 50)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_edge_right(self, temp_bmp_rgb):
        """Test crop at right edge."""
        crop_box = (70, 10, 100, 50)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_edge_top(self, temp_bmp_rgb):
        """Test crop at top edge."""
        crop_box = (10, 0, 50, 30)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_edge_bottom(self, temp_bmp_rgb):
        """Test crop at bottom edge."""
        crop_box = (10, 70, 50, 100)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_corner_top_left(self, temp_bmp_rgb):
        """Test crop at top-left corner."""
        crop_box = (0, 0, 30, 30)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_corner_bottom_right(self, temp_bmp_rgb):
        """Test crop at bottom-right corner."""
        crop_box = (70, 70, 100, 100)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_single_pixel(self, temp_bmp_rgb):
        """Test crop of a single pixel."""
        crop_box = (50, 50, 51, 51)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_single_row(self, temp_bmp_rgb):
        """Test crop of a single row."""
        crop_box = (10, 50, 90, 51)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_single_column(self, temp_bmp_rgb):
        """Test crop of a single column."""
        crop_box = (50, 10, 51, 90)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    # ========================================================================
    # Error handling tests
    # ========================================================================

    def test_load_region_invalid_box_negative(self, temp_bmp_rgb):
        """Test that negative coordinates raise ValueError."""
        with Image.open(temp_bmp_rgb) as img:
            with pytest.raises(ValueError):
                img.load_region((-1, 10, 50, 50))
            
            with pytest.raises(ValueError):
                img.load_region((10, -1, 50, 50))

    def test_load_region_invalid_box_reversed(self, temp_bmp_rgb):
        """Test that reversed coordinates raise ValueError."""
        with Image.open(temp_bmp_rgb) as img:
            with pytest.raises(ValueError):
                img.load_region((50, 10, 10, 50))  # right < left
            
            with pytest.raises(ValueError):
                img.load_region((10, 50, 50, 10))  # lower < upper

    def test_load_region_invalid_box_exceeds(self, temp_bmp_rgb):
        """Test that box exceeding image bounds raises ValueError."""
        with Image.open(temp_bmp_rgb) as img:
            with pytest.raises(ValueError):
                img.load_region((10, 10, 150, 50))  # right > width
            
            with pytest.raises(ValueError):
                img.load_region((10, 10, 50, 150))  # lower > height

    def test_load_region_c_optimized_invalid_box(self, temp_bmp_rgb):
        """Test that load_region_c_optimized also validates input."""
        with Image.open(temp_bmp_rgb) as img:
            with pytest.raises(ValueError):
                img.load_region_c_optimized((-1, 10, 50, 50))

    # ========================================================================
    # Different image modes
    # ========================================================================

    def test_load_region_rgba(self, temp_bmp_rgba):
        """Test load_region with RGBA image."""
        crop_box = (10, 10, 50, 50)
        
        with Image.open(temp_bmp_rgba) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgba) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_load_region_palette(self, temp_bmp_palette):
        """Test load_region with palette image."""
        crop_box = (10, 10, 50, 50)
        
        with Image.open(temp_bmp_palette) as img:
            # load_region should work for palette images
            region = img.load_region(crop_box)
            assert region.size == (40, 40)

    # ========================================================================
    # crop() method with use_partial_load parameter
    # ========================================================================

    def test_crop_default_uses_optimization(self, temp_bmp_rgb):
        """Test that crop() uses optimization by default for BMP."""
        crop_box = (10, 10, 50, 50)
        
        with Image.open(temp_bmp_rgb) as img:
            # Should use load_region internally
            region = img.crop(crop_box)
            assert region.size == (40, 40)

    def test_crop_use_partial_load_false(self, temp_bmp_rgb):
        """Test crop() with use_partial_load=False uses traditional method."""
        crop_box = (10, 10, 50, 50)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.crop(crop_box, use_partial_load=False))
        
        np.testing.assert_array_equal(result, expected)

    def test_crop_use_partial_load_true(self, temp_bmp_rgb):
        """Test crop() with use_partial_load=True."""
        crop_box = (10, 10, 50, 50)
        
        with Image.open(temp_bmp_rgb) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(temp_bmp_rgb) as img:
            result = np.array(img.crop(crop_box, use_partial_load=True))
        
        np.testing.assert_array_equal(result, expected)

    # ========================================================================
    # Class-level ENABLE_PARTIAL_LOAD flag
    # ========================================================================

    def test_enable_partial_load_flag(self, temp_bmp_rgb):
        """Test ENABLE_PARTIAL_LOAD class flag."""
        crop_box = (10, 10, 50, 50)
        
        # Save original value
        original = BmpImageFile.ENABLE_PARTIAL_LOAD
        
        try:
            # Disable partial loading
            BmpImageFile.ENABLE_PARTIAL_LOAD = False
            
            with Image.open(temp_bmp_rgb) as img:
                # Should use traditional method
                region = img.crop(crop_box)
                assert region.size == (40, 40)
        finally:
            # Restore original value
            BmpImageFile.ENABLE_PARTIAL_LOAD = original

    # ========================================================================
    # Multiple operations on same file
    # ========================================================================

    def test_multiple_crops_same_file(self, temp_bmp_large):
        """Test multiple crop operations on the same file."""
        crops = [
            (0, 0, 100, 100),
            (100, 100, 200, 200),
            (200, 200, 300, 300),
        ]
        
        for crop_box in crops:
            with Image.open(temp_bmp_large) as img:
                img.load()
                expected = np.array(img._new(img._crop(img.im, crop_box)))
            
            with Image.open(temp_bmp_large) as img:
                result = np.array(img.load_region_c_optimized(crop_box))
            
            np.testing.assert_array_equal(result, expected)

    # ========================================================================
    # File not closed during operation
    # ========================================================================

    def test_file_handle_not_closed(self, temp_bmp_rgb):
        """Ensure file handle is still valid after load_region."""
        with Image.open(temp_bmp_rgb) as img:
            region1 = img.load_region((10, 10, 50, 50))
            # File should still be open for another operation
            region2 = img.load_region((20, 20, 60, 60))
            
            assert region1.size == (40, 40)
            assert region2.size == (40, 40)


class TestBmpPartialLoadNonBmp:
    """Test that non-BMP formats are not affected."""

    @pytest.fixture
    def temp_png(self):
        """Create a temporary PNG file."""
        data = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        img = Image.fromarray(data, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'PNG')
        yield filepath
        os.unlink(filepath)

    def test_png_no_load_region(self, temp_png):
        """Test that PNG images don't have load_region method."""
        with Image.open(temp_png) as img:
            assert not hasattr(img, 'load_region')

    def test_png_crop_works_normally(self, temp_png):
        """Test that PNG crop works normally."""
        with Image.open(temp_png) as img:
            region = img.crop((10, 10, 50, 50))
            assert region.size == (40, 40)


class TestBmpPartialLoadRle:
    """Test that RLE-compressed BMPs fall back to traditional loading."""

    @pytest.fixture
    def rle8_bmp(self):
        """Get path to RLE8 compressed BMP file."""
        return "Tests/images/hopper_rle8.bmp"

    def test_rle_no_optimization(self, rle8_bmp):
        """Test that RLE BMPs don't use partial loading optimization."""
        if not os.path.exists(rle8_bmp):
            pytest.skip("RLE8 test image not found")
        
        with Image.open(rle8_bmp) as img:
            # RLE compression should NOT support partial loading
            # _bmp_info should be None for RLE
            assert img._bmp_info is None or img._bmp_info.get("compression", 0) != 0
            
            # crop should still work (via fallback)
            region = img.crop((5, 5, 50, 50))
            assert region.size == (45, 45)


class TestBmpPartialLoadStress:
    """Stress tests for partial loading."""

    @pytest.fixture
    def stress_bmp(self):
        """Create a larger BMP for stress testing."""
        data = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        img = Image.fromarray(data, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'BMP')
        yield filepath
        os.unlink(filepath)

    def test_many_sequential_crops(self, stress_bmp):
        """Test many sequential crops from the same image."""
        import random
        random.seed(123)
        
        for _ in range(50):
            left = random.randint(0, 900)
            upper = random.randint(0, 900)
            right = left + random.randint(10, 100)
            lower = upper + random.randint(10, 100)
            crop_box = (left, upper, right, lower)
            
            # Compare all three methods
            with Image.open(stress_bmp) as img:
                img.load()
                expected = np.array(img._new(img._crop(img.im, crop_box)))
            
            with Image.open(stress_bmp) as img:
                result1 = np.array(img.load_region(crop_box))
            
            with Image.open(stress_bmp) as img:
                result2 = np.array(img.load_region_c_optimized(crop_box))
            
            np.testing.assert_array_equal(result1, expected)
            np.testing.assert_array_equal(result2, expected)

    def test_large_crop(self, stress_bmp):
        """Test cropping a large region (almost entire image)."""
        crop_box = (10, 10, 990, 990)
        
        with Image.open(stress_bmp) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(stress_bmp) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)

    def test_tiny_crop(self, stress_bmp):
        """Test cropping a tiny region (3x3 pixels)."""
        crop_box = (500, 500, 503, 503)
        
        with Image.open(stress_bmp) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(stress_bmp) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)


class TestBmpPartialLoadDifferentBitDepths:
    """Test partial loading with different bit depths."""

    def _create_bmp_and_test(self, mode, size=(100, 100)):
        """Helper to create BMP and test partial loading."""
        if mode in ('L', 'P'):
            data = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size)
            img = Image.fromarray(data, mode='L')
            if mode == 'P':
                img = img.convert('P')
        elif mode == 'RGB':
            data = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[0], size[1], 3)
            img = Image.fromarray(data, mode='RGB')
        elif mode == 'RGBA':
            data = np.arange(size[0] * size[1] * 4, dtype=np.uint8).reshape(size[0], size[1], 4)
            img = Image.fromarray(data, mode='RGBA')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        try:
            img.save(filepath, 'BMP')
            
            crop_box = (10, 10, 50, 50)
            
            # Traditional
            with Image.open(filepath) as opened:
                opened.load()
                expected = np.array(opened._new(opened._crop(opened.im, crop_box)))
            
            # load_region
            with Image.open(filepath) as opened:
                result1 = np.array(opened.load_region(crop_box))
            
            # load_region_c_optimized
            with Image.open(filepath) as opened:
                result2 = np.array(opened.load_region_c_optimized(crop_box))
            
            np.testing.assert_array_equal(result1, expected)
            np.testing.assert_array_equal(result2, expected)
        finally:
            os.unlink(filepath)

    def test_8bit_grayscale(self):
        """Test 8-bit grayscale BMP."""
        self._create_bmp_and_test('L')

    def test_8bit_palette(self):
        """Test 8-bit palette BMP."""
        self._create_bmp_and_test('P')

    def test_24bit_rgb(self):
        """Test 24-bit RGB BMP."""
        self._create_bmp_and_test('RGB')

    def test_32bit_rgba(self):
        """Test 32-bit RGBA BMP."""
        self._create_bmp_and_test('RGBA')


class TestBmpPartialLoadRowAlignment:
    """Test partial loading with different row alignments.
    
    BMP rows are padded to 4-byte boundaries. Test widths that exercise
    different padding scenarios.
    """

    @pytest.fixture(params=[
        99,   # 99 * 3 = 297 bytes -> needs 3 bytes padding
        100,  # 100 * 3 = 300 bytes -> needs 0 bytes padding
        101,  # 101 * 3 = 303 bytes -> needs 1 byte padding
        102,  # 102 * 3 = 306 bytes -> needs 2 bytes padding
    ])
    def aligned_bmp(self, request):
        """Create BMP with specific width to test row alignment."""
        width = request.param
        data = np.random.randint(0, 256, (50, width, 3), dtype=np.uint8)
        img = Image.fromarray(data, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            filepath = f.name
        
        img.save(filepath, 'BMP')
        yield filepath, width
        os.unlink(filepath)

    def test_alignment_correctness(self, aligned_bmp):
        """Test that different row alignments are handled correctly."""
        filepath, width = aligned_bmp
        
        crop_box = (5, 5, min(width - 5, 50), 40)
        
        with Image.open(filepath) as img:
            img.load()
            expected = np.array(img._new(img._crop(img.im, crop_box)))
        
        with Image.open(filepath) as img:
            result = np.array(img.load_region_c_optimized(crop_box))
        
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

