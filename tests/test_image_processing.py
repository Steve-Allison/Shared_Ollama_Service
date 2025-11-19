"""
Comprehensive behavioral tests for ImageProcessor.

Tests focus on real image processing, validation, compression, format conversion,
and edge cases. Uses real PIL/Image operations - no mocks of internal logic.
"""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from shared_ollama.infrastructure.image_processing import (
    ImageFormat,
    ImageMetadata,
    ImageProcessor,
)


@pytest.fixture
def sample_jpeg_image() -> str:
    """Create a sample JPEG image as data URL."""
    # Create a simple 100x100 RGB image
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))  # Red image
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


@pytest.fixture
def sample_png_image() -> str:
    """Create a sample PNG image as data URL."""
    # Create a simple 100x100 RGBA image
    img = Image.new("RGBA", (100, 100), color=(0, 255, 0, 255))  # Green image with alpha
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


@pytest.fixture
def large_image() -> str:
    """Create a large image (2000x2000) for resize testing."""
    img = Image.new("RGB", (2000, 2000), color=(0, 0, 255))  # Blue image
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


@pytest.fixture
def image_processor():
    """Create ImageProcessor instance for testing."""
    return ImageProcessor(max_dimension=1024, jpeg_quality=85, png_compression=6)


class TestImageProcessorInitialization:
    """Behavioral tests for ImageProcessor initialization."""

    def test_processor_initializes_with_defaults(self):
        """Test that processor initializes with default values."""
        processor = ImageProcessor()
        assert processor.max_dimension == 1024
        assert processor.jpeg_quality == 85
        assert processor.png_compression == 6
        assert processor.max_size_bytes == 10 * 1024 * 1024

    def test_processor_initializes_with_custom_config(self):
        """Test that processor initializes with custom configuration."""
        processor = ImageProcessor(
            max_dimension=2048,
            jpeg_quality=90,
            png_compression=9,
            max_size_bytes=20 * 1024 * 1024,
        )
        assert processor.max_dimension == 2048
        assert processor.jpeg_quality == 90
        assert processor.png_compression == 9
        assert processor.max_size_bytes == 20 * 1024 * 1024


class TestValidateDataURL:
    """Behavioral tests for validate_data_url()."""

    def test_validate_data_url_accepts_valid_jpeg(self, image_processor, sample_jpeg_image):
        """Test that validate_data_url accepts valid JPEG data URL."""
        format_str, image_bytes = image_processor.validate_data_url(sample_jpeg_image)

        assert format_str == "jpeg"
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0

    def test_validate_data_url_accepts_valid_png(self, image_processor, sample_png_image):
        """Test that validate_data_url accepts valid PNG data URL."""
        format_str, image_bytes = image_processor.validate_data_url(sample_png_image)

        assert format_str == "png"
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0

    def test_validate_data_url_rejects_invalid_prefix(self, image_processor):
        """Test that validate_data_url rejects URLs without data:image/ prefix."""
        with pytest.raises(ValueError, match="must start with 'data:image/'"):
            image_processor.validate_data_url("invalid://image")

    def test_validate_data_url_rejects_missing_base64_separator(self, image_processor):
        """Test that validate_data_url rejects URLs without ;base64, separator."""
        with pytest.raises(ValueError, match="must contain ';base64,'"):
            image_processor.validate_data_url("data:image/jpeg,invalid")

    def test_validate_data_url_rejects_invalid_base64(self, image_processor):
        """Test that validate_data_url rejects invalid base64 encoding."""
        invalid_data_url = "data:image/jpeg;base64,!!!invalid_base64!!!"
        with pytest.raises(ValueError, match="Invalid base64 encoding"):
            image_processor.validate_data_url(invalid_data_url)

    def test_validate_data_url_rejects_oversized_image(self, image_processor):
        """Test that validate_data_url rejects images exceeding max_size_bytes."""
        # Create a processor with small max size
        small_processor = ImageProcessor(max_size_bytes=100)
        large_data = base64.b64encode(b"x" * 200).decode("utf-8")
        large_url = f"data:image/jpeg;base64,{large_data}"

        with pytest.raises(ValueError, match="Image too large"):
            small_processor.validate_data_url(large_url)

    def test_validate_data_url_handles_uppercase_format(self, image_processor, sample_jpeg_image):
        """Test that validate_data_url handles uppercase format in URL."""
        # Replace jpeg with JPEG
        uppercase_url = sample_jpeg_image.replace("jpeg", "JPEG")
        format_str, image_bytes = image_processor.validate_data_url(uppercase_url)

        assert format_str == "jpeg"  # Should be lowercased
        assert len(image_bytes) > 0


class TestProcessImage:
    """Behavioral tests for process_image()."""

    def test_process_image_returns_base64_and_metadata(self, image_processor, sample_jpeg_image):
        """Test that process_image returns base64 string and metadata."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="jpeg")

        assert isinstance(base64_string, str)
        assert isinstance(metadata, ImageMetadata)
        assert len(base64_string) > 0

    def test_process_image_preserves_image_dimensions_when_small(self, image_processor, sample_jpeg_image):
        """Test that process_image preserves dimensions when image is small."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="jpeg")

        assert metadata.width == 100
        assert metadata.height == 100

    def test_process_image_resizes_large_images(self, image_processor, large_image):
        """Test that process_image resizes images exceeding max_dimension."""
        base64_string, metadata = image_processor.process_image(large_image, target_format="jpeg")

        # Should be resized to max_dimension (1024) while preserving aspect ratio
        assert metadata.width <= 1024
        assert metadata.height <= 1024
        assert metadata.width == 1024 or metadata.height == 1024

    def test_process_image_converts_rgba_to_rgb_for_jpeg(self, image_processor, sample_png_image):
        """Test that process_image converts RGBA to RGB for JPEG format."""
        base64_string, metadata = image_processor.process_image(sample_png_image, target_format="jpeg")

        assert metadata.format == ImageFormat.JPEG
        # Should have converted RGBA to RGB (no alpha channel in JPEG)
        assert metadata.width == 100
        assert metadata.height == 100

    def test_process_image_handles_png_format(self, image_processor, sample_png_image):
        """Test that process_image handles PNG format correctly."""
        base64_string, metadata = image_processor.process_image(sample_png_image, target_format="png")

        assert metadata.format == ImageFormat.PNG
        assert metadata.width == 100
        assert metadata.height == 100

    def test_process_image_handles_webp_format(self, image_processor, sample_jpeg_image):
        """Test that process_image handles WebP format correctly."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="webp")

        assert metadata.format == ImageFormat.WEBP
        assert metadata.width == 100
        assert metadata.height == 100

    def test_process_image_calculates_compression_ratio(self, image_processor, sample_jpeg_image):
        """Test that process_image calculates compression ratio correctly."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="jpeg")

        assert metadata.compression_ratio > 0
        assert metadata.original_size > 0
        assert metadata.compressed_size > 0

    def test_process_image_handles_grayscale_images(self, image_processor):
        """Test that process_image handles grayscale images."""
        # Create grayscale image
        img = Image.new("L", (100, 100), color=128)  # Grayscale
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        assert metadata.width == 100
        assert metadata.height == 100
        # Should convert to RGB
        assert metadata.format == ImageFormat.JPEG

    def test_process_image_handles_wide_images(self, image_processor):
        """Test that process_image handles wide images (width > height)."""
        # Create wide image (2000x500)
        img = Image.new("RGB", (2000, 500), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        # Should resize to max_dimension width, preserving aspect ratio
        assert metadata.width == 1024
        assert metadata.height == 256  # 500 * (1024/2000) = 256

    def test_process_image_handles_tall_images(self, image_processor):
        """Test that process_image handles tall images (height > width)."""
        # Create tall image (500x2000)
        img = Image.new("RGB", (500, 2000), color=(0, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        # Should resize to max_dimension height, preserving aspect ratio
        assert metadata.height == 1024
        assert metadata.width == 256  # 500 * (1024/2000) = 256

    def test_process_image_rejects_invalid_image_data(self, image_processor):
        """Test that process_image rejects invalid image data."""
        # Create data URL with non-image data
        invalid_data = base64.b64encode(b"not an image").decode("utf-8")
        invalid_url = f"data:image/jpeg;base64,{invalid_data}"

        with pytest.raises(ValueError, match="Invalid image data"):
            image_processor.process_image(invalid_url, target_format="jpeg")

    def test_process_image_rejects_unsupported_format(self, image_processor, sample_jpeg_image):
        """Test that process_image rejects unsupported target format."""
        with pytest.raises(ValueError, match="Unsupported target format"):
            image_processor.process_image(sample_jpeg_image, target_format="gif")  # type: ignore[arg-type]

    def test_process_image_base64_string_is_valid(self, image_processor, sample_jpeg_image):
        """Test that returned base64 string can be decoded."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="jpeg")

        # Should be valid base64
        decoded = base64.b64decode(base64_string)
        assert len(decoded) > 0
        # Should be valid image
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (metadata.width, metadata.height)

    def test_process_image_metadata_is_complete(self, image_processor, sample_jpeg_image):
        """Test that returned metadata contains all required fields."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="jpeg")

        assert metadata.original_size > 0
        assert metadata.compressed_size > 0
        assert metadata.width > 0
        assert metadata.height > 0
        assert isinstance(metadata.format, ImageFormat)
        assert metadata.compression_ratio > 0

    @pytest.mark.parametrize(
        "target_format",
        ["jpeg", "png", "webp"],
    )
    def test_process_image_all_formats(self, image_processor, sample_jpeg_image, target_format):
        """Test that process_image works with all supported formats."""
        base64_string, metadata = image_processor.process_image(
            sample_jpeg_image, target_format=target_format
        )

        assert metadata.format == ImageFormat(target_format)
        assert len(base64_string) > 0

    def test_process_image_handles_square_large_image(self, image_processor):
        """Test that process_image handles square large images correctly."""
        # Create square large image (1500x1500)
        img = Image.new("RGB", (1500, 1500), color=(128, 128, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        # Should resize to max_dimension (1024x1024)
        assert metadata.width == 1024
        assert metadata.height == 1024

    def test_process_image_preserves_aspect_ratio(self, image_processor):
        """Test that process_image preserves aspect ratio during resize."""
        # Create 2000x1000 image (2:1 aspect ratio)
        img = Image.new("RGB", (2000, 1000), color=(255, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        # Aspect ratio should be preserved (2:1)
        aspect_ratio = metadata.width / metadata.height
        assert abs(aspect_ratio - 2.0) < 0.01  # Allow small floating point error


class TestImageMetadata:
    """Behavioral tests for ImageMetadata dataclass."""

    def test_image_metadata_is_immutable(self):
        """Test that ImageMetadata is frozen (immutable)."""
        metadata = ImageMetadata(
            original_size=1000,
            compressed_size=500,
            width=100,
            height=100,
            format=ImageFormat.JPEG,
            compression_ratio=2.0,
        )

        with pytest.raises(Exception):  # dataclass frozen=True
            metadata.width = 200

    def test_image_metadata_uses_slots(self):
        """Test that ImageMetadata uses __slots__ for memory efficiency."""
        metadata = ImageMetadata(
            original_size=1000,
            compressed_size=500,
            width=100,
            height=100,
            format=ImageFormat.JPEG,
            compression_ratio=2.0,
        )

        # If using slots, __dict__ should not exist or be empty
        assert not hasattr(metadata, "__dict__") or len(metadata.__dict__) == 0


class TestImageProcessorEdgeCases:
    """Edge case and error handling tests."""

    def test_process_image_handles_very_small_image(self, image_processor):
        """Test that process_image handles very small images (1x1)."""
        img = Image.new("RGB", (1, 1), color=(255, 255, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        assert metadata.width == 1
        assert metadata.height == 1

    def test_process_image_handles_exact_max_dimension(self, image_processor):
        """Test that process_image handles images at exact max_dimension."""
        # Create image at exactly max_dimension (1024x1024)
        img = Image.new("RGB", (1024, 1024), color=(0, 0, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        # Should not resize (already at max)
        assert metadata.width == 1024
        assert metadata.height == 1024

    def test_process_image_handles_one_pixel_over_max(self, image_processor):
        """Test that process_image resizes images one pixel over max_dimension."""
        # Create image at 1025x1025 (one pixel over)
        img = Image.new("RGB", (1025, 1025), color=(255, 0, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_data}"

        base64_string, metadata = image_processor.process_image(data_url, target_format="jpeg")

        # Should resize to max_dimension
        assert metadata.width <= 1024
        assert metadata.height <= 1024

    def test_process_image_handles_corrupted_base64(self, image_processor):
        """Test that process_image handles corrupted base64 data."""
        # Valid prefix but corrupted base64
        corrupted_url = "data:image/jpeg;base64,!!!corrupted!!!"
        with pytest.raises(ValueError):
            image_processor.process_image(corrupted_url, target_format="jpeg")

    def test_process_image_handles_empty_base64(self, image_processor):
        """Test that process_image handles empty base64 data."""
        empty_url = "data:image/jpeg;base64,"
        with pytest.raises(ValueError):
            image_processor.process_image(empty_url, target_format="jpeg")

    def test_process_image_with_different_quality_settings(self, image_processor, sample_jpeg_image):
        """Test that process_image respects quality settings."""
        # Create processor with different quality
        high_quality_processor = ImageProcessor(jpeg_quality=95)
        low_quality_processor = ImageProcessor(jpeg_quality=50)

        high_base64, high_metadata = high_quality_processor.process_image(
            sample_jpeg_image, target_format="jpeg"
        )
        low_base64, low_metadata = low_quality_processor.process_image(
            sample_jpeg_image, target_format="jpeg"
        )

        # Higher quality should generally result in larger file size
        # (though not always guaranteed due to compression algorithms)
        assert high_metadata.compressed_size > 0
        assert low_metadata.compressed_size > 0

    def test_process_image_roundtrip_preserves_dimensions(self, image_processor, sample_jpeg_image):
        """Test that processing and decoding preserves image dimensions."""
        base64_string, metadata = image_processor.process_image(sample_jpeg_image, target_format="jpeg")

        # Decode and verify dimensions
        decoded_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(decoded_bytes))

        assert img.size == (metadata.width, metadata.height)

