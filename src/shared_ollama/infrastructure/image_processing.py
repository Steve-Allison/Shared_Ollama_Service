"""Image processing utilities for VLM endpoints.

This module provides image validation, compression, format conversion, and
optimization for vision-language models. Images are processed to fit within
dimension limits, compressed for bandwidth efficiency, and converted to
target formats.

Key Features:
    - Data URL validation and parsing
    - Image format detection and conversion (JPEG, PNG, WebP)
    - Automatic resizing to fit dimension limits (preserves aspect ratio)
    - Image compression for bandwidth optimization
    - RGBA to RGB conversion for JPEG compatibility
    - Metadata extraction (dimensions, size, compression ratio)

Dependencies:
    - Pillow (PIL): Required for image processing operations
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

try:
    from PIL import Image
except ImportError as exc:
    msg = "Pillow is required for image processing. Install with: pip install Pillow"
    raise ImportError(msg) from exc

logger = logging.getLogger(__name__)


class ImageFormat(StrEnum):
    """Supported image formats for processing and conversion.

    Enumeration of image formats supported by the image processor.
    All formats are supported for input, and can be converted to
    any other format for output.

    Attributes:
        JPEG: JPEG format (best for photos, lossy compression).
        PNG: PNG format (best for graphics, lossless compression).
        WEBP: WebP format (modern, good compression, supports transparency).
    """

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


@dataclass(slots=True, frozen=True)
class ImageMetadata:
    """Metadata about a processed image.

    Immutable metadata object containing information about image processing
    results, including original and processed sizes, dimensions, format, and
    compression statistics.

    Attributes:
        original_size: Original image size in bytes before processing.
        compressed_size: Processed image size in bytes after compression/resizing.
        width: Image width in pixels after processing.
        height: Image height in pixels after processing.
        format: Final image format after processing (ImageFormat enum).
        compression_ratio: Compression ratio (compressed_size / original_size).
            Values < 1.0 indicate compression, > 1.0 indicate expansion.
    """

    original_size: int  # bytes
    compressed_size: int  # bytes
    width: int
    height: int
    format: ImageFormat
    compression_ratio: float


class ImageProcessor:
    """Handles image validation, compression, and conversion for VLM models.

    Processes images for vision-language model requests by validating data URLs,
    resizing to fit dimension limits, compressing for bandwidth efficiency,
    and converting to target formats.

    The processor preserves aspect ratio during resizing and handles format
    conversions (including RGBA to RGB for JPEG compatibility). Compression
    settings are configurable per format.

    Attributes:
        max_dimension: Maximum width or height in pixels. Images exceeding
            this are resized while preserving aspect ratio.
        jpeg_quality: JPEG compression quality (1-100). Higher = better
            quality but larger files. Default: 85 (good balance).
        png_compression: PNG compression level (0-9). Higher = better compression
            but slower. Default: 6 (balanced).
        max_size_bytes: Maximum image size in bytes before processing.
            Images exceeding this are rejected. Default: 10MB.

    Note:
        This class implements ImageProcessorInterface and is used by
        ImageProcessorAdapter in the infrastructure layer.
    """

    def __init__(
        self,
        max_dimension: int = 2667,
        jpeg_quality: int = 85,
        png_compression: int = 6,
        max_size_bytes: int = 10 * 1024 * 1024,  # 10MB
    ):
        """Initialize image processor.

        Args:
            max_dimension: Maximum width or height in pixels. Images larger
                than this are resized while preserving aspect ratio.
                Range: [256, 2667]. Default: 2667.
            jpeg_quality: JPEG compression quality (1-100). Higher values
                produce better quality but larger files. Default: 85.
            png_compression: PNG compression level (0-9). Higher values compress
                better but process slower. Default: 6.
            max_size_bytes: Maximum image size in bytes before processing.
                Images exceeding this are rejected. Default: 10MB (10 * 1024 * 1024).

        Note:
            Compression settings affect file size vs quality tradeoff.
            Defaults are optimized for VLM model performance and bandwidth.
        """
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        self.max_size_bytes = max_size_bytes

    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Validate and parse image data URL.

        Validates the data URL format, extracts image format and raw bytes,
        and checks size limits. Does not perform image processing.

        Args:
            data_url: Base64-encoded data URL. Format:
                "data:image/{format};base64,{base64_data}"

        Returns:
            Tuple of (format, image_bytes) where:
                - format: Image format string (e.g., "jpeg", "png", "webp")
                - image_bytes: Raw image bytes decoded from base64

        Raises:
            ValueError: If:
                - data_url doesn't start with "data:image/"
                - data_url is missing ";base64," separator
                - base64 decoding fails
                - image size exceeds max_size_bytes

        Note:
            This method performs validation only. Use process_image() for
            actual image processing, resizing, and compression.
        """
        if not data_url.startswith("data:image/"):
            raise ValueError("Image URL must start with 'data:image/'")
        if ";base64," not in data_url:
            raise ValueError("Image URL must contain ';base64,' separator")

        # Extract format and data
        header, base64_data = data_url.split(";base64,", 1)
        img_format = header.split("/", 1)[1].lower()

        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as exc:
            raise ValueError(f"Invalid base64 encoding: {exc}") from exc

        if len(image_bytes) > self.max_size_bytes:
            raise ValueError(
                f"Image too large: {len(image_bytes)} bytes " f"(max: {self.max_size_bytes})"
            )

        return img_format, image_bytes

    def process_image(
        self,
        data_url: str,
        target_format: Literal["jpeg", "png", "webp"] = "jpeg",
    ) -> tuple[str, ImageMetadata]:
        """Process and optimize image for VLM model.

        Validates, resizes, compresses, and converts an image to the target
        format. Images are optimized to fit within dimension limits and
        compressed for efficient transmission to the VLM model.

        Processing Steps:
            1. Validate and parse data URL
            2. Load image and validate image data
            3. Convert color modes (RGBA to RGB for JPEG)
            4. Resize if dimensions exceed max_dimension (preserves aspect ratio)
            5. Compress and convert to target format
            6. Encode as base64 data URL
            7. Extract and return metadata

        Args:
            data_url: Base64-encoded image data URL. Must be valid format.
            target_format: Target image format. One of "jpeg", "png", or "webp".
                Default: "jpeg" (best compression for photos).

        Returns:
            Tuple of (base64_string, metadata) where:
                - base64_string: Processed image as base64 data URL in format
                  "data:image/{target_format};base64,{base64_data}"
                - metadata: ImageMetadata object with processing results

        Raises:
            ValueError: If:
                - data URL is invalid (see validate_data_url)
                - image data is corrupted or unreadable
                - image format is unsupported
                - image processing fails

        Note:
            Aspect ratio is always preserved during resizing. RGBA images
            are converted to RGB with white background for JPEG format
            (JPEG doesn't support transparency).
        """
        _orig_format, image_bytes = self.validate_data_url(data_url)
        original_size = len(image_bytes)

        # Load image (force load to catch truncated streams)
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
        except Exception as exc:
            raise ValueError(f"Invalid image data: {exc}") from exc

        # Convert RGBA to RGB for JPEG using match/case (Python 3.13+)
        match (target_format, img.mode):
            case ("jpeg", "RGBA"):
                # Create white background for JPEG (doesn't support alpha)
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha as mask
                img = background
            case (_, mode) if mode not in {"RGB", "RGBA"}:
                img = img.convert("RGB")
            case _:
                pass  # Already in correct format

        # Resize if needed (preserve aspect ratio)
        width, height = img.size
        if width > self.max_dimension or height > self.max_dimension:
            # Use match/case for cleaner conditional logic (define defaults first for type checkers)
            new_width = width
            new_height = height
            match width > height:
                case True:
                    new_width = self.max_dimension
                    new_height = int(height * (self.max_dimension / width))
                case False:
                    new_height = self.max_dimension
                    new_width = int(width * (self.max_dimension / height))

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            width, height = new_width, new_height

        # Compress using match/case for format selection (Python 3.13+)
        output = io.BytesIO()
        match target_format:
            case "jpeg":
                img.save(
                    output,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                )
            case "webp":
                # WebP supports both lossy and lossless - use lossy with same quality as JPEG
                img.save(
                    output,
                    format="WEBP",
                    quality=self.jpeg_quality,
                    method=6,  # Slowest/best compression
                )
            case "png":
                img.save(
                    output,
                    format="PNG",
                    compress_level=self.png_compression,
                    optimize=True,
                )
            case _:
                raise ValueError(f"Unsupported target format: {target_format}")

        compressed_bytes = output.getvalue()
        compressed_size = len(compressed_bytes)
        base64_string = base64.b64encode(compressed_bytes).decode("utf-8")

        metadata = ImageMetadata(
            original_size=original_size,
            compressed_size=compressed_size,
            width=width,
            height=height,
            format=ImageFormat(target_format),
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
        )

        logger.info(
            f"Processed image: {width}x{height} {target_format}, "
            f"compressed {original_size} → {compressed_size} bytes "
            f"(ratio: {metadata.compression_ratio:.2f}x)"
        )

        return base64_string, metadata
