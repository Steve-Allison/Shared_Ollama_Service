"""Image processing utilities for VLM endpoints.

Handles image validation, compression, format conversion, and optimization
for vision-language models.
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
    """Supported image formats."""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


@dataclass(slots=True, frozen=True)
class ImageMetadata:
    """Metadata about a processed image."""

    original_size: int  # bytes
    compressed_size: int  # bytes
    width: int
    height: int
    format: ImageFormat
    compression_ratio: float


class ImageProcessor:
    """Handles image validation, compression, and conversion for VLM models."""

    def __init__(
        self,
        max_dimension: int = 1024,
        jpeg_quality: int = 85,
        png_compression: int = 6,
        max_size_bytes: int = 10 * 1024 * 1024,  # 10MB
    ):
        """Initialize image processor.

        Args:
            max_dimension: Maximum width/height (preserves aspect ratio)
            jpeg_quality: JPEG compression quality (1-100)
            png_compression: PNG compression level (0-9)
            max_size_bytes: Maximum image size in bytes
        """
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        self.max_size_bytes = max_size_bytes

    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Validate and parse image data URL.

        Args:
            data_url: Base64-encoded data URL

        Returns:
            Tuple of (format, image_bytes)

        Raises:
            ValueError: If data URL is invalid
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
                f"Image too large: {len(image_bytes)} bytes "
                f"(max: {self.max_size_bytes})"
            )

        return img_format, image_bytes

    def process_image(
        self,
        data_url: str,
        target_format: Literal["jpeg", "png", "webp"] = "jpeg",
    ) -> tuple[str, ImageMetadata]:
        """Process and optimize image for VLM model.

        Args:
            data_url: Base64-encoded data URL
            target_format: Target image format

        Returns:
            Tuple of (base64_string, metadata)

        Raises:
            ValueError: If image is invalid
        """
        _orig_format, image_bytes = self.validate_data_url(data_url)
        original_size = len(image_bytes)

        # Load image
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise ValueError(f"Invalid image data: {exc}") from exc

        # Convert RGBA to RGB for JPEG using match/case (Python 3.13+)
        match (target_format, img.mode):
            case ("jpeg", "RGBA"):
                # Create white background for JPEG (doesn't support alpha)
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha as mask
                img = background
            case (_, mode) if mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            case _:
                pass  # Already in correct format

        # Resize if needed (preserve aspect ratio)
        width, height = img.size
        if width > self.max_dimension or height > self.max_dimension:
            # Use match/case for cleaner conditional logic
            match width > height:
                case True:
                    new_width = self.max_dimension
                    new_height = int(height * (self.max_dimension / width))
                case False:
                    new_height = self.max_dimension
                    new_width = int(width * (self.max_dimension / height))

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(
                f"Resized image from {width}x{height} to {new_width}x{new_height}"
            )
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
