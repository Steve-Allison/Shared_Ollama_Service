import base64
from pathlib import Path

image_path = Path("tests/_test_files/slide_9.png")
base64_file_path = Path("tests/_test_files/slide_9_base64.txt")

# Method 1: Read the existing base64 export if available
try:
    full_base64_data_url = base64_file_path.read_text(encoding="utf-8").strip()
except OSError:
    full_base64_data_url = ""

# Method 2: Fallback to reading binary image and encoding (should produce valid base64)
if not full_base64_data_url:
    image_bytes_raw = image_path.read_bytes()
    image_base64_fallback = base64.b64encode(image_bytes_raw).decode("utf-8")
    full_base64_data_url = f"data:image/png;base64,{image_base64_fallback}"

print(full_base64_data_url)