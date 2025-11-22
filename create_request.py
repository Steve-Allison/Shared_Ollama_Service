import base64
import json
import os

tmp_dir = "/Users/steveallison/AI_Projects+Code/Shared_Ollama_Service/.gemini/tmp/5bb35cc71b2c48e001f024367a07cd29c5931219aaf89be8fec89fda78cb0b21"
os.makedirs(tmp_dir, exist_ok=True)

image_path = "tests/_test_files/slide_9.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

payload = {
    "model": "qwen3-vl:8b-instruct-q4_K_M",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this picture?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }
    ],
}

with open(os.path.join(tmp_dir, "vlm_request.json"), "w") as f:
    json.dump(payload, f)
