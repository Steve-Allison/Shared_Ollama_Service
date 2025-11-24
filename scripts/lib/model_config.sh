#!/bin/bash

# Model configuration loader backed by config/models.yaml.
# Selects the appropriate models based on total system RAM. Configuration-only,
# no .env overrides or manual fiddling required.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config/models.yaml"

detect_ram_gb() {
    if command -v sysctl >/dev/null 2>&1; then
        echo $(( (sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
    elif [ -r /proc/meminfo ]; then
        echo $(( $(grep -i memtotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
    else
        echo 32
    fi
}

load_profile_from_config() {
    python3 <<'PY' "$CONFIG_FILE" "$RAM_GB"
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to read config/models.yaml") from exc

config_path = Path(sys.argv[1])
ram_gb = int(sys.argv[2])

if not config_path.exists():
    raise SystemExit(f"Configuration file not found: {config_path}")

with config_path.open("r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}

profiles = data.get("profiles") or []
if not isinstance(profiles, list) or not profiles:
    raise SystemExit("No profiles defined in config/models.yaml")

def matches(profile):
    min_ram = profile.get("min_ram_gb", 0)
    max_ram = profile.get("max_ram_gb")
    if max_ram is None:
        return ram_gb >= min_ram
    return ram_gb >= min_ram and ram_gb <= max_ram

selected = None
for profile in profiles:
    if isinstance(profile, dict) and matches(profile):
        selected = profile
        break

if selected is None:
    selected = profiles[-1]

defaults = data.get("defaults") or {}

def csv_list(items):
    if not isinstance(items, list):
        items = [items]
    return ",".join(str(item) for item in items if item)

memory_hints = selected.get("memory_hints") or {}
memory_hint_pairs = ",".join(f"{k}:{v}" for k, v in memory_hints.items())

profile = {
    "DEFAULT_VLM_MODEL": selected.get("vlm_model", ""),
    "DEFAULT_TEXT_MODEL": selected.get("text_model", ""),
    "REQUIRED_MODELS_CSV": csv_list(selected.get("required_models", [])),
    "WARMUP_MODELS_CSV": csv_list(selected.get("warmup_models", [])),
    "MODEL_MEMORY_HINTS_CSV": memory_hint_pairs,
    "LARGEST_MODEL_GB": selected.get("largest_model_gb", 8),
    "INFERENCE_BUFFER_GB": selected.get("inference_buffer_gb", defaults.get("inference_buffer_gb", 4)),
    "SERVICE_OVERHEAD_GB": selected.get("service_overhead_gb", defaults.get("service_overhead_gb", 2)),
}

print(json.dumps(profile))
PY
}

load_model_config() {
    if [[ -n "${MODEL_CONFIG_LOADED:-}" ]]; then
        return
    fi

    RAM_GB=$(detect_ram_gb)
    PROFILE_JSON=$(load_profile_from_config)

    if [ -z "$PROFILE_JSON" ]; then
        echo "Failed to load model configuration from $CONFIG_FILE" >&2
        exit 1
    fi

    DEFAULT_VLM_MODEL=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['DEFAULT_VLM_MODEL'])" "$PROFILE_JSON")
    DEFAULT_TEXT_MODEL=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['DEFAULT_TEXT_MODEL'])" "$PROFILE_JSON")
    REQUIRED_MODELS_CSV=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['REQUIRED_MODELS_CSV'])" "$PROFILE_JSON")
    WARMUP_MODELS_CSV=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['WARMUP_MODELS_CSV'])" "$PROFILE_JSON")
    MODEL_MEMORY_HINTS_CSV=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['MODEL_MEMORY_HINTS_CSV'])" "$PROFILE_JSON")
    LARGEST_MODEL_GB=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['LARGEST_MODEL_GB'])" "$PROFILE_JSON")
    INFERENCE_BUFFER_GB=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['INFERENCE_BUFFER_GB'])" "$PROFILE_JSON")
    SERVICE_OVERHEAD_GB=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['SERVICE_OVERHEAD_GB'])" "$PROFILE_JSON")

    if [[ -z "$DEFAULT_VLM_MODEL" || -z "$DEFAULT_TEXT_MODEL" ]]; then
        echo "Invalid model configuration: missing vlm/text model entries." >&2
        exit 1
    fi

    IFS=',' read -r -a REQUIRED_MODELS <<< "$REQUIRED_MODELS_CSV"
    IFS=',' read -r -a WARMUP_MODELS <<< "$WARMUP_MODELS_CSV"
    MODEL_MEMORY_HINTS="$MODEL_MEMORY_HINTS_CSV"

    MODEL_CONFIG_LOADED=1
}

