#!/usr/bin/env bash
# FML-bench-Lite OpenOOD baseline/candidate execution entry point.
#
# Executed by the OpenResearch control plane inside a fresh detached checkout of
# an exact commit SHA. Every failure must propagate (no `|| true`, no silent
# retries). This script is harness infrastructure; worker-modifiable
# files are algorithm.py and split_config.json.
#
# Usage: bash public_validation/run.sh {val|test}
set -euo pipefail

SPLIT="${1:?usage: run.sh val or test}"
case "$SPLIT" in val|test) ;; *) echo "unknown split: $SPLIT" >&2; exit 2;; esac

FROZEN_VENV="${FROZEN_VENV:-/home/zhoujunjie/openresearch-envs/pycil-baseline}"
DATA_ROOT="${DATA_ROOT:-/home/zhoujunjie/openresearch-data}"
EVALUATOR_SHA256="daa13119eb4e1e5aff6b86b5a1aac2c12d21b246b842a66d7bfa13cd2fb7b92c"

PY="$FROZEN_VENV/bin/python"
[ -x "$PY" ] || { echo "FATAL: frozen venv python missing: $PY" >&2; exit 3; }

export PATH="$FROZEN_VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0

# --- dataset caches ---
mkdir -p data
# CIFAR-10
if [ -d "$DATA_ROOT/cifar-10/cifar-10-batches-py" ]; then
    cp -r "$DATA_ROOT/cifar-10/cifar-10-batches-py" data/
elif [ -f "$DATA_ROOT/cifar-10-python.tar.gz" ]; then
    tar -xzf "$DATA_ROOT/cifar-10-python.tar.gz" -C data
else
    echo "FATAL: CIFAR-10 dataset missing" >&2
    exit 4
fi

# CIFAR-100 (for val OOD)
if [ -d "$DATA_ROOT/cifar-100-python" ]; then
    cp -r "$DATA_ROOT/cifar-100-python" data/
elif [ -f "$DATA_ROOT/cifar-100-python.tar.gz" ]; then
    tar -xzf "$DATA_ROOT/cifar-100-python.tar.gz" -C data
fi

# SVHN (for test OOD)
mkdir -p data/svhn
if [ -f "$DATA_ROOT/svhn/test_32x32.mat" ]; then
    cp "$DATA_ROOT/svhn/test_32x32.mat" data/svhn/
fi

# --- evaluator: pinned identity, server cache first ---
EVAL_CACHE="${EVAL_CACHE:-/home/zhoujunjie/openresearch-evaluator/openood/train_eval_baseline.py}"
check_evaluator() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$EVALUATOR_SHA256" ]; }

if [ -f "$EVAL_CACHE" ] && check_evaluator "$EVAL_CACHE"; then
    cp "$EVAL_CACHE" train_eval_baseline.py
elif [ -f train_eval_baseline.py ] && check_evaluator train_eval_baseline.py; then
    : # already present and correct
else
    echo "FATAL: evaluator could not be sourced/verified (want sha256 $EVALUATOR_SHA256)" >&2
    exit 5
fi

# --- environment identity ---
ENV_IDENTITY_JSON=$("$PY" - <<'PYEOF'
import hashlib, json, pathlib, platform, subprocess, sys
identity = {
    "hostname": platform.node(),
    "os_release": "",
    "kernel": platform.release(),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
}
try:
    identity["os_release"] = open("/etc/os-release").read().split("PRETTY_NAME=")[1].split("\n")[0].strip('"')
except Exception as exc:
    identity["os_release"] = f"unreadable: {exc}"
import torch
identity["torch"] = torch.__version__
identity["torch_cuda"] = torch.version.cuda
identity["cudnn"] = torch.backends.cudnn.version()
identity["cuda_available"] = torch.cuda.is_available()
identity["gpu_count"] = torch.cuda.device_count()
if torch.cuda.is_available():
    identity["gpu_name"] = torch.cuda.get_device_name(0)
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
identity["evaluator_sha256"] = sha256_file("train_eval_baseline.py")
lock = "/home/zhoujunjie/openresearch-envs/requirements-lock.txt"
identity["requirements_lock_sha256"] = sha256_file(lock) if pathlib.Path(lock).exists() else ""
probe = subprocess.run(
    ["nvidia-smi", "--id=0", "--query-gpu=index,name,driver_version,memory.total",
     "--format=csv,noheader"], capture_output=True, text=True)
identity["nvidia_smi_gpu0"] = probe.stdout.strip() or f"exit {probe.returncode}: {probe.stderr.strip()}"
print("ENV_IDENTITY_JSON " + json.dumps(identity, sort_keys=True))
PYEOF
)
echo "$ENV_IDENTITY_JSON"

# --- training + evaluation (metric of record) ---
rm -rf results_tmp
"$PY" train_eval_baseline.py --split "$SPLIT"

# --- emit machine-readable result ---
"$PY" - "$SPLIT" <<'PYEOF'
import json, pathlib, sys
info = json.loads(pathlib.Path(f"results_tmp/{sys.argv[1]}_info.json").read_text())
means = info["openood"]["means"]
print("BASELINE_RESULT_JSON " + json.dumps({
    "schema": "openresearch.baseline-result.v1",
    "metric": "auroc_mean",
    "value": means["auroc_mean"],
    "fpr95_mean": means.get("fpr95_mean"),
    "source": f"results_tmp/{sys.argv[1]}_info.json",
}, sort_keys=True))
PYEOF
