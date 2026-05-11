# exp004_topo_traj Server Run Guide

## Goal

Server v1 extends the local v0 pipeline to 300-500 GSM8K samples and validates end-to-end stability for:

- CoT generation
- teacher-forced hidden states / attentions cache
- marker extraction
- geometry tensor extraction
- topology feature extraction
- geometry-only / topology-only / joint-concat probe

Current server v1 is still pipeline validation, not a final paper conclusion.

## Clone Or Pull

Clone into the server workspace:

```bash
cd /cloud/cloud-ssd1/workspace
git clone https://github.com/in0brain/Topo_traj.git
cd Topo_traj
```

If the repository already exists:

```bash
cd /cloud/cloud-ssd1/workspace/Topo_traj
git pull
```

## Environment

Activate the existing conda environment:

```bash
conda activate /cloud/cloud-ssd1/conda_envs/topo_traj
```

Install server-compatible Python dependencies from the project root:

```bash
cd /cloud/cloud-ssd1/workspace/Topo_traj
python -m pip install -r requirements_topo_traj_server.txt
```

The server currently has `torch 2.1.0+cu121`, so `requirements_topo_traj_server.txt` does not install or upgrade `torch`. It pins `transformers==4.45.2` to avoid newer Transformers releases that require PyTorch `>= 2.4`.

## Version Check

```bash
python - <<'PY'
import torch
import transformers
import tokenizers
import accelerate
import yaml
import numpy as np

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("accelerate:", accelerate.__version__)
print("numpy:", np.__version__)
print("yaml ok")
PY
```

Expected key versions:

```text
torch: 2.1.0+cu121
cuda: True
transformers: 4.45.2
```

## Model Path

Open:

```bash
experiments/exp004_topo_traj/configs/topo_traj_server.yaml
```

Confirm or update:

```yaml
model:
  model_name: "/cloud/cloud-ssd1/models/Qwen2.5-7B-Instruct"
```

Set `model.model_name` to the actual local model directory on the server if it differs.

## Initial File Checks

```bash
find experiments/exp004_topo_traj/scripts -maxdepth 1 -name "*.py" -type f | sort
find experiments/exp004_topo_traj/scripts -maxdepth 1 -name "*.py" -type f | wc -l
```

Expected `.py` count: `8`.

## Python Syntax Check

```bash
python -m py_compile \
  experiments/exp004_topo_traj/scripts/01_prepare_data.py \
  experiments/exp004_topo_traj/scripts/02_generate_cot.py \
  experiments/exp004_topo_traj/scripts/03_cache_forward.py \
  experiments/exp004_topo_traj/scripts/04_extract_markers.py \
  experiments/exp004_topo_traj/scripts/05_extract_geometry.py \
  experiments/exp004_topo_traj/scripts/06_train_geometry_probe.py \
  experiments/exp004_topo_traj/scripts/07_extract_topology_raw.py \
  experiments/exp004_topo_traj/scripts/08_train_topology_joint_probe.py
```

## Model Loading Check

```bash
python - <<'PY'
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

cfg = yaml.safe_load(open("experiments/exp004_topo_traj/configs/topo_traj_server.yaml", encoding="utf-8"))
p = cfg["model"]["model_name"]

tok = AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    p,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("loaded:", p)
print("layers:", model.config.num_hidden_layers)
print("hidden_size:", model.config.hidden_size)
print("cuda:", torch.cuda.is_available())
PY
```

## Smoke Test

Run a small smoke test with `--limit 5` where supported. `01_prepare_data.py` uses `experiment.max_samples` from the config, so it prepares the configured server sample file.

```bash
python experiments/exp004_topo_traj/scripts/01_prepare_data.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml

python experiments/exp004_topo_traj/scripts/02_generate_cot.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --limit 5

python experiments/exp004_topo_traj/scripts/03_cache_forward.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --limit 5

python experiments/exp004_topo_traj/scripts/04_extract_markers.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --limit 5

python experiments/exp004_topo_traj/scripts/05_extract_geometry.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --limit 5

python experiments/exp004_topo_traj/scripts/07_extract_topology_raw.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --limit 5
```

If you run the smoke test and then want the full 500-sample run, clean the smoke-test artifacts first or use `--overwrite` for downstream outputs. A clean reset of experiment outputs is:

```bash
rm -rf experiments/exp004_topo_traj/data/*.jsonl
rm -rf experiments/exp004_topo_traj/cache
rm -rf experiments/exp004_topo_traj/features
rm -rf experiments/exp004_topo_traj/outputs
mkdir -p experiments/exp004_topo_traj/logs
```

## Full 500-Sample Run

Use `tmux` before launching the full run:

```bash
tmux new -s topo500
cd /cloud/cloud-ssd1/workspace/Topo_traj
conda activate /cloud/cloud-ssd1/conda_envs/topo_traj
```

After disconnecting, reattach with:

```bash
tmux attach -t topo500
```

Run the full pipeline with logs:

```bash
python experiments/exp004_topo_traj/scripts/01_prepare_data.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml \
  2>&1 | tee experiments/exp004_topo_traj/logs/01_prepare_data.log

python experiments/exp004_topo_traj/scripts/02_generate_cot.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml \
  2>&1 | tee experiments/exp004_topo_traj/logs/02_generate_cot.log

python experiments/exp004_topo_traj/scripts/03_cache_forward.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml \
  2>&1 | tee experiments/exp004_topo_traj/logs/03_cache_forward.log

python experiments/exp004_topo_traj/scripts/04_extract_markers.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite \
  2>&1 | tee experiments/exp004_topo_traj/logs/04_extract_markers.log

python experiments/exp004_topo_traj/scripts/05_extract_geometry.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite \
  2>&1 | tee experiments/exp004_topo_traj/logs/05_extract_geometry.log

python experiments/exp004_topo_traj/scripts/06_train_geometry_probe.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite \
  2>&1 | tee experiments/exp004_topo_traj/logs/06_train_geometry_probe.log

python experiments/exp004_topo_traj/scripts/07_extract_topology_raw.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite \
  2>&1 | tee experiments/exp004_topo_traj/logs/07_extract_topology_raw.log

python experiments/exp004_topo_traj/scripts/08_train_topology_joint_probe.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite \
  2>&1 | tee experiments/exp004_topo_traj/logs/08_train_topology_joint_probe.log
```

## Post-Run Checks

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

base = Path("experiments/exp004_topo_traj")

print("=== Generation summary ===")
gen_path = base / "data/generations_server.jsonl"
gen = [json.loads(l) for l in open(gen_path, encoding="utf-8")]
print("generations:", len(gen))
print("canonical marker rate:", sum(x.get("has_marker", False) for x in gen) / len(gen))
print("step parse rate:", sum(x.get("num_steps", 0) > 0 for x in gen) / len(gen))
print("accuracy:", sum(x.get("is_correct", False) for x in gen) / len(gen))

print("\n=== Forward manifest ===")
fm = [json.loads(l) for l in open(base / "cache/forward_manifest.jsonl", encoding="utf-8")]
print({s: sum(x["status"] == s for x in fm) for s in ["cached", "skipped", "error"]})

print("\n=== Marker manifest ===")
mm = [json.loads(l) for l in open(base / "cache/markers_manifest.jsonl", encoding="utf-8")]
print({s: sum(x["status"] == s for x in mm) for s in ["parsed", "skipped", "error"]})

print("\n=== Feature shapes ===")
for name in ["geometry_raw.npy", "labels.npy", "topology_raw.npy", "topology_labels.npy"]:
    p = base / "features" / name
    print(name, np.load(p).shape if p.exists() else "MISSING")

print("\n=== Final metrics ===")
m = json.load(open(base / "outputs/topology_joint_probe/metrics.json", encoding="utf-8"))
print(json.dumps({
    "num_samples": m["dataset"]["num_samples"],
    "class_counts": m["dataset"]["class_counts"],
    "geometry_cv": m["feature_sets"]["geometry_only"]["cv"]["mean"],
    "topology_cv": m["feature_sets"]["topology_only"]["cv"]["mean"],
    "joint_cv": m["feature_sets"]["joint_concat"]["cv"]["mean"],
}, ensure_ascii=False, indent=2))
PY
```

## Common Issues

- If `output_attentions` is empty, check `attn_implementation: eager`.
- If GPU memory is insufficient, first run `03_cache_forward.py --no-attentions` to verify hidden-state caching. The formal topology pipeline must keep attentions enabled.
- If `seq_len` exceeds `max_seq_len`, the sample is skipped. You can reduce `max_new_tokens` or accept the skip.
- If correct/wrong classes are imbalanced, do not interpret the metrics as a formal result.
- Current raw topology is only a v1 sanity check. CausalGaze refined edges and more complex topology features should be added later.
