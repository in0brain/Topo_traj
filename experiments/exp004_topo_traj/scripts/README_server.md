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

## Clone

```bash
cd /data
git clone https://github.com/in0brain/Topo_traj.git
cd Topo_traj
```

## Environment

Create the conda environment from the project file:

```bash
conda env create -f environment_topo_traj.yml
conda activate topo_traj
```

Alternatively, create it manually:

```bash
conda create -n topo_traj python=3.10 -y
conda activate topo_traj
python -m pip install -r requirements_topo_traj.txt
```

## Model Path

Open:

```bash
experiments/exp004_topo_traj/configs/topo_traj_server.yaml
```

Update:

```yaml
model:
  model_name: "/data/models/Qwen2.5-1.5B-Instruct"
```

Set `model.model_name` to the actual local model directory on the server. The path above is only a template.

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
from transformers import AutoTokenizer, AutoModelForCausalLM

p = "/data/models/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(p, local_files_only=True)
print("loaded", model.config.num_hidden_layers, model.config.hidden_size)
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

If you run the smoke test and then want the full 500-sample run, use `--overwrite` for downstream outputs or clean `data/`, `cache/`, `features/`, and `outputs/` first. Do not mix smoke-test artifacts with the final server run.

## Full 300-500 Sample Run

```bash
python experiments/exp004_topo_traj/scripts/01_prepare_data.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml

python experiments/exp004_topo_traj/scripts/02_generate_cot.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml

python experiments/exp004_topo_traj/scripts/03_cache_forward.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml

python experiments/exp004_topo_traj/scripts/04_extract_markers.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite

python experiments/exp004_topo_traj/scripts/05_extract_geometry.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite

python experiments/exp004_topo_traj/scripts/06_train_geometry_probe.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite

python experiments/exp004_topo_traj/scripts/07_extract_topology_raw.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite

python experiments/exp004_topo_traj/scripts/08_train_topology_joint_probe.py --config experiments/exp004_topo_traj/configs/topo_traj_server.yaml --overwrite
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
