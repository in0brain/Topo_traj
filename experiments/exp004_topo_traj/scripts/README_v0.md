# exp004_topo_traj v0

## Current Goal

Build the minimal v0 pipeline for topology-augmented reasoning trajectory experiments:

- Experiment 0: prepare a small GSM8K sample.
- Experiment 1: generate fixed Step-format CoT and save answer correctness plus step-count metadata.

This v0 stage covers data preparation, Step-format CoT generation, second-pass forward caching for selected layers, marker position parsing, raw Step-Layer geometry tensor extraction, raw attention topology feature extraction, and logistic regression probes for geometry/topology sanity checks. It does not implement CausalGaze-style refinement or fusion.

## Environment

This experiment prioritizes the conda environment `topo_traj` instead of the existing project `.venv`.

Option 1: create from conda environment file:

```powershell
cd D:\Works
conda env create -f environment_topo_traj.yml
conda activate topo_traj
```

Option 2: manually create a conda environment:

```powershell
cd D:\Works
conda create -n topo_traj python=3.10 -y
conda activate topo_traj
python -m pip install --upgrade pip
python -m pip install -r requirements_topo_traj.txt
```

For local fallback only, the existing `.venv` can be prepared with:

```powershell
cd D:\Works
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_topo_traj.txt
```

## Run Experiment 0

```powershell
cd D:\Works
python experiments\exp004_topo_traj\scripts\01_prepare_data.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml
```

## Run Experiment 1

Test one sample with the local model first:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\02_generate_cot.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --limit 1
```

After the one-sample test passes, run all samples:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\02_generate_cot.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml
```

Optional model override:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\02_generate_cot.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --model-name D:/Models/Qwen2.5-1.5B-Instruct --limit 1
```

## Run Experiment 2

Test one valid sample first:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\03_cache_forward.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --limit 1
```

Inspect the cached tensors:

```powershell
cd D:\Works
python -c "import torch; p='experiments/exp004_topo_traj/cache/forward/gsm8k_test_000000.pt'; x=torch.load(p, map_location='cpu'); print(x.keys()); print(x['seq_len']); print(x['selected_hidden_states'].keys()); print({k:v.shape for k,v in x['selected_hidden_states'].items()}); print({k:v.shape for k,v in x['selected_attentions'].items()})"
```

After the one-sample cache test passes, run all valid samples:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\03_cache_forward.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml
```

## Run Experiment 3

Test one cached sample first:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\04_extract_markers.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --limit 1
```

Inspect the marker manifest:

```powershell
cd D:\Works
Get-Content experiments\exp004_topo_traj\cache\markers_manifest.jsonl
```

After the one-sample marker test passes, run all cached samples:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\04_extract_markers.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml
```

## Run Experiment 4

Test one parsed sample first:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\05_extract_geometry.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --limit 1
```

Inspect the tensor shapes:

```powershell
cd D:\Works
python -c "import numpy as np, json; g=np.load('experiments/exp004_topo_traj/features/geometry_raw.npy'); y=np.load('experiments/exp004_topo_traj/features/labels.npy'); print(g.shape); print(y.shape); print(json.load(open('experiments/exp004_topo_traj/features/geometry_metadata.json', encoding='utf-8')))"
```

After the one-sample geometry test passes, run all parsed samples:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\05_extract_geometry.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --overwrite
```

## Run Experiment 5

Train the geometry-only PCA + logistic regression probe:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\06_train_geometry_probe.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml
```

If probe outputs already exist, add `--overwrite` when rerunning.

Try different feature modes:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\06_train_geometry_probe.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --feature-mode term_only --overwrite

python experiments\exp004_topo_traj\scripts\06_train_geometry_probe.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --feature-mode late_only --overwrite
```

## Run Experiment 6

Test one extracted sample first:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\07_extract_topology_raw.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --limit 1
```

Inspect the topology tensor shape:

```powershell
cd D:\Works
python -c "import numpy as np, json; t=np.load('experiments/exp004_topo_traj/features/topology_raw.npy'); y=np.load('experiments/exp004_topo_traj/features/topology_labels.npy'); print(t.shape); print(y.shape); print(json.load(open('experiments/exp004_topo_traj/features/topology_raw_metadata.json', encoding='utf-8')))"
```

After the one-sample topology test passes, run all parsed samples:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\07_extract_topology_raw.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --overwrite
```

## Run Experiment 7

Suggested manual command:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\08_train_topology_joint_probe.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml
```

If probe outputs already exist, add `--overwrite` when rerunning.

Run a subset of feature sets:

```powershell
cd D:\Works
conda activate topo_traj
python experiments\exp004_topo_traj\scripts\08_train_topology_joint_probe.py --config experiments\exp004_topo_traj\configs\topo_traj_v0.yaml --feature-sets topology_only joint_concat --overwrite
```

View the metrics:

```powershell
cd D:\Works
python -c "import json; p='experiments/exp004_topo_traj/outputs/topology_joint_probe/metrics.json'; m=json.load(open(p, encoding='utf-8')); print(json.dumps(m, ensure_ascii=False, indent=2))"
```

View the report:

```powershell
cd D:\Works
Get-Content experiments\exp004_topo_traj\outputs\topology_joint_probe\report.md
```

## Outputs

- `experiments/exp004_topo_traj/data/gsm8k_30.jsonl`: sampled GSM8K records with parsed gold answers.
- `experiments/exp004_topo_traj/data/generations.jsonl`: generated CoT records with raw output, canonicalized output, correctness, marker flags, and step-count metadata.
- `experiments/exp004_topo_traj/cache/forward/{id}.pt`: selected hidden states and attentions from teacher-forcing forward over canonicalized `model_output`.
- `experiments/exp004_topo_traj/cache/forward_manifest.jsonl`: cache status, sequence lengths, labels, and skip/error records.
- `experiments/exp004_topo_traj/cache/markers/{id}.json`: Step marker, termination marker, and selected representative token positions.
- `experiments/exp004_topo_traj/cache/markers_manifest.jsonl`: marker parsing status, selected positions, warnings, and skip/error records.
- `experiments/exp004_topo_traj/features/geometry_raw.npy`: raw geometry tensor with shape `[N, 3, 3, hidden_dim]`.
- `experiments/exp004_topo_traj/features/labels.npy`: labels with `0=correct` and `1=wrong`.
- `experiments/exp004_topo_traj/features/ids.json`: sample IDs aligned with the first axis of `geometry_raw.npy`.
- `experiments/exp004_topo_traj/features/geometry_metadata.json`: geometry shape, layer/position order, and indexing rules.
- `experiments/exp004_topo_traj/features/geometry_manifest.jsonl`: per-sample extraction status.
- `experiments/exp004_topo_traj/features/geometry_pca.npy`: full-data PCA representation for later inspection and visualization only.
- `experiments/exp004_topo_traj/outputs/geometry_probe/metrics.json`: train/test and CV metrics for the geometry-only probe.
- `experiments/exp004_topo_traj/outputs/geometry_probe/predictions.jsonl`: held-out test predictions.
- `experiments/exp004_topo_traj/outputs/geometry_probe/report.md`: concise probe report.
- `experiments/exp004_topo_traj/outputs/geometry_probe/scaler.joblib`: final full-data scaler.
- `experiments/exp004_topo_traj/outputs/geometry_probe/pca_model.joblib`: final full-data PCA model.
- `experiments/exp004_topo_traj/outputs/geometry_probe/logreg_model.joblib`: final full-data logistic regression model.
- `experiments/exp004_topo_traj/features/topology_raw.npy`: raw attention topology tensor with shape `[N, 3, 3, 8]`.
- `experiments/exp004_topo_traj/features/topology_labels.npy`: labels aligned with `topology_raw.npy`.
- `experiments/exp004_topo_traj/features/topology_ids.json`: sample IDs aligned with `topology_raw.npy`.
- `experiments/exp004_topo_traj/features/topology_raw_metadata.json`: topology feature names, local window settings, and attention indexing rules.
- `experiments/exp004_topo_traj/features/topology_raw_manifest.jsonl`: per-sample topology extraction status.
- `experiments/exp004_topo_traj/outputs/topology_joint_probe/metrics.json`: comparable geometry-only, topology-only, and joint-concat probe metrics.
- `experiments/exp004_topo_traj/outputs/topology_joint_probe/predictions.jsonl`: held-out test predictions for each feature set.
- `experiments/exp004_topo_traj/outputs/topology_joint_probe/report.md`: concise comparison report.
- `experiments/exp004_topo_traj/outputs/topology_joint_probe/models/`: final full-data scaler/PCA/logistic regression artifacts per feature set.
- `experiments/exp004_topo_traj/outputs/debug_generations_preview.jsonl`: first five generated records for raw/canonical output comparison.
- `experiments/exp004_topo_traj/outputs/metrics_v0.json`: reserved for v0 metric summaries.

## Acceptance Criteria

- `gsm8k_30.jsonl` contains 30 GSM8K test samples.
- `generations.jsonl` contains one record per sample.
- Each generation record includes `id`, `question`, `gold_answer`, `prompt`, `model_output_raw`, `model_output`, `pred_answer`, `is_correct`, `num_steps`, `has_marker_raw`, `has_marker`, `canonicalized_marker`, `canonicalized_steps`, `model_name`, and `generation_config`.
- Generation completion prints total samples, successful generations, Raw marker rate, Canonical marker rate, Step parse rate, accuracy, and output paths.
- `03_cache_forward.py --limit 1` writes one `.pt` cache file for the first valid sample.
- Each `.pt` file includes `selected_hidden_states`; when attentions are enabled it also includes `selected_attentions`.
- `forward_manifest.jsonl` records cached, skipped, and error statuses.
- `04_extract_markers.py --limit 1` writes one marker JSON for the first cached sample.
- Each marker JSON includes `step_markers`, `termination_marker`, `selected_positions`, and `selected_marker_positions`.
- Full marker extraction should parse close to the 29 cached records with zero errors.
- `05_extract_geometry.py --limit 1` writes `geometry_raw.npy` with shape `[1, 3, 3, hidden_dim]`.
- Full geometry extraction should produce labels and IDs whose first dimension matches the geometry tensor.
- `06_train_geometry_probe.py` reads `geometry_raw.npy`, writes `geometry_pca.npy`, and writes probe metrics/report files.
- Probe training handles small-sample warnings without crashing.
- `07_extract_topology_raw.py --limit 1` writes `topology_raw.npy` with shape `[1, 3, 3, 8]`.
- Full topology extraction should produce topology labels and IDs whose first dimension matches the topology tensor.
- `08_train_topology_joint_probe.py` checks geometry/topology ID and label alignment before training.
- Joint probe metrics include `geometry_only`, `topology_only`, and `joint_concat` under the same train/test split and CV folds.
- The Experiment 7 report must state that v0 metrics are pipeline sanity checks only.

## Next Step

Prepare the server version: clean up `environment_topo_traj.yml`, add `README_server.md`, and scale the v0 pipeline to 300-500 samples.
