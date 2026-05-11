
# Works

A research project for paper reproduction and hallucination-detection experiments.

## Project Structure

```text
Works/
├─ AGENTS.md
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ .env.example
├─ papers/
├─ docs/
├─ configs/
├─ scripts/
├─ src/
├─ experiments/
├─ data/
├─ outputs/
├─ checkpoints/
├─ tests/
└─ tools/
````

* `papers/`: papers and reading notes
* `docs/`: project plans, ideas, and experiment logs
* `configs/`: configuration files for data, model, training, and evaluation
* `scripts/`: shell scripts and sync scripts for local or VM usage
* `src/`: main source code
* `experiments/`: experiment-specific notes, config snapshots, and summaries
* `data/`: datasets and processed data
* `outputs/`: logs, predictions, metrics, and plots
* `checkpoints/`: saved model checkpoints
* `tests/`: test files
* `tools/`: utility scripts

## Quick Start

### 1. Create environment

Using pip:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate works_env
```

### 2. Prepare local configuration

Copy `.env.example` to `.env` and edit it for your local machine.

Example items to configure:

* local model endpoint
* model name
* data path
* output path
* checkpoint path

### 3. Run the project

```bash
python -m src.main
```

## Notes

* Do not commit large datasets, checkpoints, or local secrets.
* Save experiment outputs under `outputs/`.
* Save important experiment summaries under `experiments/`.
* Keep machine-specific settings in `.env` or local config files.


