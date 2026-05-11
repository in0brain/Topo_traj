# AGENTS.md

## Project purpose
This project is for paper reproduction and hallucination-detection experiments.

## Rules
- Prefer minimal and local edits.
- Do not rename directories unless necessary.
- Keep source code under /src.
- Keep configs under /configs.
- Keep experiment notes under /experiments and /docs.
- Do not modify files under /papers unless explicitly asked.
- Do not store large datasets or checkpoints in git.

## Entry points
- Main code entry: src/main.py
- Training pipeline: src/pipeline/train_pipeline.py
- Evaluation pipeline: src/pipeline/eval_pipeline.py
- Reflection pipeline: src/pipeline/reflect_pipeline.py

## Common directories
- Source code: /src
- Configs: /configs
- Scripts: /scripts
- Papers: /papers
- Docs: /docs
- Outputs: /outputs
- Checkpoints: /checkpoints
- Tests: /tests

## Coding preferences
- Keep functions small and readable.
- Add type hints when reasonable.
- Avoid large refactors unless requested.
- Prefer config-driven changes over hard-coded constants.

## Notes for experiments
- Save run outputs to /outputs
- Save important summaries to /experiments
- Keep reproducibility in mind