# LLMOps

LLMOps is a course project that demonstrates end-to-end training, evaluation, and deployment workflows for a language model (LLM) using Google Cloud Storage and Vertex AI. The repository contains data preparation tools, training/evaluation pipelines, helper scripts, and examples to reproduce experiments and analyses from the course.

## Overview

- Purpose: Provide a reproducible pipeline to prepare tarot-card training data, fine-tune or evaluate models, and produce predictions and evaluation reports using Vertex AI and Google Cloud Storage.
- Scope: data generation, training orchestration, evaluation, model registration, and simple local demos.

## Status

- Work-in-progress: core scripts and pipelines are present under `scripts/`, `src/`, and `data_prep/`.

## Key Concepts and Components

- Data preparation: `data_prep/` contains generators and utilities to build the tarot training datasets used for experiments.
- Pipelines and orchestration: `scripts/pipeline_runner.py` and related pipeline JSON files orchestrate training and evaluation runs.
- Model handler & serving: `src/handler.py` and `register_model_with_custom_handler.py` help register models with a custom handler for serving on Vertex AI.
- Experiments and results: `PIPELINE_RESULTS/` stores evaluation outputs, predictions, and TensorBoard logs from runs.

## Prerequisites

- Python 3.11 (tested on 3.11.6)
- A Python virtual environment. This repo uses `uv` in Project Setup examples, but any `venv`/`pip`/`pipx` workflow works.
- Google Cloud project with Vertex AI enabled and authentication configured (`gcloud auth login` + `gcloud config set project <PROJECT_ID>`).

Required Python packages (example):

```
google-cloud-aiplatform
google-cloud-bigquery
google-cloud-storage
python-dotenv
pandas
tensorflow  # if training locally or using TensorBoard
```

Install with pip in an activated environment:

```bash
pip install -r requirements.txt
```

(If `requirements.txt` is not present, install the packages above manually.)

## Quickstart

1. Prepare environment and authenticate to GCP.

```bash
# create + activate venv, e.g. using python -m venv .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# login to gcloud and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. Generate or verify training data (examples available):

```bash
python data_prep/data_generator.py
```

3. Run a pipeline (local or Vertex AI orchestration):

```bash
python scripts/pipeline_runner.py --config pipeline_config.json
```

4. View results in `PIPELINE_RESULTS/` (predictions, evaluation CSVs, and TensorBoard logs).

## Important Files and Folders

- `data_prep/` — data generation and preprocessing scripts.
- `scripts/` — orchestration helpers and runners like `pipeline_runner.py`, `run_evaluation_only.py`.
- `src/` — library code, constants, and `handler.py` for custom model serving.
- `PIPELINE_RESULTS/` — outputs from pipeline runs (predictions, evaluations, tensorboard logs).
- `chainlit_app/` — simple demo app and related utilities.
- `model-7151280801659748352/` — example model artifacts and training snapshots (large files may be omitted from git).

## Running Evaluation Only

To run only evaluation on existing predictions or a model, use:

```bash
python scripts/run_evaluation_only.py --predictions PATH --evaluation-config config.json
```

## Registering a Model with a Custom Handler

There are helper scripts to register models with Vertex AI using a custom handler implementation. See `scripts/register_model_with_custom_handler.py` and `Sessions/register_model_with_custom_handler_class.py` for examples.

## Development notes

- Virtual environment: repository references using `uv` in Project Setup examples. Any standard virtual environment workflow is compatible.
- Python version: 3.11.6 was used during development; newer 3.11.x releases should be compatible.

## How to Reproduce Experiments

1. Generate dataset via `data_prep/` scripts.
2. Configure pipeline parameters (JSON config files in repo root or `Project Setup/`).
3. Launch `scripts/pipeline_runner.py` to run training/evaluation on Vertex AI or locally.
4. Inspect outputs under `PIPELINE_RESULTS/` and TensorBoard logs for metrics.

## Contributing

Contributions are welcome. Please open issues or pull requests describing the improvement and include reproducible steps.

## Contact

For questions about the project or reproducing results, open a GitHub issue in this repository.

---
© Course project
