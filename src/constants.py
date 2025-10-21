"""Main constants of the project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present in repo root
load_dotenv()

# Project root
PROJECT_ROOT_PATH = Path(__file__).parents[1]

# GCP configuration
PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID")
REGION: str = os.getenv("GCP_REGION", "europe-west2")
BUCKET_NAME: str | None = os.getenv("GCP_BUCKET_NAME")
ENDPOINT_ID: str | None = os.getenv("GCP_ENDPOINT_ID")
PROJECT_NUMBER: str | None = os.getenv("GCP_PROJECT_NUMBER")

# Paths
RAW_DATASET_URI: str = f"gs://{BUCKET_NAME}/yoda_sentences.csv"
PIPELINE_ROOT_PATH: str = f"{BUCKET_NAME}/vertexai-pipeline-root/"
