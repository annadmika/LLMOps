"""Script to register a model with a custom handler."""

from pathlib import Path
import sys
import os

# Ensure repository root is on sys.path so `from src import ...` works when running
# this script directly (without installing the package). We compute the project
# root as the parent of the `scripts/` directory.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import typer
from google.cloud import aiplatform, storage

from src.constants import BUCKET_NAME, PROJECT_ID, PROJECT_ROOT_PATH, REGION

HANDLER_PATH = PROJECT_ROOT_PATH / "src" / "handler.py"


def register_model_with_custom_handler(
    model_uri: str = typer.Option(..., "--model-uri", "-m", help="GCS model artifact URI (gs://... or console URL)"),
    display_name: str = typer.Option(..., "--display-name", "-d", help="Display name for the Vertex Model resource"),
    parent_model: str | None = typer.Option(None, "--parent-model", help="Optional parent model resource name"),
    serving_container_image_uri: str = typer.Option(
        "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311",
        "--serving-container-image-uri",
        help="Serving container image URI",
    ),
    handler_path: Path = HANDLER_PATH,
):
    """Registers a model with a custom handler in Vertex AI."""
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Normalize model_uri: accept gs:// paths or console URLs and convert to bucket/object
    # Example console URL:
    # https://console.cloud.google.com/storage/browser/BUCKET_NAME/path/to/artifact
    if model_uri.startswith("gs://"):
        # strip gs://bucket/ -> bucket and object
        _, rest = model_uri.split("gs://", 1)
        parts = rest.split("/", 1)
        bucket = parts[0]
        object_path = parts[1] if len(parts) > 1 else ""
    elif "storage/browser" in model_uri:
        # extract after storage/browser/
        try:
            idx = model_uri.index("storage/browser/") + len("storage/browser/")
            rest = model_uri[idx:]
            parts = rest.split("/", 1)
            bucket = parts[0]
            object_path = parts[1] if len(parts) > 1 else ""
        except ValueError:
            raise ValueError("Could not parse model_uri; please provide a gs:// path or a Cloud Console storage URL")
    else:
        # assume it's a plain gcs path without scheme
        if model_uri.count("/") >= 1:
            parts = model_uri.split("/", 1)
            bucket = parts[0]
            object_path = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError("Unrecognized model_uri format: provide a gs:// path or console storage URL")

    # Upload the custom handler to the same artifact folder in GCS so the serving container can find it
    blob_path = f"{object_path.rstrip('/')}/handler.py" if object_path else "handler.py"
    client = storage.Client()
    full_gs_path = f"gs://{bucket}/{blob_path}"
    # Helpful logging so you can confirm the resolved artifact location before/after upload
    print(f"[register_model] Resolved bucket={bucket!r}, object_path={object_path!r}")
    print(f"[register_model] Uploading handler.py -> {full_gs_path}")
    client.bucket(bucket).blob(blob_path).upload_from_filename(str(handler_path))
    print(f"[register_model] Uploaded handler.py -> {full_gs_path}")

    # Register the model with the custom handler
    aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_ports=[8080],
        parent_model=parent_model,
    )


if __name__ == "__main__":
    typer.run(register_model_with_custom_handler)
