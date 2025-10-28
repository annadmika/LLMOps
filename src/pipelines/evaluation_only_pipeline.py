import sys
from pathlib import Path

# Make sure 'components' is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kfp import compiler, dsl
from kfp.dsl import Dataset
from pipeline_components.evaluation_component import evaluation_component


@dsl.pipeline(name="tarot-evaluation-only")
def evaluation_only_pipeline(predictions_uri: str):
    """Re-runs the updated evaluation component on existing predictions artifact."""

    # Import the existing predictions as a Dataset artifact
    imported_predictions = dsl.importer(
        artifact_uri=predictions_uri,
        artifact_class=Dataset,
        reimport=True,
    )

    # ✅ Pass the artifact output, not the task object
    evaluation_component(predictions=imported_predictions.outputs["artifact"])


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=evaluation_only_pipeline,
        package_path="evaluation_only_pipeline.json",
    )
    print("✅  evaluation_only_pipeline.json compiled successfully.")
