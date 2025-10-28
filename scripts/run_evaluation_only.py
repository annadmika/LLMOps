# run_evaluation_only.py
from google.cloud import aiplatform

PROJECT_ID = "llm-ops-475209"
REGION = "europe-west2"
PIPELINE_ROOT = "gs://anna-pavel-bucket/pipeline_root"
PREDICTIONS_URI = "gs://anna-pavel-bucket/vertexai-pipeline-root/54825872111/anna-pavel-project-pipeline-20251026003411/inference-component_4846667871079628800/predictions"  # ← paste from Step 1

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name="tarot-evaluation-only-run",
    template_path="evaluation_only_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"predictions_uri": PREDICTIONS_URI},
)
job.submit()

print("🚀 Submitted evaluation-only pipeline job.")
