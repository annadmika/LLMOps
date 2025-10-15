import os
from google.cloud import aiplatform, storage
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

print(f"🔧 Project: {PROJECT_ID}")
print(f"🌍 Region: {REGION}")
print(f"🪣 Bucket: {BUCKET_NAME}")

# 2. Test Vertex AI client
try:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    print("✅ Vertex AI client initialized successfully.")
except Exception as e:
    print("❌ Vertex AI initialization failed:", e)

# 3. Test Google Cloud Storage client
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs())
    print(f"✅ Accessed bucket '{BUCKET_NAME}' successfully. Found {len(blobs)} files.")
except Exception as e:
    print("❌ Failed to access GCS bucket:", e)
