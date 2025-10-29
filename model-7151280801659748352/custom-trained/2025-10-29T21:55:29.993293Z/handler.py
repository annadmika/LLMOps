"""Handler for Hugging Face model inference requests.

This handler attempts to load the tokenizer and model from the mounted
artifact directory (default /opt/huggingface/model) which is how Vertex AI
serving places model files. If the artifact directory is empty it will fall
back to a known model ID.
"""

from typing import Any, Dict, List
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/opt/huggingface/model"
DEFAULT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


class EndpointHandler:
    """Handler for processing inference requests using a Hugging Face model.

    The handler loads tokenizer and model from `model_dir` when present. It
    uses GPU if available and falls back to CPU otherwise. This avoids hard
    coding device names like `cuda:0` which may not exist in the serving
    environment.
    """

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        """Load tokenizer and model from the specified directory or fallback.

        Inputs:
        - model_dir: path where Vertex mounts the model artifact.
        """
        # Prefer files in the model artifact directory if present
        model_source = model_dir if os.path.isdir(model_dir) and os.listdir(model_dir) else os.environ.get("HF_MODEL_ID", DEFAULT_MODEL_ID)

        # Load tokenizer from the artifact or model hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)

        # Load model with device-aware settings
        if torch.cuda.is_available():
            # Use automatic device mapping when GPU is available
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()
        else:
            # CPU-only: load in 32-bit and keep on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to("cpu").eval()

    def generate(self, prompt: str, skip_special_tokens: bool = False, **kwargs: Any) -> str:
        """Generate text based on the input prompt."""
        tokenized_input = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )

        # Move inputs to model device
        device = next(self.model.parameters()).device
        tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

        generation_output = self.model.generate(
            **tokenized_input,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        return self.tokenizer.batch_decode(
            generation_output, skip_special_tokens=skip_special_tokens
        )[0]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Process inference requests.

        Expects a JSON payload with an `instances` list where each instance is a
        dict containing an `input` key with the prompt text. Optional
        `parameters` can be provided and will be forwarded to generate().
        """
        return {
            "predictions": [
                self.generate(instance["input"], **data.get("parameters", {}))
                for instance in data["instances"]
            ]
        }
