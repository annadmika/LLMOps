"""Chainlit app integrating a custom LLM chat model API."""
# Ensure repo root is on sys.path so 'src' imports resolve when running the script directly.
import sys
from pathlib import Path

# main.py is at src/app/main.py, so we need to go up 2 levels to reach repo root
repo_root = Path(__file__).resolve().parents[2]  # Changed from parents[1]
sys.path.insert(0, str(repo_root))

import re
import subprocess

import chainlit as cl
import requests
from chainlit.message import Message
from transformers import AutoTokenizer

from src.constants import ENDPOINT_ID, PROJECT_NUMBER

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_URL = f"https://europe-west2-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/europe-west2/endpoints/{ENDPOINT_ID}:predict"


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    await cl.Message(content="🤖 Yoda chatbot ready, I am. Ask me anything, you may!").send()


@cl.set_starters  # type: ignore
async def set_starters():
    """Set starter messages for the Chainlit app."""
    return [
        cl.Starter(
            label="Message #1 - Lightsaber talk",
            message="Paint the handle of your lightsaber on your hip dull gray.",
        ),
        cl.Starter(
            label="Message #2 - Not very nice",
            message="He is not very nice.",
        ),
        cl.Starter(
            label="Message #3 - Motivational quote",
            message="You must believe in the force.",
        ),
    ]


@cl.on_message
async def handle_message(message: Message):
    """Handle incoming messages from the user."""
    try:
        response = call_model_api(message)
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


def build_prompt(tokenizer: AutoTokenizer, sentence: str):
    """Build a prompt from a sentence applying the chat template."""
    return tokenizer.apply_chat_template(  # type: ignore
        [
            {"role": "user", "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_response(generated_text: str) -> str:
    """Extract the model's response from the generated text."""
    matches = re.findall(
        r"(?:<\|assistant\|>)([^<]*)",
        generated_text,
    )
    if matches:
        return matches[0].strip()
    # Fallback: return the text as-is if pattern doesn't match
    return generated_text.strip()


def call_model_api(message: Message) -> str:
    """Call the custom LLM chat model API."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

    access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()

    templated_input = build_prompt(tokenizer, message.content)
    model_input = {
        "instances": [{"input": templated_input}],
        "parameters": {
            "max_new_tokens": 64,
            "temperature": 0.1,
            "top_p": 0.8,
        },
    }
    
    # Make request
    http_response = requests.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=model_input,
        timeout=60,
    )
    
    # Check for HTTP errors
    http_response.raise_for_status()
    
    # Parse JSON response
    response_data = http_response.json()
    
    # Debug: print the actual response structure
    print(f"API Response: {response_data}")
    
    # Handle different response formats
    if "predictions" in response_data:
        raw_model_response = response_data["predictions"][0]
        # Check if it's a dict with "output" key or just a string
        if isinstance(raw_model_response, dict):
            raw_text = raw_model_response.get("output", str(raw_model_response))
        else:
            raw_text = raw_model_response
    elif "error" in response_data:
        raise Exception(f"API Error: {response_data['error']}")
    else:
        raise Exception(f"Unexpected response format: {response_data}")

    extracted_response = extract_response(raw_text)
    return extracted_response