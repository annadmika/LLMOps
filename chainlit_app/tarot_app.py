"""🌙 Tarot Oracle — Chainlit app connected to your fine-tuned Phi-3-mini model on Vertex AI with human feedback."""

import os
import re
import random
import asyncio
import subprocess
import requests
import chainlit as cl
from chainlit.message import Message
from transformers import AutoTokenizer
from dotenv import load_dotenv
from langfuse import Langfuse, observe  # Langfuse tracing

# ======================================================
# ENV + LANGFUSE INIT
# ======================================================
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    blocked_instrumentation_scopes=["chainlit"],  # ✅ prevents infinite recursion
)

# ======================================================
# MODEL + PROMPT CONFIG
# ======================================================
MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_ID = "2842752129242759168"
PROJECT_NUMBER = "llm-ops-475209"
REGION = "europe-west2"
ENDPOINT_URL = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/projects/"
    f"{PROJECT_NUMBER}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
)
DEFAULT_GEN = {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9, "do_sample": True}

SYSTEM_PROMPT = (
    "You are The Oracle — a mystical tarot reader. "
    "Respond directly to the querent's question and the cards drawn. "
    "Do not invent new questions or rephrase theirs. "
    "Your tone is poetic, compassionate, and insightful. "
    "Interpret the cards in a Situation–Action–Outcome flow when possible."
)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

# ======================================================
# FEEDBACK LOGGING HELPER
# ======================================================
def log_feedback(user_input: str, model_output: str, rating: int, comment: str | None = None):
    """Log human feedback to Langfuse."""
    try:
        langfuse.feedback(
            name="oracle_human_feedback",
            value=rating,
            comment=comment or "User feedback on Oracle response",
            metadata={
                "user_input": user_input,
                "oracle_output": model_output,
            },
        )
        print(f"🌟 Logged feedback ({rating}) to Langfuse successfully.")
    except Exception as e:
        print(f"⚠️ Could not log feedback: {e}")

# ======================================================
# STARTUP INTRO MESSAGE
# ======================================================
@cl.on_chat_start
async def on_chat_start():
    """Mystical introduction when the app launches."""
    cl.user_session.set("history", [])
    intro = (
        "🌙 *The Oracle awakens...*\n\n"
        "Ask your question, and tell me the cards you have drawn — "
        "The mists of fate are listening. 🔮"
    )
    await cl.Message(content=intro, author="The Oracle").send()

# ======================================================
# MAIN MESSAGE HANDLER
# ======================================================
@cl.on_message
async def handle_message(message: Message):
    """Handle user messages with automatic Langfuse tracing and human feedback."""
    user_text = message.content.strip()

    # 🌙 Display animated thinking message
    thinking_text = random.choice([
        "🔮 The oracle is reading your fortune...",
        "🌙 The cards are revealing their secrets...",
        "✨ The winds of fate are whispering softly...",
        "🕯️ The veil between worlds grows thin...",
        "🌌 The stars are aligning to reveal your path...",
    ])
    thinking = cl.Message(
        content=f"<span class='oracle-thinking'>{thinking_text}</span>",
        author="The Oracle"
    )
    await thinking.send()

    try:
        # 🔮 Call model (runs in a background thread)
        reply = await asyncio.to_thread(call_model_api, user_text)
        await thinking.remove()

        # ✨ Typing animation for Oracle's response
        oracle_msg = cl.Message(content="", author="The Oracle")
        await oracle_msg.send()
        for para in reply.split("\n\n"):
            if not para.strip():
                continue
            for word in para.split():
                oracle_msg.content += word + " "
                await oracle_msg.update()
                await asyncio.sleep(random.uniform(0.03, 0.07))
            oracle_msg.content += "\n\n"
            await oracle_msg.update()
            await asyncio.sleep(random.uniform(0.3, 0.7))

        # 🌟 Ask for user feedback
        feedback_actions = [
            cl.Action(name="rating_1", payload={"rating": 1}, label="💤 Confusing"),
            cl.Action(name="rating_2", payload={"rating": 2}, label="🌫️ Vague"),
            cl.Action(name="rating_3", payload={"rating": 3}, label="🌙 Decent"),
            cl.Action(name="rating_4", payload={"rating": 4}, label="🔮 Insightful"),
            cl.Action(name="rating_5", payload={"rating": 5}, label="🌟 Enlightening"),
        ]

        result = await cl.AskActionMessage(
            content="✨ How did this reading feel to you?",
            actions=feedback_actions,
            timeout=180,
        ).send()

        # In some Chainlit versions, result is a dict like {"rating": 3}
        if result and isinstance(result, dict) and "rating" in result:
            rating = int(result["rating"])
            await cl.Message(content=f"Thank you for your feedback! ({rating}/5)").send()
            print(f"🧭 Feedback received: {rating}")
            log_feedback(user_text, reply, rating)

        # In newer versions, result may be an Action object
        elif result and hasattr(result, "payload") and "rating" in result.payload:
            rating = int(result.payload["rating"])
            await cl.Message(content=f"Thank you for your feedback! ({rating}/5)").send()
            print(f"🧭 Feedback received: {rating}")
            log_feedback(user_text, reply, rating)

        else:
            print("⚠️ No feedback received or timeout.")




    except Exception as e:
        await thinking.remove()
        await cl.Message(content=f"⚠️ Error:\n{e}", author="The Oracle").send()

# ======================================================
# MODEL CALL 
# ======================================================
@observe(name="Tarot Reading Generation")
def call_model_api(user_input: str) -> str:
    """Send user input to the Vertex AI model with automatic Langfuse logging."""
    tokenizer = TOKENIZER
    access_token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()

    langfuse.update_current_span(input=user_input)

    with langfuse.start_as_current_generation(name="Vertex AI Tarot Generation") as gen:
        history = cl.user_session.get("history") or []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        messages.append({"role": "user", "content": user_input})

        templated_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        payload = {"instances": [{"input": templated_input}], "parameters": DEFAULT_GEN}

        response = requests.post(
            ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        ).json()

        raw = response["predictions"][0]
        text = extract_response(raw).strip() or "✨ The cards whisper softly..."

        gen.update(
            input=templated_input,
            output=text,
            model=response.get("modelDisplayName", "phi-3-mini"),
            metadata={"endpoint": ENDPOINT_ID, **DEFAULT_GEN},
        )

    langfuse.update_current_span(output={"oracle_response": text})
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": text})
    cl.user_session.set("history", history[-6:])
    return text

# ======================================================
# TEXT CLEANING
# ======================================================
def extract_response(generated_text: str) -> str:
    """Extract clean assistant response text from Phi-3 output."""
    match = re.search(r"(?:<\|assistant\|>)([^<]*)", generated_text)
    return match.group(1) if match else generated_text
