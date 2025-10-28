"""🌙 Tarot Oracle — Chainlit app connected to your fine-tuned Phi-3-mini model on Vertex AI, with safe intent guard & optional Gemini Judge."""

import os
import re
import random
import asyncio
import subprocess
import requests
import chainlit as cl

from chainlit.message import Message
from transformers import AutoTokenizer

# ---- Optional dependencies: degrade gracefully if missing ----
INTENT_EMBEDDER = None
UTIL = None
try:
    from sentence_transformers import SentenceTransformer, util
    INTENT_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    UTIL = util
except Exception:
    INTENT_EMBEDDER = None
    UTIL = None

# ---- Optional Gemini Judge (skips if no API key or SDK) ----
GEMINI_OK = False
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_OK = True
except Exception:
    GEMINI_OK = False


# ======================================================
# CONFIGURATION
# ======================================================
MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_ID = "2842752129242759168"  # your deployed endpoint
PROJECT_NUMBER = "llm-ops-475209"
REGION = "europe-west2"

ENDPOINT_URL = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/projects/"
    f"{PROJECT_NUMBER}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
)

DEFAULT_GEN = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

SYSTEM_PROMPT = (
    "You are The Oracle — a mystical tarot reader. "
    "Respond directly to the querent's question and the cards drawn. "
    "Do not invent new questions or rephrase theirs. "
    "Your tone is poetic, compassionate, and insightful. "
    "Interpret the cards in a Situation–Action–Outcome flow when possible."
)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

THINKING_MESSAGES = [
    "🔮 *The oracle is reading your fortune...*",
    "🌙 *The cards are revealing their secrets...*",
    "✨ *The winds of fate are whispering to me softly...*",
    "🕯️ *The veil between worlds is growing thin...*",
    "🌌 *The stars are aligning to reveal your path...*",
]

# ======================================================
# INTENT DETECTION
# ======================================================
TAROT_PROMPTS = [
    "What will happen in my relationship? I pulled the cards...",
    "What should I focus on in the coming months?",
    "I pulled the cards ... what does it mean?",
    "Can you interpret my tarot spread?",
    "I drew these cards: ... what does this mean?",
]
KEYWORDS = {"card", "cards", "tarot", "spread", "pulled", "drew", "reading", "upright", "reversed"}

def is_tarot_query(user_input: str, threshold: float = 0.45) -> bool:
    text = user_input.lower()
    # quick lexical check
    if any(k in text for k in KEYWORDS):
        return True
    # semantic check if available
    if INTENT_EMBEDDER and UTIL:
        q = INTENT_EMBEDDER.encode(user_input, convert_to_tensor=True)
        refs = INTENT_EMBEDDER.encode(TAROT_PROMPTS, convert_to_tensor=True)
        sim = UTIL.cos_sim(q, refs)
        max_sim = float(sim.max().item())
        return max_sim >= threshold
    # fallback: be permissive if no embedder
    return True


# ======================================================
# STARTUP
# ======================================================
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="General Guidance", message="What should I focus on in the coming months?"),
        cl.Starter(label="Career Question", message="What will happen if I change jobs this year?"),
        cl.Starter(
            label="Relationship Reading",
            message="What will happen in my relationship? I pulled the cards 7 of cups upright, the tower reversed, and the 3 of wands.",
        ),
    ]

@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [])
    cl.user_session.set("gen", DEFAULT_GEN.copy())
    intro = (
        "🌙 *The Oracle awakens...*\n\n"
        "Ask your question, and tell me the cards you have drawn... "
        "The mists of fate are listening. 🔮"
    )
    await cl.Message(content=intro, author="The Oracle").send()


# ======================================================
# MAIN MESSAGE HANDLER
# ======================================================
@cl.on_message
async def handle_message(message: Message):
    user_text = message.content.strip()

    # Guardrail: require tarot-like input
    if not is_tarot_query(user_text):
        await cl.Message(
            author="The Oracle",
            content="✨ I'm sorry, dear one — I’m here to read your tarot spread. Please share your question **and** the cards you have drawn (e.g., “I pulled The Tower reversed, 7 of Cups upright, and 3 of Wands”)."
        ).send()
        return

    # Thinking animation
    thinking_text = random.choice(THINKING_MESSAGES)
    thinking = cl.Message(
        content=f"<span class='oracle-thinking'>{thinking_text}</span>",
        author="The Oracle"
    )
    await thinking.send()

    try:
        # run model call off-thread
        model_reply = await asyncio.to_thread(call_model_api, user_text)
        await thinking.remove()

        # stream oracle reply
        oracle_msg = cl.Message(content="", author="The Oracle")
        await oracle_msg.send()

        paragraphs = model_reply.split("\n\n")
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            for word in paragraph.split():
                oracle_msg.content += word + " "
                await oracle_msg.update()
                # natural pacing
                if word.endswith((",", ";")):
                    await asyncio.sleep(random.uniform(0.12, 0.2))
                elif word.endswith((".", "!", "?")):
                    await asyncio.sleep(random.uniform(0.25, 0.45))
                elif "…" in word or "..." in word:
                    await asyncio.sleep(random.uniform(0.35, 0.55))
                else:
                    await asyncio.sleep(random.uniform(0.03, 0.06))
            oracle_msg.content += "\n\n"
            await oracle_msg.update()
            await asyncio.sleep(random.uniform(0.45, 0.85))

        await oracle_msg.update()

        # Judge (optional)
        if GEMINI_OK:
            judge_feedback = await ask_gemini_judge(user_text, model_reply)
            await cl.Message(content=f"🧠 **Gemini Judge Verdict**\n{judge_feedback}", author="LLM Judge").send()
        else:
            # silently skip if Gemini not configured
            pass

    except Exception as e:
        await thinking.remove()
        await cl.Message(content=f"⚠️ Error:\n{e}", author="The Oracle").send()


# ======================================================
# GEMINI JUDGE (optional, safe)
# ======================================================
async def ask_gemini_judge(user_input: str, model_output: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        You are a tarot reading evaluator.

        Evaluate the following reading for:
        - Relevance to the user's question
        - Accuracy of interpretation
        - Empathy and mystical tone
        - Coherence and overall quality

        Return:
        Score: <number from 1–10>
        Feedback: <1–2 short sentences>

        ---
        User's question:
        {user_input}

        Oracle's response:
        {model_output}
        ---
        """
        resp = model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4, top_p=0.9, max_output_tokens=256
            ),
        )
        text = (resp.text or "").strip() or "No response from Gemini."
        return text
    except Exception as e:
        return f"⚠️ Gemini Judge Error: {e}"


# ======================================================
# MODEL CALL — Vertex AI endpoint
# ======================================================
def call_model_api(user_input: str) -> str:
    # get OAuth token
    access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()

    # build chat messages
    history = cl.user_session.get("history") or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    user_prompt = (
        "The querent asks for a tarot reading. "
        "Read the following question and respond directly as The Oracle.\n\n"
        f"User's question: {user_input}"
    )
    messages.append({"role": "user", "content": user_prompt})

    templated_input = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    payload = {
        "instances": [{"input": templated_input}],
        "parameters": DEFAULT_GEN,
    }

    resp = requests.post(
        ENDPOINT_URL,  # <-- keep as string (DO NOT .encode())
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=300,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        pred = data["predictions"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected response: {data}")

    if isinstance(pred, str):
        raw = pred
    elif isinstance(pred, dict):
        raw = (
            pred.get("generated_text")
            or pred.get("content")
            or pred.get("output_text")
            or pred.get("text")
            or ""
        )
    else:
        raw = str(pred)

    text = (extract_response(raw) or raw).strip()
    if not text:
        text = "✨ The cards whisper softly... but no words reach you this time."

    # maintain short history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": text})
    cl.user_session.set("history", history[-6:])

    return text


# ======================================================
# TEXT CLEANING
# ======================================================
def extract_response(generated_text: str) -> str:
    if "<|assistant|>" in generated_text:
        matches = re.findall(r"(?:<\|assistant\|>)([^<]*)", generated_text)
        if matches:
            return matches[0]
    return generated_text
