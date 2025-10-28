from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"
)

trace = langfuse.trace(name="connection_test", input={"test": "ok"})
trace.generation(name="test_gen", input={"x": 1}, output={"y": 2})
trace.update(output={"done": True})
langfuse.flush()
print("✅ done")
