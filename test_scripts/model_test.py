import os
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from core.model import build_chat_graph  # or wherever you put it


# --- LangSmith tracing ---
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "Toto")
# optional: os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# -----------------------------
# Tool definition
# -----------------------------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


# -----------------------------
# Build graph (LLM + tools)
# -----------------------------
graph, invoke, ainvoke = build_chat_graph(
    model_id="deepseek/deepseek-v3.2",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    tools=[add_numbers],
)

# -----------------------------
# Run
# -----------------------------
result = invoke(
    [HumanMessage(content="What is 34 plus 15 plus 3 plus 234 plus 93 ? Use the tool.")],
    config={"run_name": "toto"},
)

for msg in result["messages"]:
    print(msg)
