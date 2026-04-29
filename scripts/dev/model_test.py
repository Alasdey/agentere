import os
import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from tools.placeholder import placeholder
from model.model import build_chat_graph  # or wherever you put it


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
    tools=[add_numbers, placeholder],
)

# -----------------------------
# Run
# -----------------------------
prompt = "Use the tool with tool_call"

result = invoke(
    [HumanMessage(content="What is 34 plus 15 plus 3 plus 234 plus 93 ?"+prompt)],
    config={"run_name": "toto"},
)

for msg in result["messages"]:
    print(msg)


# -----------------------------
async def async_test():    
    prompts = [
        "What is 1 plus 2? Use the tool."+prompt,
        "What is 3 plus 4? Use the tool."+prompt,
        "What is 5 plus 6? Use the tool."+prompt,
        "What is 7 plus 8? Use the tool."+prompt,
    ]

    tasks = [
        ainvoke(
            [HumanMessage(content=p)],
            config={"run_name": f"openrouter-deepseekv3-tool-test-async-{i}"},
        )
        for i, p in enumerate(prompts)
    ]

    results = await asyncio.gather(*tasks)

    for i, res in enumerate(results):
        print(f"\n--- Async result {i} ---")
        for msg in res["messages"]:
            print(msg)


asyncio.run(async_test())