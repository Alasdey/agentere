# core/graph.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Callable

import mlflow

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver


def default_should_continue(state: MessagesState) -> str:
    """Route to tools if the last assistant message contains tool calls."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_chat_graph(
    *,
    model_id: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    enable_tools: bool = True,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    should_continue: Optional[Callable[[MessagesState], str]] = None,
    recursion_limit: int = 100,
    run_name_model: str = "chat_model",
    run_name_model_with_tools: str = "chat_model+tools",
    run_name_tool_node: str = "tool_node",
    **llm_kwargs,
):
    """
    Build a minimal LangGraph chat loop:

        START -> call_model -> (tools -> call_model)* -> END

    This function:
      - Creates the LLM
      - Optionally binds tools
      - Builds the graph
      - Returns (graph, invoke, ainvoke)

    No assumptions about task, prompts, datasets, or evaluation.
    """

    # --- LLM creation (integrated) ---

    base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment or arguments.")

    llm: BaseChatModel = ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        **llm_kwargs,
    )

    # --- Tool handling ---

    tools = list(tools or [])
    tools_active = enable_tools and bool(tools)
    should_continue = should_continue or default_should_continue

    if tools_active:
        llm_to_use = llm.bind_tools(tools).with_config({"run_name": run_name_model_with_tools})
    else:
        llm_to_use = llm.with_config({"run_name": run_name_model})

    # --- Node ---

    @mlflow.trace(name="call_model")
    def call_model(state: MessagesState, *, config: Optional[Dict[str, Any]] = None):
        messages: List[AnyMessage] = state["messages"]
        ai = llm_to_use.invoke(messages, config=config)
        return {"messages": [ai]}

    # --- Graph ---

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)

    if tools_active:
        builder.add_node("tools", ToolNode(tools).with_config({"run_name": run_name_tool_node}))

    builder.add_edge(START, "call_model")

    if tools_active:
        builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        builder.add_edge("tools", "call_model")
    else:
        builder.add_edge("call_model", END)

    graph = builder.compile(checkpointer=checkpointer)

    # --- Convenience wrapper ---

    @mlflow.trace(name="graph_ainvoke")
    async def ainvoke(messages: Sequence[AnyMessage], *, config: Optional[Dict[str, Any]] = None) -> MessagesState:
        cfg = dict(config or {})
        cfg.setdefault("recursion_limit", recursion_limit)
        return await graph.ainvoke({"messages": list(messages)}, config=cfg)

    return graph, ainvoke
