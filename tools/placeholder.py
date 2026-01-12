from langchain_core.tools import tool

@tool
def placeholder(message: str = "No-op") -> str:
    """A no-op placeholder tool for testing tool routing."""
    return f"[PlaceholderTool] Called successfully. Message: {message}"
