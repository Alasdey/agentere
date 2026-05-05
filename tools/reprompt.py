from contextvars import ContextVar
from langchain_core.tools import tool

CURRENT_USER_PROMPT: ContextVar[str] = ContextVar("current_user_prompt", default="")


@tool
async def reprompt(comment: str = "") -> str:
    """
    Returns the original task prompt (text + pairs to classify) exactly as it was given.
    Call this to refocus on the task after a long reasoning chain or tool sequence.
    """
    return CURRENT_USER_PROMPT.get()
