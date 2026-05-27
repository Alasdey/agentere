from langchain_core.tools import tool

from utils.context import CURRENT_USER_PROMPT


@tool
async def reprompt(comment: str = "") -> str:
    """
    Returns the original task prompt (text + pairs to classify) exactly as it was given.
    Call this to refocus on the task after a long reasoning chain or tool sequence.
    """
    return CURRENT_USER_PROMPT.get()
