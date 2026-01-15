# tools/__init__.py
from .coherence import coherence_check
from .placeholder import placeholder

TOOL_REGISTRY = {
    "coherence_check": coherence_check,
    "placeholder": placeholder,
}

def get_enabled_tools(tool_names: list):
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]