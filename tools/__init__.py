
# tools/__init__.py
from .coherence import coherence
from .counterfactual import counterfactual_check
from .placeholder import placeholder
from .encoder import encoder
from .bare_causes import bare_causes
from .eci import eci

TOOL_REGISTRY = {
    "coherence": coherence,
    "placeholder": placeholder,
    "counterfactual_check": counterfactual_check,
    "encoder": encoder,
    "bare_causes": bare_causes,
    "eci": eci,
}

def get_enabled_tools(tool_names: list):
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]
