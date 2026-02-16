
# tools/__init__.py
from .coherence import coherence_check
from .counterfactual import counterfactual_check
from .placeholder import placeholder
from .encoder_predictions import encoder_predictions
from .bare_causes import bare_causes
from .eci import eci

TOOL_REGISTRY = {
    "coherence_check": coherence_check,
    "placeholder": placeholder,
    "counterfactual_check": counterfactual_check,
    "encoder_predictions": encoder_predictions,
    "bare_causes": bare_causes,
    "eci": eci,
}

def get_enabled_tools(tool_names: list):
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]
