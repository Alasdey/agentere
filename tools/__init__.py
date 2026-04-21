
# tools/__init__.py
from .coherence import coherence
from .counterfactual import counterfactual_check
from .encoder import encoder
from .bare_causes import bare_causes
from .eci import eci
from .few_shot import few_shot_examples
from .placeholder import placeholder

TOOL_REGISTRY = {
    "coherence": coherence,
    "counterfactual_check": counterfactual_check,
    "encoder": encoder,
    "bare_causes": bare_causes,
    "eci": eci,
    "few_shot_examples": few_shot_examples,
    "placeholder": placeholder,
}

def get_enabled_tools(tool_names: list):
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]
