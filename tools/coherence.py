# tools/coherence.py
from typing import Any, Dict, List, Tuple, Optional
import os
import yaml

from langchain_core.tools import tool

# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_coherence_rules() -> List[Dict[str, Any]]:
    """
    Loads coherence rules from the config.yaml file located in the project root.
    """
    # Assuming config.yaml is in the parent directory of the 'tools' folder
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config.get("coherence_rules", [])

# =============================================================================
# TOOL
# =============================================================================

@tool
def coherence_check(
    pairs: List[Dict[str, str]],
) -> str:
    """
    Analyzes the logical coherence of extracted relations.

    Rules are automatically loaded from 'config.yaml'. 
    This tool detects violations (e.g., missing transitive links, conflicting directions).

    Args:
        pairs: A list of relations in the format [{"pair": "ID1,ID2", "label": "REL"}, ...].
               The pair string implies direction ID1 -> ID2.

    Returns:
        A human-readable summary of any incoherence found.
    """
    # --- Load Rules ---
    try:
        rules = load_coherence_rules()
    except Exception as e:
        return f"Error loading configuration: {str(e)}"

    if not rules:
        return "No rules configured in config.yaml. Cannot perform check."

    # --- Parse predictions ---
    rel: Dict[Tuple[str, str], str] = {}
    mentions = set()

    for o in pairs or []:
        p = (o.get("pair") or "").strip()
        if "," not in p:
            continue
        a, b = [x.strip() for x in p.split(",", 1)]
        lab = (o.get("label") or "").strip()
        if not a or not b or not lab:
            continue
        rel[(a, b)] = lab
        mentions.add(a)
        mentions.add(b)

    mentions = list(mentions)

    def get(a: str, b: str) -> Optional[str]:
        return rel.get((a, b))

    def pick(m1: str, m2: str, m3: Optional[str], d: str) -> Tuple[str, str]:
        if d == "12": return m1, m2
        if d == "21": return m2, m1
        if d == "13": return m1, m3
        if d == "31": return m3, m1
        if d == "23": return m2, m3
        if d == "32": return m3, m2
        raise ValueError(f"Bad dir: {d}")

    violations: List[Dict[str, Any]] = []
    triggered = 0

    # --- Check rules ---
    for ridx, r in enumerate(rules):
        rtype = (r.get("type") or "").lower()
        neg = bool(r.get("negated", False))

        if rtype == "two":
            r1 = r["if"]["rel"]
            r2 = r["then"]["rel"]
            d2 = r["then"].get("dir", "12")

            for m1 in mentions:
                for m2 in mentions:
                    if m1 == m2:
                        continue
                    if get(m1, m2) != r1:
                        continue

                    triggered += 1
                    a, b = pick(m1, m2, None, d2)
                    found = get(a, b)

                    ok = (found == r2) if not neg else (found is not None and found != r2)
                    if not ok:
                        violations.append({
                            "rule_index": ridx,
                            "desc": f"Two-mention rule: If '{m1}' to '{m2}' is '{r1}', then '{a}' to '{b}' should be '{r2}'.",
                            "found": found if found is not None else "<missing>",
                        })

        elif rtype == "three":
            r1 = r["if1"]["rel"]
            r2 = r["if2"]["rel"]
            r3 = r["then"]["rel"]
            d3 = r["then"].get("dir", "13")

            for m1 in mentions:
                for m2 in mentions:
                    if m1 == m2: continue
                    for m3 in mentions:
                        if m3 == m1 or m3 == m2: continue

                        if get(m1, m2) != r1: continue
                        if get(m2, m3) != r2: continue

                        triggered += 1
                        a, b = pick(m1, m2, m3, d3)
                        found = get(a, b)

                        ok = (found == r3) if not neg else (found is not None and found != r3)
                        if not ok:
                            violations.append({
                                "rule_index": ridx,
                                "desc": f"Three-mention rule (Transitivity): If '{m1}'->'{m2}' is '{r1}' and '{m2}'->'{m3}' is '{r2}', then '{a}'->'{b}' should be '{r3}'.",
                                "found": found if found is not None else "<missing>",
                            })

    # --- Generate Human Summary ---
    if triggered == 0:
        return "No relations matched the rule premises, so no coherence checks were triggered."

    if not violations:
        return f"Checked {triggered} potential rule applications. No incoherence found; the graph is logically consistent."

    summary_lines = [
        f"I found **{len(violations)} incoherence violations** out of {triggered} checks.\n",
        "Here are the specific issues detected:\n"
    ]

    for v in violations:
        summary_lines.append(f"- **Violation:** {v['desc']}")
        summary_lines.append(f"  - **Expected:** Relation between the indicated mentions.")
        summary_lines.append(f"  - **Found:** {v['found']}\n")

    return "\n".join(summary_lines)