
from typing import Any, Dict, List, Tuple, Optional

from langchain_core.tools import tool
from langsmith.run_helpers import traceable

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

@tool
def coherence_check(
    *,
    pairs: List[Dict[str, str]],
    rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    General coherence checker for arbitrary relation labels.

    pairs: [{"pair":"A,B","label":"REL"}, ...]   # directional A->B
    rules: list of rules in one of these forms:

      Two-mention rule:
        {"type":"two", "if": {"rel":"R1"}, "then":{"rel":"R2", "dir":"12"}, "negated": False}

      Three-mention rule:
        {"type":"three",
         "if1":{"rel":"R1"}, "if2":{"rel":"R2"},
         "then":{"rel":"R3", "dir":"13"}, "negated": False}
    """

    # --- parse predictions ---
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

    # --- check rules ---
    for ridx, r in enumerate(rules or []):
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
                    if ok:
                        continue

                    kind = "missing" if found is None else "conflict"
                    violations.append({
                        "rule_index": ridx,
                        "type": "two",
                        "bindings": {"m1": m1, "m2": m2},
                        "expected": {"pair": f"{a},{b}", "label": r2, "negated": neg},
                        "found": found if found is not None else "<missing>",
                        "kind": kind,
                    })

        elif rtype == "three":
            r1 = r["if1"]["rel"]
            r2 = r["if2"]["rel"]
            r3 = r["then"]["rel"]
            d3 = r["then"].get("dir", "13")

            for m1 in mentions:
                for m2 in mentions:
                    if m1 == m2:
                        continue
                    for m3 in mentions:
                        if m3 == m1 or m3 == m2:
                            continue

                        if get(m1, m2) != r1:
                            continue
                        if get(m2, m3) != r2:
                            continue

                        triggered += 1
                        a, b = pick(m1, m2, m3, d3)
                        found = get(a, b)

                        ok = (found == r3) if not neg else (found is not None and found != r3)
                        if ok:
                            continue

                        kind = "missing" if found is None else "conflict"
                        violations.append({
                            "rule_index": ridx,
                            "type": "three",
                            "bindings": {"m1": m1, "m2": m2, "m3": m3},
                            "expected": {"pair": f"{a},{b}", "label": r3, "negated": neg},
                            "found": found if found is not None else "<missing>",
                            "kind": kind,
                        })

    coherence_rate = (1.0 - len(violations) / triggered) if triggered else 1.0

    return {
        "triggered": triggered,
        "violations": violations,
        "coherence_rate": coherence_rate,
    }
