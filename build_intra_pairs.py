"""Build a Dr.ECI input file from agentere's MAVEN-ERE run.

Produces every event-mention pair whose two mentions are in the SAME sentence,
across the 50 documents agentere evaluated (rows 0-49 of the test split), in the
Dr.ECI JSONL schema (single-sentence window). Each record also carries
`agentere_pred` (1 if agentere predicted causal for that pair, else 0) so the two
systems can be compared head-to-head on identical pairs.

Run in agentere's environment (needs `datasets` + the HF cache):
    cd /home/jovyan/project/temp/agentere && uv run python build_intra_pairs.py
"""
import json
from datasets import load_dataset

RUN = "/home/jovyan/project/temp/agentere/logs/allatonce/run_2026-07-23T13-14-40.775667Z_98a84450.json"
OUT = "/home/jovyan/project/litt/Dr.ECI/data/MAVEN-ERE/agentere_intra_pairs.jsonl"
N_DOCS = 50  # agentere used head(50) of the test split, deterministic

run = json.load(open(RUN))
preds = run["results"]["per_pair_predictions"]

# agentere's POSITIVE predictions, keyed by (doc_id, unordered mention-id pair).
# Anything not here = agentere predicted norel (non-causal).
agentere_pos = {
    (p["id"], frozenset(p["pair"].split(",")))
    for p in preds if p["pred"] in ("causes", "causedby")
}

ds = load_dataset("Nofing/MAVEN-ERE-standard-eci", split="test")


def sent_of(tok_idx, sentences):
    for si, (s, e) in enumerate(sentences):
        if s <= tok_idx < e:
            return si
    return -1


out = []
for i in range(N_DOCS):
    row = ds[i]
    doc_id, tokens = row["id"], row["tokens"]
    mentions, spans = row["mentions"], row["spans"]
    sentences, rel = row["sentences"], row["relations"]

    gold = {frozenset(pr) for pr in (rel.get("causes", []) + rel.get("causedby", []))}  # index pairs
    msent = [sent_of(spans[m][0], sentences) for m in range(len(mentions))]

    for a in range(len(mentions)):
        for b in range(a + 1, len(mentions)):
            if msent[a] == -1 or msent[a] != msent[b]:
                continue  # keep only same-sentence pairs
            si = msent[a]
            s, e = sentences[si]
            src, tgt = (a, b) if spans[a][0] <= spans[b][0] else (b, a)  # earlier token = source
            out.append({
                "item_id": len(out),
                "doc_name": doc_id,
                "doc_idx": i,
                "sent_in_doc_id": si,
                "pair": f"{mentions[src]},{mentions[tgt]}",
                "words": tokens[s:e],  # single-sentence window
                "events": [[t - s for t in spans[src]], [t - s for t in spans[tgt]]],  # sentence-local
                "bi_causal_label": 1 if frozenset((src, tgt)) in gold else 0,
                "tri_causal_label": 1 if frozenset((src, tgt)) in gold else 0,
                "events_id": [doc_id, mentions[src], mentions[tgt]],  # unique join key
                "agentere_pred": 1 if (doc_id, frozenset((mentions[src], mentions[tgt]))) in agentere_pos else 0,
                "sentence_relation": "intra",
            })

with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

pos = sum(o["bi_causal_label"] for o in out)
ag_pos = sum(o["agentere_pred"] for o in out)
print(f"{len(out)} intra-sentence pairs ({pos} causal / {len(out) - pos} non-causal) -> {OUT}")
print(f"agentere predicted causal on {ag_pos} of them")
print(f"stage-1 LLM calls if you run all: {len(out) * 3}")
