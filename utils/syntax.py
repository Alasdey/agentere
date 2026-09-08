"""spaCy-derived syntactic annotation, appended after the document text.

Driven by two orthogonal knobs under `syntax:` in config.yaml. Whenever either is set, every
prompt-construction site splices the rendered block in directly after the document text via
`doc_text_with_syntax`, so the same annotation appears in the few-shot demonstrations, in the
CoT-synthesis input and in the prediction prompt. No prompt YAML is edited.

  `level`     — "off" | "mentions" | "args" | "paths". Mention-centric and cumulative. Cost
                grows with the mention count, and "paths" is quadratic in mentions per
                sentence, which is what the size caps below exist to bound.
  `discourse` — any of ["skeleton", "participants"]. Document-linear: one row per sentence,
                one per entity. These are the only sections that describe the 47% of sentences
                holding no event mention (44% of all tokens), and "participants" is the only
                signal anywhere in the block that crosses a sentence boundary — the dependency
                paths of `level: paths` are clamped to a single sentence by construction.

The two are independent so an experiment can attribute a change to whole-text context or to the
per-mention ladder rather than to both at once; `discourse` alone with `level: off` is valid.

Why the parser never sees doc_text
----------------------------------
`doc_text` is the dataset's `tokens` joined by single spaces with `<ID surface>` markup wrapped
around each mention span (verified: an exact rebuild on 209/233 EventStoryLine, 183/183
Causal-TimeBank and 200/200 MECI documents). Feeding that to spaCy would make the parser
tokenise `<`, the id digits and `>` as words and wreck every attachment. So the Doc is built
straight from `tokens` with `sentences` as `sent_starts`, which gives

  * exact 1:1 token alignment, so `spans` maps a mention id to its precise tokens, and
  * the dataset's own sentence segmentation — the same one `doc["mention_sentence"]` and
    `few_shot.intra_only` use. Verified on 40 ESL docs: 0 token and 0 sentence mismatches.

Mention surfaces come from `doc["mentions_map"]`, not from the markup in `doc_text`. For the 30
discontinuous ESL spans the two disagree — `doc_text` marks only the first token (`<14 pulled>`)
while `mentions_map` joins the whole span (`"pulled on"`) — and `mentions_map` is what
`utils/formatting.format_pair_lines` shows the model, so the block agrees with the pair list.

English only
------------
`en_core_web_sm` is the only pipeline loaded. Documents in any other language (MECI's
`causal-da/es/tr/ur`) get no block rather than a nonsense parse. The label set is therefore
ClearNLP/OntoNotes, **not** Universal Dependencies: `dobj`/`prep`/`pobj`/`nsubjpass`, with no
`obj` and no `obl` (verified against the installed model). English also never populates
`token.morph["Voice"]` nor `token.morph["Polarity"]`, so the passive is derived from
`nsubjpass`/`auxpass` children and negation from a `neg` child. Negation is worth the trouble
despite appearing on only 1% of mentions: "the assistance is *not* helping" inverts the causal
claim, and a modal auxiliary ("could have caused") is surfaced for the same reason.

Where the block is produced
---------------------------
`annotate_docs` is a post-pass over an already-materialised doc list, called at exactly two
places:

  * `main.py`, right after the test docs are listed — before `pregenerate_cot` and inference.
  * `tools/few_shot.py::_load_train_split`, at both of its returns, after the `pool_size`
    truncation so only the pool is parsed rather than the whole train split.

It is deliberately not inside `load_hf_dataset_parsed`: the analysis scripts
(`scripts/analysis/show_traces.py`, `vote_threshold.py`) call that loader too and must keep
joining runs on pristine text.

`doc["doc_text"]` is never mutated. That is load-bearing — it is also the TF-IDF / sentence-
transformer few-shot similarity query and pool (`tools/few_shot.py`), the causal-graph node
label (`utils/causal_graph.py`) and the join key in the analysis scripts.

Not covered, by design: the standalone `scripts/experiments/*` prompts (llm_perpair.py,
simple_eval.py, autoprompt.py) build their own prompts and never see a block.

No `register_reset` is needed: the pipeline depends only on module constants, and the rendered
blocks live on doc dicts that are rebuilt every run (`few_shot._reset_caches` nulls the train
cache on `set_cfg`). Add one the day the model becomes config-driven.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from utils.runtime_config import get_cfg

# The sentinel that opens every block. Imported by scripts/analysis/show_traces.py, which uses
# it as a boundary when it carves the document text back out of a prompt.
HEADER = "### Syntax"

LEVELS = ("off", "mentions", "args", "paths")

# Document-linear sections, selected by syntax.discourse and independent of `level`.
#   "skeleton"     — one line per sentence over the WHOLE document, including the ~47% of
#                    sentences that contain no event mention and are otherwise invisible.
#   "participants" — named entities and the sentences each recurs in: a cheap coreference
#                    proxy for participants shared between events. Needs the NER pipe.
DISCOURSE_COMPONENTS = ("skeleton", "participants")

_MODEL = "en_core_web_sm"
_ENGLISH = frozenset({"en", "eng", "english"})

# ClearNLP / OntoNotes dependency labels, as emitted by en_core_web_sm.
_SUBJ = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
_OBJ = frozenset({"dobj", "dative", "attr", "oprd", "ccomp", "xcomp"})
_OBL = frozenset({"prep", "agent", "npadvmod"})
_CONNECTIVE_DEPS = frozenset({"mark", "prep", "cc", "advmod"})
_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})

_MORPH_KEYS = ("Tense", "VerbForm")

# Size caps. Uncapped, one Causal-TimeBank document emits 559 path lines against a 1910-token
# body, and one EventStoryLine document 222 — measured block/document char ratios of 5.0 median
# and 13.7 max, i.e. a prompt dominated by its own annotation. These bring "paths" down to
# roughly 1.5x the document. _MAX_PATH_LEN is the most effective of them: over both English
# corpora, paths of length <= 5 are 87% of all paths, and the long remainder are both the most
# verbose to render and the weakest evidence of a causal link between two events.
#
# _MAX_MENTION_LINES and _MAX_BLOCK_CHARS are the backstops that make the block size a
# property of the setting rather than of the document: Causal-TimeBank has documents with 275
# mentions, which without them produced a 58,000-character block (~14k tokens) around a
# 1,300-character document.
_MAX_ARGS = 3
_MAX_FILLER_TOKENS = 10
_MAX_CONNECTIVES = 12
_MAX_PATH_LEN = 5
_MAX_PATHS_PER_SENTENCE = 30
_MAX_PATHS_PER_DOC = 100
_MAX_MENTION_LINES = 60
_MAX_BLOCK_CHARS = 8000
_MAX_PARTICIPANTS = 15

# spaCy's Language object is not safe for concurrent load or use. Annotation runs once per run
# before any inference concurrency starts, but `few_shot.preload` may be on a worker thread
# while main() annotates the test docs, so serialise the whole parse.
_LOCK = threading.Lock()

_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(message, file=sys.stderr)


def level() -> str:
    """The validated `syntax.level` setting; "off" when the block is absent from config."""
    lvl = (get_cfg().get("syntax") or {}).get("level", "off")
    if lvl not in LEVELS:
        raise ValueError(f"[syntax] unknown syntax.level {lvl!r}; expected one of {LEVELS}")
    return lvl


def discourse() -> Tuple[str, ...]:
    """The validated `syntax.discourse` components, in canonical order.

    Orthogonal to `level`: these are document-linear (one line per sentence, one per entity)
    rather than mention-centric, so they can be run on their own to separate what whole-text
    context contributes from what the per-mention ladder contributes.
    """
    got = (get_cfg().get("syntax") or {}).get("discourse") or []
    if isinstance(got, str):
        got = [got]
    unknown = [c for c in got if c not in DISCOURSE_COMPONENTS]
    if unknown:
        raise ValueError(f"[syntax] unknown syntax.discourse component(s) {unknown}; "
                         f"expected any of {list(DISCOURSE_COMPONENTS)}")
    return tuple(c for c in DISCOURSE_COMPONENTS if c in got)


def _lang_key(lang: str) -> str:
    """Normalise a dataset language code. MECI uses `causal-en` / `causal-tr` / ...; the other
    datasets default to `eng` (dataprep.py)."""
    key = (lang or "eng").strip().lower()
    if key.startswith("causal-"):
        key = key[len("causal-"):]
    return key


@lru_cache(maxsize=2)
def _load_pipeline(with_ner: bool = False):
    """The spaCy pipeline, or None when spaCy or the model is unavailable. Cached including the
    failure, so a missing install warns once rather than once per document.

    NER is excluded unless the "participants" discourse component asks for it — it is the one
    pipe nothing else here needs, and it is not free.
    """
    try:
        import spacy  # lazy: matches the sklearn/sentence-transformers pattern in few_shot.py
    except ImportError:
        _warn_once("no-spacy", "[syntax] spaCy is not installed — no annotation will be "
                               "appended. Add `spacy` and `en-core-web-sm` to pyproject.toml.")
        return None
    try:
        return spacy.load(_MODEL) if with_ner else spacy.load(_MODEL, exclude=["ner"])
    except OSError:
        _warn_once("no-model", f"[syntax] spaCy model {_MODEL!r} is not installed — no "
                               f"annotation will be appended. Add `en-core-web-sm` to "
                               f"pyproject.toml.")
        return None


# ── Doc construction ──────────────────────────────────────────────────────────

def _build_doc(nlp, tokens: Sequence[str], sentences: Sequence[Sequence[int]]):
    """A parsed spaCy Doc over the dataset's own tokens and sentence boundaries.

    `spaces` defaults to all-True, which is exactly right: doc_text is the tokens joined by
    single spaces. Pre-setting sent_starts keeps the parser inside the dataset's segmentation
    (verified: 0 sentence mismatches over 40 ESL docs).
    """
    from spacy.tokens import Doc

    starts = {int(s[0]) for s in sentences}
    doc = Doc(nlp.vocab, words=list(tokens),
              sent_starts=[i in starts for i in range(len(tokens))])
    return nlp(doc)


def _mention_index(doc: dict) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """(token index -> mention id, mention id -> its token indices), for mentions that have a
    surface in mentions_map."""
    tok2mention: Dict[int, str] = {}
    mention_tokens: Dict[str, List[int]] = {}
    mentions_map = doc.get("mentions_map", {})
    for mid, span in zip(doc.get("mentions", []), doc.get("spans", [])):
        if not span or mid not in mentions_map:
            continue
        idxs = [int(i) for i in span]
        mention_tokens[mid] = idxs
        for i in idxs:
            tok2mention[i] = mid
    return tok2mention, mention_tokens


def _span_root(sdoc, idxs: Sequence[int]):
    """The head token of a mention span. Set-based rather than `sdoc[a:b].root`, because 30
    EventStoryLine spans are discontinuous ("pulled ... on", "held ... hostage")."""
    inside = set(idxs)
    for i in idxs:
        tok = sdoc[i]
        if tok.head.i not in inside or tok.head.i == i:
            return tok
    return sdoc[idxs[0]]


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _mention_label(mid: str, mentions_map: Dict[str, str]) -> str:
    """`<ID surface>`, matching the markup the model reads in doc_text and the ids it must
    emit. Surfaces come from mentions_map, so they agree with format_pair_lines."""
    return f"<{mid} {mentions_map[mid]}>"


def _token_label(i: int, sdoc, tok2mention: Dict[int, str], mentions_map: Dict[str, str]) -> str:
    """`<ID surface>` when the token belongs to a mention, else its bare surface form."""
    mid = tok2mention.get(i)
    return _mention_label(mid, mentions_map) if mid else sdoc[i].text


def _render_range(lo: int, hi: int, sdoc, tok2mention: Dict[int, str],
                  mentions_map: Dict[str, str], exclude: frozenset = frozenset()) -> str:
    """Tokens [lo, hi] rendered with mention markup re-applied, truncated for length.

    A mention is emitted once, as a whole `<ID surface>` unit at its first token; its remaining
    tokens are skipped so it is never repeated. Tokens sitting inside a discontinuous mention
    but not part of it ("a gun" within "pulled ... on") still render, so nothing is lost.
    `exclude` drops token indices belonging to the mention this filler is an argument of, so a
    particle inside the mention ("into" in "checks into") cannot make the filler quote its own
    governor.
    """
    parts: List[str] = []
    emitted: set = set()
    truncated = False
    for i in range(lo, hi + 1):
        if i in exclude:
            continue
        mid = tok2mention.get(i)
        if mid:
            if mid in emitted:
                continue
            emitted.add(mid)
            parts.append(_mention_label(mid, mentions_map))
        else:
            parts.append(sdoc[i].text)
        if len(parts) >= _MAX_FILLER_TOKENS and i < hi:
            truncated = True
            break
    return " ".join(parts) + (" ..." if truncated else "")


def _morph(tok) -> str:
    """Tense/VerbForm plus derived Voice, Polarity and modality.

    en_core_web_sm populates neither Voice nor Polarity in `morph`, so both are read off the
    children: nsubjpass/auxpass for the passive, `neg` for negation. Negation matters out of
    all proportion to its 1% frequency — "the assistance is *not* helping" inverts the causal
    claim, and without this the mention line said only `lemma=help`. A modal auxiliary is
    surfaced for the same reason: "could have caused" asserts far less than "caused".
    """
    vals = []
    for key in _MORPH_KEYS:
        got = tok.morph.get(key)
        if got:
            vals.append(f"{key}={got[0]}")
    if any(c.dep_ in _PASSIVE_DEPS for c in tok.children):
        vals.append("Voice=Pass")
    if any(c.dep_ == "neg" for c in tok.children):
        vals.append("Polarity=Neg")
    modal = next((c.text for c in tok.children if c.dep_ == "aux" and c.tag_ == "MD"), None)
    if modal:
        vals.append(f"Modal={modal}")
    return "|".join(vals)


_ROLE_ORDER = {"subj": 0, "obj": 1, "obl": 2}


def _args_of(root, own_tokens: frozenset, sdoc, tok2mention, mentions_map) -> str:
    """subj/obj/obl arguments of a mention's head token, as `role="filler"` pairs, always in
    subj/obj/obl order. A ClearNLP `prep` child's subtree already contains its pobj, so the
    filler reads "into the Betty Ford Center"."""
    found: List[Tuple[int, str, str]] = []
    for child in root.children:
        if child.i in own_tokens:
            continue  # a particle belonging to the mention itself is not one of its arguments
        if child.dep_ in _SUBJ:
            role = "subj"
        elif child.dep_ in _OBJ:
            role = "obj"
        elif child.dep_ in _OBL:
            role = "obl"
        else:
            continue
        idxs = [t.i for t in child.subtree]
        filler = _render_range(min(idxs), max(idxs), sdoc, tok2mention, mentions_map,
                               exclude=own_tokens)
        if filler:
            found.append((_ROLE_ORDER[role], role, filler))
    found.sort(key=lambda f: (f[0], f[2]))
    return " ".join(f'{role}="{filler}"' for _, role, filler in found[:_MAX_ARGS])


def _connectives(start: int, end: int, sdoc, tok2mention, mentions_map) -> str:
    """Discourse/relational markers in a sentence, as `surface(dep -> head)`. These are the
    "lexical anchors" the prompts ask the model to find."""
    out: List[str] = []
    for i in range(start, end):
        tok = sdoc[i]
        if tok.dep_ not in _CONNECTIVE_DEPS:
            continue
        head = _token_label(tok.head.i, sdoc, tok2mention, mentions_map)
        out.append(f"{tok.text}({tok.dep_} -> {head})")
        if len(out) >= _MAX_CONNECTIVES:
            break
    return " ".join(out)


def _dep_path(sdoc, a, b, start: int, end: int) -> Optional[List[Tuple[int, str, bool]]]:
    """Shortest undirected dependency path from token `a` to token `b`, confined to [start, end).

    Returns [(token index, edge label, edge points from this node up to its head)], excluding
    the origin. Clamping to the sentence range means a stray cross-sentence head arc can never
    produce a path through another sentence, whether or not the parser honoured sent_starts.
    """
    if a.i == b.i:
        return None
    prev: Dict[int, Tuple[int, str, bool]] = {a.i: (-1, "", False)}
    queue = deque([a.i])
    while queue:
        cur = queue.popleft()
        if cur == b.i:
            break
        tok = sdoc[cur]
        neighbours: List[Tuple[int, str, bool]] = []
        if tok.head.i != cur and start <= tok.head.i < end:
            neighbours.append((tok.head.i, tok.dep_, True))
        for child in tok.children:
            if start <= child.i < end:
                neighbours.append((child.i, child.dep_, False))
        for nxt, label, up in neighbours:
            if nxt not in prev:
                prev[nxt] = (cur, label, up)
                queue.append(nxt)
    if b.i not in prev:
        return None
    path: List[Tuple[int, str, bool]] = []
    node = b.i
    while node != a.i:
        parent, label, up = prev[node]
        path.append((node, label, up))
        node = parent
    path.reverse()
    return path


def _format_path(a_label: str, path, sdoc, tok2mention, mentions_map) -> str:
    parts = [a_label]
    for idx, label, up in path:
        parts.append(f"-{label}->" if up else f"<-{label}-")
        parts.append(_token_label(idx, sdoc, tok2mention, mentions_map))
    return " ".join(parts) + f" (len={len(path)})"


# ── Document-linear sections ──────────────────────────────────────────────────

def _sentence_root(sdoc, start: int, end: int):
    """The head of a sentence: the token in [start, end) whose own head lies outside it."""
    for i in range(start, end):
        tok = sdoc[i]
        if tok.head.i == i or not (start <= tok.head.i < end):
            return tok
    return sdoc[start]


def _skeleton(doc: dict, sdoc, mention_sentence: Dict[str, int],
              mention_tokens: Dict[str, List[int]]) -> List[str]:
    """One line per sentence across the WHOLE document — predicate, subject, object, polarity,
    modality and how many event mentions it holds.

    This is the only part of the block that speaks about sentences containing no event mention,
    which are 47% of all sentences and 44% of all tokens in the two English corpora. Contentless
    boilerplate (the URL and dateline rows that open most ECB+ documents) is dropped: no
    mentions, a non-verbal root, and neither a subject nor an object.
    """
    counts: Dict[int, int] = {}
    for mid in mention_tokens:
        si = mention_sentence.get(mid)
        if si is not None:
            counts[si] = counts.get(si, 0) + 1

    out: List[str] = []
    for si, (start, end) in enumerate(doc.get("sentences", [])):
        root = _sentence_root(sdoc, start, end)
        subj = next((c for c in root.children if c.dep_ in _SUBJ), None)
        obj = next((c for c in root.children if c.dep_ in _OBJ), None)
        n = counts.get(si, 0)
        if not n and root.pos_ not in ("VERB", "AUX") and subj is None and obj is None:
            continue
        bits = [f"  S{si + 1}", f"pred={root.lemma_}"]
        morph = _morph(root)
        if morph:
            bits.append(morph)
        bits.append(f'subj="{subj.text}"' if subj is not None else "subj=-")
        bits.append(f'obj="{obj.text}"' if obj is not None else "obj=-")
        bits.append(f"events={n}")
        out.append(" ".join(bits))
    return out


def _participants(doc: dict, sdoc) -> List[str]:
    """Named entities and the sentences each appears in.

    An entity recurring across sentences is a cheap stand-in for coreference: two events that
    share a participant are likelier to be causally linked, and nothing else in the block
    carries information across a sentence boundary. Recurring entities are listed first, then
    by first appearance; ties broken on the surface so the section is deterministic.
    """
    sentences = doc.get("sentences", [])
    where: Dict[Tuple[str, str], set] = {}
    for ent in sdoc.ents:
        for si, (start, end) in enumerate(sentences):
            if start <= ent.start < end:
                where.setdefault((ent.text, ent.label_), set()).add(si + 1)
                break
    if not where:
        return []
    ranked = sorted(where.items(), key=lambda kv: (-len(kv[1]), min(kv[1]), kv[0][0]))
    out: List[str] = []
    for (text, label), sents in ranked[:_MAX_PARTICIPANTS]:
        marks = ", ".join(f"S{i}" for i in sorted(sents))
        out.append(f'  "{text}" ({label}) {marks}' + ("  <- recurs" if len(sents) > 1 else ""))
    omitted = len(ranked) - len(out)
    if omitted > 0:
        out.append(f"  ({omitted} further entity/entities omitted)")
    return out


# ── Block rendering ───────────────────────────────────────────────────────────

def _render(doc: dict, sdoc, lvl: str, comps: Tuple[str, ...] = ()) -> str:
    mentions_map = doc.get("mentions_map", {})
    mention_sentence = doc.get("mention_sentence", {})
    sentences = doc.get("sentences", [])
    tok2mention, mention_tokens = _mention_index(doc)
    if not mention_tokens and not comps:
        return ""

    header = f"{HEADER} (spaCy {_MODEL}"
    if lvl != "off":
        header += "; dep = relation to the head token"
    if lvl == "paths":
        header += "; in paths, A -dep-> B means A is a dependent of B, A <-dep- B the reverse"
    header += ")"
    lines: List[str] = [header]
    mention_tokens_all = mention_tokens
    if lvl == "off":
        mention_tokens = {}

    # Document-linear sections come first, and not only because overview-then-detail reads
    # better: _MAX_BLOCK_CHARS truncates from the end, so anything after the mention sections
    # would be the first thing dropped on a large document — exactly backwards, since these are
    # bounded and cheap while the "paths" list that would survive is the quadratic part.
    if "skeleton" in comps:
        rows = _skeleton(doc, sdoc, mention_sentence, mention_tokens_all)
        if rows:
            lines.append("### Discourse (one row per sentence; events = event mentions in it)")
            lines.extend(rows)
    if "participants" in comps:
        rows = _participants(doc, sdoc)
        if rows:
            lines.append("### Participants (named entities and the sentences they appear in)")
            lines.extend(rows)

    by_sentence: Dict[int, List[str]] = {}
    for mid in mention_tokens:
        by_sentence.setdefault(mention_sentence.get(mid, -1), []).append(mid)

    paths_budget = _MAX_PATHS_PER_DOC
    mention_budget = _MAX_MENTION_LINES
    omitted_mentions = 0
    pair_ids = doc.get("pair_list_ids") or []

    for sent_idx, (start, end) in enumerate(sentences):
        mids = by_sentence.get(sent_idx)
        if not mids:
            continue
        mids.sort(key=lambda m: mention_tokens[m][0])
        if mention_budget <= 0:
            omitted_mentions += len(mids)
            continue
        if len(mids) > mention_budget:
            omitted_mentions += len(mids) - mention_budget
            mids = mids[:mention_budget]
        mention_budget -= len(mids)
        lines.append(f"S{sent_idx + 1}")

        for mid in mids:
            own = frozenset(mention_tokens[mid])
            root = _span_root(sdoc, mention_tokens[mid])
            bits = [f"  {_mention_label(mid, mentions_map)}",
                    f"lemma={root.lemma_}", f"pos={root.pos_}"]
            morph = _morph(root)
            if morph:
                bits.append(morph)
            bits.append(f"dep={root.dep_}")
            bits.append("head=-" if root.dep_ == "ROOT" or root.head.i == root.i
                        else f"head={_token_label(root.head.i, sdoc, tok2mention, mentions_map)}")
            lines.append(" ".join(bits))

            if lvl in ("args", "paths"):
                args = _args_of(root, own, sdoc, tok2mention, mentions_map)
                if args:
                    lines.append(f"    args: {args}")

        if lvl in ("args", "paths"):
            conn = _connectives(start, end, sdoc, tok2mention, mentions_map)
            if conn:
                lines.append(f"  connectives: {conn}")

        if lvl == "paths" and paths_budget > 0:
            in_sentence = set(mids)
            # Unordered candidate pairs, deduplicated: a dependency path is symmetric, so
            # emitting both directions would only double the block. Sorted by token position
            # and never by pair_list_ids order, which data.shuffle_pair_list and
            # data.reshuffle_per_resample permute per run — the block must be a pure function
            # of the document or each resample pass would see different bytes.
            seen: set = set()
            pairs: List[Tuple[str, str]] = []
            for e1, e2 in pair_ids:
                if e1 in in_sentence and e2 in in_sentence and e1 != e2:
                    key = tuple(sorted((e1, e2)))
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
            pairs.sort(key=lambda p: (mention_tokens[p[0]][0], mention_tokens[p[1]][0]))

            rendered: List[str] = []
            for e1, e2 in pairs:
                if len(rendered) >= _MAX_PATHS_PER_SENTENCE or len(rendered) >= paths_budget:
                    break
                a, b = _span_root(sdoc, mention_tokens[e1]), _span_root(sdoc, mention_tokens[e2])
                path = _dep_path(sdoc, a, b, start, end)
                if path and len(path) <= _MAX_PATH_LEN:
                    rendered.append("    " + _format_path(
                        _mention_label(e1, mentions_map), path, sdoc, tok2mention, mentions_map))
            if rendered:
                lines.append("  paths:")
                lines.extend(rendered)
                paths_budget -= len(rendered)
                omitted = len(pairs) - len(rendered)
                if omitted > 0:
                    lines.append(f"    ({omitted} more pair(s) in this sentence omitted)")

    if omitted_mentions:
        lines.append(f"({omitted_mentions} further mention(s) not annotated)")

    if len(lines) <= 1:
        return ""

    block = "\n".join(lines)
    if len(block) > _MAX_BLOCK_CHARS:
        # Backstop, so no document can produce a block that dominates its own prompt. Cut on a
        # line boundary to keep the last entry well-formed.
        cut = block.rfind("\n", 0, _MAX_BLOCK_CHARS)
        block = block[:cut if cut > 0 else _MAX_BLOCK_CHARS] + "\n(annotation truncated)"
    return block


# ── Public API ────────────────────────────────────────────────────────────────

def annotate_docs(docs: List[dict]) -> None:
    """Set `doc["syntax_block"]` on every doc, in place.

    A no-op when `syntax.level` is "off". Never raises on a missing dependency, a missing
    model, a non-English document or a malformed one — those get an empty block, which
    `doc_text_with_syntax` renders as today's untouched prompt. A run costs hours; one bad
    document must not end it.
    """
    lvl = level()
    comps = discourse()
    for doc in docs:
        doc.setdefault("syntax_block", "")
    if (lvl == "off" and not comps) or not docs:
        return

    english = [d for d in docs if _lang_key(d.get("lang", "eng")) in _ENGLISH]
    skipped: Dict[str, int] = {}
    for d in docs:
        key = _lang_key(d.get("lang", "eng"))
        if key not in _ENGLISH:
            skipped[key] = skipped.get(key, 0) + 1
    if skipped:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(skipped.items()))
        _warn_once("skip-lang", f"[syntax] no pipeline for non-English documents "
                                f"({detail}) — they get no annotation block")

    parseable = [d for d in english
                 if d.get("tokens") and d.get("sentences")
                 and all(t and t.strip() for t in d["tokens"])]

    annotated = 0
    # One lock over both the load and the parse: spaCy's Language is safe for neither
    # concurrently, and few_shot.preload runs _load_train_split on a worker thread while
    # main() may be annotating the test docs on the event loop.
    with _LOCK:
        nlp = _load_pipeline(with_ner="participants" in comps)
        if nlp is None:
            return
        for doc in parseable:
            try:
                sdoc = _build_doc(nlp, doc["tokens"], doc["sentences"])
                block = _render(doc, sdoc, lvl, comps)
            except Exception as exc:  # never abort a run over one document
                _warn_once(f"render:{doc.get('id', '?')}",
                           f"[syntax] could not annotate doc {doc.get('id', '?')}: "
                           f"{type(exc).__name__}: {exc}")
                continue
            doc["syntax_block"] = block
            if block:
                annotated += 1

    detail = f"level={lvl}" + (f" discourse={'+'.join(comps)}" if comps else "")
    summary = f"[syntax] {detail} | annotated {annotated}/{len(docs)} docs"
    if skipped:
        summary += f" | skipped {sum(skipped.values())} non-English"
    print(summary)


def doc_text_with_syntax(doc: dict) -> str:
    """The document text with its annotation block appended. The only splice helper — every
    prompt-construction site calls this instead of reading doc["doc_text"] directly.

    Returns the text untouched when there is no block, so a doc that was never passed through
    `annotate_docs` degrades to the pre-feature prompt rather than raising.
    """
    block = doc.get("syntax_block") or ""
    text = doc["doc_text"]
    return f"{text}\n\n{block}" if block else text
