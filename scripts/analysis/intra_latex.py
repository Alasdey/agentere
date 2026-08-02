#!/usr/bin/env python3
"""
Emit a LaTeX table of the binary intra-sentence metrics (binary_intra_precision /
binary_intra_recall / binary_intra_f1) for every MLflow run started in the last
N days, with the config parameters that identify the run.

Reported parameters (MLflow param key → column):
    model.default_model_id           → Model       (model used)
    experiment.resampling.enabled    → Res.        (resampling)
    experiment.resampling.n_runs     → $n_r$       (nresampling)
    few_shot.enabled                 → FS          (fewshot)
    few_shot.n_examples              → $n_f$       (nfewshots)
    active_dataset                   → Dataset     (dataset)
    dataset.max_examples             → Ex.         (max example)
    dataset.prompt                   → Prompt      (prompt)
    few_shot.cot_generation.enabled  → CoT         (cot_generation)
    few_shot.cot_generation.blind    → Bl.         (blind)
    few_shot.cot_generation.rewrite  → Rw.         (rewrite)
    few_shot.selection               → Sel.        (fewshot_selection)

Params come from mlflow.log_params(_flatten_config(config)) in
utils/mlflow_tracker.py. A param that a run never logged (e.g. cot_generation
sub-keys on older runs) is rendered as "--" rather than silently defaulted.

Rows are blocked by dataset and, inside a block, ordered by date (newest first;
--ascending for chronological, --sort f1 to rank by score instead). The best F1
in each dataset block is bolded. Crashed/running runs log no params and no
metrics, so they are dropped unless --all is passed.

Usage (from the agentere env; the sqlite path resolves relative to the project root):
    uv run python scripts/analysis/intra_latex.py --days 7
    uv run python scripts/analysis/intra_latex.py --days 30 --dataset maven_ere
    uv run python scripts/analysis/intra_latex.py --days 14 --group          # mean±std over identical configs
    uv run python scripts/analysis/intra_latex.py --days 7 --resizebox --out scripts/analysis/out/intra.tex
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "agentere"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # mirrors config.yaml's mlflow.tracking_uri

# The three binary intra-sentence metrics logged per run.
INTRA_KEYS = ("binary_intra_precision", "binary_intra_recall", "binary_intra_f1")

# Sentinel for "this param key is absent from this run" (never a silent default).
ABSENT = "\x00absent"

# (mlflow param key, LaTeX column header, renderer) in table order.
BOOL_TRUE = {"True", "true", "1"}


def _bool(v: str) -> str:
    if v == ABSENT:
        return "-"
    return "Y" if v in BOOL_TRUE else "N"


def _num(v: str) -> str:
    return "-" if v == ABSENT else tex_escape(v)


def _dataset(v: str) -> str:
    if v == ABSENT:
        return "-"
    return tex_escape(DATASET_SHORT.get(v, v))


def _prompt(v: str) -> str:
    if v == ABSENT:
        return "-"
    return r"\texttt{" + tex_escape(v) + "}"


def _model(v: str) -> str:
    """Model ids are `vendor/name`; the vendor is dropped for width, but only when
    the name already carries it (deepseek/deepseek-v4-flash → deepseek-v4-flash),
    so nothing that distinguishes two ids is lost."""
    if v == ABSENT:
        return "-"
    vendor, sep, name = v.partition("/")
    if sep and name.lower().startswith(vendor.lower()):
        v = name
    return r"\texttt{" + tex_escape(v) + "}"


def _sel(v: str) -> str:
    return "-" if v == ABSENT else tex_escape(v)


DATASET_SHORT = {
    "maven_ere": "MAVEN-ERE",
    "event_story_line": "ESL",
    "causal_timebank": "CTB",
    "meci": "MECI",
}

PARAM_COLS = [
    ("model.default_model_id",          "Model",    "l", _model),
    ("active_dataset",                  "Dataset",  "l", _dataset),
    ("dataset.max_examples",            "Ex.",      "r", _num),
    ("dataset.prompt",                  "Prompt",   "l", _prompt),
    ("experiment.resampling.enabled",   "Res.",     "c", _bool),
    ("experiment.resampling.n_runs",    r"$n_r$",   "c", _num),
    ("few_shot.enabled",                "FS",       "c", _bool),
    ("few_shot.n_examples",             r"$n_f$",   "c", _num),
    ("few_shot.selection",              "Sel.",     "l", _sel),
    ("few_shot.cot_generation.enabled", "CoT",      "c", _bool),
    ("few_shot.cot_generation.blind",   "Bl.",      "c", _bool),
    ("few_shot.cot_generation.rewrite", "Rw.",      "c", _bool),
]
PARAM_KEYS = [k for k, _, _, _ in PARAM_COLS]


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that show up in param values (prompt names carry
    underscores, model ids carry slashes and dots)."""
    out = []
    for ch in str(s):
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def resolve_tracking_uri(uri: str) -> str:
    """Make a relative sqlite path absolute against the project root so the script
    works regardless of the current directory (same idiom as list_ablations.py)."""
    project_root = Path(__file__).resolve().parents[2]
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        db = uri[len(prefix):]
        if not db.startswith("/"):
            db = str(project_root / db)
        return f"sqlite:///{db}"
    if not uri.startswith(("http", "/")):
        return str(project_root / uri)
    return uri


def run_date(run) -> str:
    """UTC start datetime of the run (matches the Z-suffixed run name)."""
    return datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def params_of(run) -> dict[str, str]:
    return {k: run.data.params.get(k, ABSENT) for k in PARAM_KEYS}


def intra_of(run) -> dict[str, float | None]:
    m = run.data.metrics
    return {k: (m[k] if k in m else None) for k in INTRA_KEYS}


def has_all_intra(run) -> bool:
    return all(k in run.data.metrics for k in INTRA_KEYS)


def fmt_metric(v: float | None, digits: int, bold: bool = False) -> str:
    if v is None:
        return "-"
    s = f"{v:.{digits}f}"
    return r"\textbf{" + s + "}" if bold else s


def fmt_agg(vals: list[float], digits: int, bold: bool = False) -> str:
    """mean$\\pm$std over repeats; std omitted for a single run."""
    mean = statistics.fmean(vals)
    s = f"{mean:.{digits}f}"
    if len(vals) > 1:
        s += r"{\scriptsize$\pm$" + f"{statistics.stdev(vals):.{digits}f}" + "}"
    return r"\textbf{" + s + "}" if bold else s


def build_rows(runs, group: bool, digits: int) -> list[dict]:
    """One dict per table row: {'params', 'cells', 'f1', 'sort_key'}.

    group=False → one row per run (date column first).
    group=True  → one row per distinct param tuple, metrics aggregated as
                  mean±std with the repeat count in the first column.
    """
    rows = []
    if not group:
        for r in runs:
            p = params_of(r)
            m = intra_of(r)
            rows.append({
                "params": p,
                "lead": tex_escape(run_date(r)),
                "metrics": {k: fmt_metric(m[k], digits) for k in INTRA_KEYS},
                "raw": {k: m[k] for k in INTRA_KEYS},
                "f1": m["binary_intra_f1"],
                "start": r.info.start_time,
                "n": 1,
            })
        return rows

    buckets: dict[tuple, list] = {}
    for r in runs:
        buckets.setdefault(tuple(params_of(r)[k] for k in PARAM_KEYS), []).append(r)
    for key, group_runs in buckets.items():
        p = dict(zip(PARAM_KEYS, key))
        vals = {k: [intra_of(r)[k] for r in group_runs if intra_of(r)[k] is not None] for k in INTRA_KEYS}
        f1s = vals["binary_intra_f1"]
        rows.append({
            "params": p,
            "lead": str(len(group_runs)),
            "metrics": {k: (fmt_agg(vals[k], digits) if vals[k] else "-") for k in INTRA_KEYS},
            "raw": {k: (statistics.fmean(vals[k]) if vals[k] else None) for k in INTRA_KEYS},
            "f1": statistics.fmean(f1s) if f1s else None,
            # a bucket spans several runs → order it by its most recent one
            "start": max(r.info.start_time for r in group_runs),
            "n": len(group_runs),
        })
    return rows


def emit_latex(rows: list[dict], *, days: int, group: bool, digits: int,
               resizebox: bool, caption: str, label: str,
               sort_by: str, ascending: bool) -> str:
    lead_head = "Runs" if group else "Date (UTC)"
    lead_align = "c" if group else "l"
    aligns = lead_align + "".join(a for _, _, a, _ in PARAM_COLS) + "rrr"
    headers = [lead_head] + [h for _, h, _, _ in PARAM_COLS] + ["P", "R", "F1"]

    # Dataset always blocks the table; rows are ordered within a block only.
    # Default is newest run first (--ascending flips to chronological).
    if sort_by == "date":
        inner = lambda r: r["start"]
    else:
        inner = lambda r: (r["f1"] if r["f1"] is not None else float("-inf"))
    sign = 1 if ascending else -1
    rows = sorted(rows, key=lambda r: (r["params"]["active_dataset"], sign * inner(r)))
    best_f1: dict[str, float] = {}
    for r in rows:
        ds, f1 = r["params"]["active_dataset"], r["f1"]
        if f1 is not None and f1 > best_f1.get(ds, float("-inf")):
            best_f1[ds] = f1

    lines = [
        "% Generated by scripts/analysis/intra_latex.py "
        f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC, last {days} day(s))",
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
    ]
    if resizebox:
        lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines += [
        rf"  \begin{{tabular}}{{{aligns}}}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]

    prev_ds = None
    for r in rows:
        ds = r["params"]["active_dataset"]
        if prev_ds is not None and ds != prev_ds:
            lines.append(r"    \midrule")
        prev_ds = ds
        cells = [r["lead"]]
        cells += [render(r["params"][key]) for key, _, _, render in PARAM_COLS]
        is_best = r["f1"] is not None and r["f1"] == best_f1.get(ds)
        for k in INTRA_KEYS:
            cell = r["metrics"][k]
            if k == "binary_intra_f1" and is_best and cell != "-":
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if resizebox:
        lines.append("  }")
    lines += [
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7,
                    help="include runs started within the last N days (default 7)")
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=EXPERIMENT_NAME)
    ap.add_argument("--dataset", default=None,
                    help="keep only runs whose active_dataset matches (e.g. maven_ere)")
    ap.add_argument("--group", action="store_true",
                    help="collapse runs with an identical param tuple into one row (mean$\\pm$std)")
    ap.add_argument("--all", action="store_true",
                    help="keep runs that lack binary_intra metrics (crashed / still running)")
    ap.add_argument("--sort", choices=("date", "f1"), default="date",
                    help="row order inside each dataset block (default date, newest first)")
    ap.add_argument("--ascending", action="store_true",
                    help="flip the order: oldest run first (--sort date) / lowest F1 first (--sort f1)")
    ap.add_argument("--digits", type=int, default=3, help="decimal places for P/R/F1 (default 3)")
    ap.add_argument("--resizebox", action="store_true",
                    help="wrap the tabular in \\resizebox{\\textwidth}{!}{...} (needs graphicx)")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:intra-runs")
    ap.add_argument("--out", default=None, help="write the LaTeX to this path instead of stdout")
    args = ap.parse_args()

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"No experiment named '{args.experiment}' found at {tracking_uri}", file=sys.stderr)
        sys.exit(1)

    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.start_time >= {cutoff_ms}",
        max_results=50_000,
    )

    n_fetched = len(runs)
    runs = [r for r in runs if r.data.params]  # crashed-before-logging runs carry no config
    if not args.all:
        runs = [r for r in runs if has_all_intra(r)]
    if args.dataset:
        runs = [r for r in runs if r.data.params.get("active_dataset", ABSENT) == args.dataset]

    if not runs:
        print(f"No runs in the last {args.days} day(s) at {tracking_uri} "
              f"({n_fetched} fetched, all filtered out).", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(runs, group=args.group, digits=args.digits)
    caption = args.caption or (
        f"Binary intra-sentence precision, recall and F1 over the last {args.days:g} day(s) "
        f"({len(runs)} run{'s' if len(runs) != 1 else ''}"
        + (f", {len(rows)} configurations, mean$\\pm$std over repeats" if args.group else "")
        + "). Res./FS/CoT/Bl./Rw.\\ are resampling, few-shot, CoT generation, blind and rewrite; "
          r"Ex.\ is the evaluated example cap; "
          r"$n_r$ is the number of resampling runs and $n_f$ the number of few-shot examples; "
          r"``--'' marks a parameter the run did not log."
    )
    tex = emit_latex(rows, days=args.days, group=args.group, digits=args.digits,
                     resizebox=args.resizebox, caption=caption, label=args.label,
                     sort_by=args.sort, ascending=args.ascending)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} rows ({len(runs)} runs) → {out}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
