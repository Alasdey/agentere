#!/usr/bin/env python3
"""
Generate LaTeX figures from EventStoryLine 0.9 in tikzmarkbox style.
Compile with: pdflatex (standard, no special engine needed)
Run:          uv run scripts/gen_figure.py
Outputs:      scripts/figure_esl_1.tex, scripts/figure_esl_2.tex
"""

import re
import os
import sys
from datasets import load_dataset
import graphviz as gv

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Regex ──────────────────────────────────────────────────────────────────────
TAG_RE    = re.compile(r'<(\d+)\s+([^>]+)>')
CAUSES_RE = re.compile(r'<([^>\s]+)\s+[^>\n]+>\s+causes\s+<([^>\s]+)\s+[^>\n]+>')

# ── Text helpers ───────────────────────────────────────────────────────────────

def strip_url(text):
    m = re.search(r'\.\s*(?:s?html?|php|xml|aspx)(?:\s*\?\s*\S+)?\s+', text, re.IGNORECASE)
    if m:
        return text[m.end():].lstrip()
    m = re.search(r'(?<!\d)\d{7,}(?!\d)\s+', text)
    return text[m.end():].lstrip() if m else text

_ESC = [('&','\\&'), ('%','\\%'), ('$','\\$'), ('#','\\#'),
        ('_','\\_'), ('{','\\{'), ('}','\\}')]

def ltx(s):
    for old, new in _ESC:
        s = s.replace(old, new)
    return s

def detokenize(s):
    s = re.sub(r'\s+([.,;:!?])',           r'\1',  s)
    s = re.sub(r"\s+'(s|re|ve|ll|d|t)\b",  r"'\1", s, flags=re.I)
    s = re.sub(r'\(\s+',                   '(',    s)
    s = re.sub(r'\s+\)',                   ')',     s)
    s = re.sub(r'  +',                     ' ',    s)
    return s.strip()

def tagged_to_markbox(raw):
    body = strip_url(raw)
    parts, pos = [], 0
    for m in TAG_RE.finditer(body):
        parts.append(ltx(body[pos:m.start()]))
        parts.append(f'\\tikzmarkbox{{n{m.group(1)}}}{{{ltx(m.group(2).strip())}}}')
        pos = m.end()
    parts.append(ltx(body[pos:]))
    return detokenize(''.join(parts))

# ── Parsing ────────────────────────────────────────────────────────────────────

def get_mentions(raw):
    return {m.group(1): m.group(2).strip() for m in TAG_RE.finditer(raw)}

def get_edges(annots):
    seen, out = set(), []
    for m in CAUSES_RE.finditer(annots):
        e = (m.group(1), m.group(2))
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

# ── Graph PNG via Graphviz ─────────────────────────────────────────────────────

def render_graph_png(nodes, edges, mentions, dest_path):
    """Render the causal graph with Graphviz dot and save as PNG."""
    g = gv.Digraph()
    g.attr(rankdir='LR', nodesep='0.55', ranksep='0.9', bgcolor='white', dpi='300')
    g.attr('node', shape='box', style='filled,rounded',
           fillcolor='#e6e6ff', color='#0000ff',
           fontname='Helvetica-Bold', fontsize='16', margin='0.15,0.08')
    g.attr('edge', color='#b30000', arrowhead='normal', penwidth='2.2',
           arrowsize='1.1')
    for n in nodes:
        g.node(f'n{n}', label=mentions.get(n, n))
    for s, t in edges:
        g.edge(f'n{s}', f'n{t}')
    with open(dest_path, 'wb') as f:
        f.write(g.pipe(format='png'))

# ── LaTeX building blocks ──────────────────────────────────────────────────────

PREAMBLE = r"""\documentclass{article}
\usepackage[a4paper, margin=2cm]{geometry}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{xcolor}
\usepackage{graphicx}

% Inline event-mention box
\newcommand{\tikzmarkbox}[3][]{%
  \tikz[remember picture, baseline=(#2.base)]%
    \node[draw=blue, fill=blue!10, rectangle, rounded corners,
      inner sep=1pt, outer sep=0pt, #1, anchor=base] (#2) {#3};%
}

\parindent=0pt
\begin{document}
"""
POSTAMBLE = "\n\\end{document}\n"


def build_figure(doc_id, raw_text, annots, out_dir, fw=12.0):
    men = get_mentions(raw_text)
    edg = get_edges(annots)
    nds = sorted({n for e in edg for n in e},
                 key=lambda x: int(x) if x.isdigit() else x)

    safe = re.sub(r'\.[^.]+$', '', doc_id)
    safe = re.sub(r'[^\w-]', '_', safe)
    png_name = f'graph_{safe}.png'
    render_graph_png(nds, edg, men, os.path.join(out_dir, png_name))

    # ── 1. Annotated text ─────────────────────────────────────────────────────
    text_section = (
        f"\\begin{{tikzpicture}}[remember picture]\n"
        f"\\node[text width={fw}cm, align=justify] (mainText) {{%\n"
        f"{tagged_to_markbox(raw_text)}%\n"
        f"}};\n"
        f"\\end{{tikzpicture}}\n"
    )

    # ── 2. Causal graph (Graphviz-rendered PNG) ───────────────────────────────
    graph_section = f"\\includegraphics[width={fw}cm]{{{png_name}}}\n"

    # ── 3. Legend ──────────────────────────────────────────────────────────────
    legend_section = (
        "\\begin{center}\\small\n"
        "\\tikz[baseline=(b.base)]{\\node[draw=blue, fill=blue!10, rounded corners,"
        " inner sep=2pt, font=\\small] (b) {event};}\\enspace event mention\\qquad\n"
        "\\tikz[baseline=0pt]{\\draw[-{Stealth[length=2mm]}, thick, red!70!black,"
        " shorten >=2pt](0,0.5ex)--(2em,0.5ex);}\\enspace causes\n"
        "\\end{center}\n"
    )

    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n\n"
        f"{{\\normalsize\\bfseries EventStoryLine~0.9"
        f"\\enspace$\\cdot$\\enspace\\texttt{{{ltx(doc_id)}}}}}\n\n"
        "\\medskip\n\n"
        + text_section + "\n"
        "\\medskip\n\n"
        + graph_section + "\n"
        "\\smallskip\n\n"
        + legend_section + "\n"
        f"\\caption{{Annotated example from EventStoryLine~0.9"
        f" (\\texttt{{{ltx(doc_id)}}})."
        f" Blue boxes are event mentions. Arrows show causes relations.}}\n"
        f"\\label{{fig:{doc_id.replace('.', '-').replace('_', '-')}}}\n"
        "\\end{figure}\n"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate LaTeX figures from a HuggingFace event-causality dataset.')
    parser.add_argument('dataset', help='HuggingFace dataset repo ID')
    parser.add_argument('ids', nargs='+', help='Document IDs to render (id column values)')
    parser.add_argument('--split', default='train', help='Dataset split (default: train)')
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split})…", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split)

    id_to_row = {row['id']: row for row in ds}

    for doc_id in args.ids:
        if doc_id not in id_to_row:
            print(f"  ERROR: id '{doc_id}' not found in split '{args.split}'", file=sys.stderr)
            continue
        row  = id_to_row[doc_id]
        body = build_figure(row['id'], row['text'], row['annots'], out_dir=DIR)
        safe = re.sub(r'\.[^.]+$', '', doc_id)
        safe = re.sub(r'[^\w-]', '_', safe)
        dest = os.path.join(DIR, f'figure_{safe}.tex')
        with open(dest, 'w') as f:
            f.write(PREAMBLE + '\n' + body + POSTAMBLE)
        print(f"  {doc_id} → {dest}", file=sys.stderr)


if __name__ == '__main__':
    main()
