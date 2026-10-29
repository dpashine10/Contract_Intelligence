"""Minimal Streamlit demo.

  uv run streamlit run interfaces/web_app.py
  uv run streamlit run interfaces/web_app.py -- --config configs/baseline_frozen.yaml   # baseline instead
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.config import load_config  # noqa: E402
from app.pipeline import build_pipeline  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/final_frozen.yaml")
args, _ = parser.parse_known_args()


@st.cache_resource(show_spinner="Indexing contracts…")
def get_pipeline(config_path: str):
    return build_pipeline(load_config(config_path))


st.set_page_config(page_title="Contract Clause Search", layout="wide")
st.title("Contract Clause Search")

pipeline = get_pipeline(args.config)
config = pipeline.config
st.caption(
    f"Config **{config['name']}** · {len(pipeline.corpus.documents)} contracts · {len(pipeline.chunks)} passages · "
    f"BM25 (k1={config['retrieval']['k1']}, b={config['retrieval']['b']}) · "
    f"abstain if {config['abstention'].get('score', 'raw_top1')} < {config['abstention']['threshold']:.3f}"
)

question = st.text_input("Ask a question about the contracts", placeholder="e.g. Does the agreement allow sublicensing?")
top_k = st.slider("Passages to show", 1, 20, config["ui"]["display_top_k"])
show_anyway = st.checkbox("Diagnostic: show ranked passages even when the system abstains", value=False)

if question.strip():
    answer = pipeline.answer(question, top_k)
    texts = pipeline.texts
    shown = answer["results"] or (pipeline.search(question, top_k) if show_anyway else [])
    score = "none" if answer["abstention_score"] is None else f"{answer['abstention_score']:.3f}"
    label = answer["abstention_score_kind"]
    if answer["abstained"]:
        st.warning(
            f"**No answer returned.** No passage is relevant enough ({label} {score} is below the "
            f"threshold {answer['threshold']:.3f}), so the system abstains instead of guessing."
        )
        if shown:
            st.info("Diagnostic view: these passages were NOT returned as an answer.")
    else:
        st.success(f"{len(answer['results'])} passages ({label} {score} ≥ threshold {answer['threshold']:.3f})")
    if shown:
        for r in shown:
            with st.container(border=True):
                local = "" if r.rerank_score is None else f" · in-document score `{r.rerank_score:.3f}`"
                st.markdown(f"**#{r.rank}** · BM25 `{r.score:.3f}`{local} · `{r.file_path}` · characters `[{r.start}, {r.end})`")
                passage = html.escape(texts[r.file_path][r.start : r.end])
                st.markdown(f"<div style='white-space:pre-wrap;font-size:0.9em'>{passage}</div>", unsafe_allow_html=True)
