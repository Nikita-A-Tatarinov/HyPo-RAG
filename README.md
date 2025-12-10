# HyPo-RAG — concise quickstart

Lightweight, training-free pipeline for faithful knowledge-graph question answering. HyPo-RAG combines hypothesis-guided planning, graph-constrained path retrieval and a deterministic policy to select compact hierarchical evidence under token budgets.

## What this repo contains

- `src/` — core implementation (config, HGP generation, KG-Trie, path generation, scoring, formatting, inference, eval).
- `run_experiments.py` — experiment runner for sample and full evaluations.
- `results/` — outputs (per-run subfolders, `results/final/test_results.json`).

## Quick setup (tested on Linux)

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
# or editable install if available
pip install -e .
```

3. Set API / tokens

```bash
export OPENAI_API_KEY="<your-openai-key>"
# Optional (if using HuggingFace models locally)
export HF_TOKEN="<your-hf-token>"
```

Notes: the code falls back to simpler heuristics if optional packages (e.g., spaCy models) are missing.

## Quick runs

Run a tiny smoke test (5 examples):

```bash
python run_experiments.py --dataset rmanluo/RoG-webqsp --split test --max-examples 5
```

Run full evaluation on WebQSP (1,628 questions):

```bash
python run_experiments.py --dataset rmanluo/RoG-webqsp --split test
```

Outputs are written under `results/` (per-run folder) and a canonical snapshot is in `results/final/test_results.json`.