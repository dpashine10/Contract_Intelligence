# Environment
uv init --python 3.11
uv add -requirements 
or 
uv sync

# Download Dataset from hf
uv run hf download awinml/legalbench-rag \
  --repo-type dataset \
  --include "benchmarks/cuad.json" \
  --include "corpus/cuad/**" \
  --local-dir data/raw

# To split, benchmark, etc
uv run python -m app.data

