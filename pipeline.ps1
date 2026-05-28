uv run main.py prepare-data --force
uv run main.py train
uv run main.py compare --contaminations 0.05 0.10 0.15 0.20 --label-ratios 0.01 0.05 0.10 0.20 0.50 1.0 --scope per-house
uv run main.py dashboard
