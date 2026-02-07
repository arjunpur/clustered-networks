.PHONY: run lint format figures

run:
	uv run jupyter lab base_model.ipynb

lint:
	uv run ruff check clustered_networks/

format:
	uv run ruff format clustered_networks/
