.PHONY: test black

test:
	uv run pytest -s ./tests/*

black:
	uv run black .
