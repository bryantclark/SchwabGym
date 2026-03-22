.PHONY: test lint typecheck fix check clean

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy schwabgym

fix:
	ruff check --fix .
	ruff format .

check: lint typecheck test

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .coverage htmlcov
