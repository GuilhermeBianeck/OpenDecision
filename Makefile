.PHONY: test lint check build demo

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

check: lint test

build:
	python -m build

demo:
	opendecision decide --model demo --state 'The billing invoice was charged twice' --question 'Which team?' --choices billing technical
