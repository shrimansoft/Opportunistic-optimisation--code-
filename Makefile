.PHONY: help install 


help:
	@echo "Available targets:"
	@echo "\t install \t- Install dependencies using Poetry"
	@echo "\t shell   \t- Activate Poetry virtual environment"

install:
	poetry install
shell:
	poetry shell