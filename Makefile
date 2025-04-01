SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

install:
	poetry install

streamlit-dev:
	poetry run streamlit run streamlit_app.py