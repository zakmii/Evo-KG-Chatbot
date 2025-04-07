SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

install:
	poetry install

streamlit-dev:
	nohup poetry run streamlit run streamlit_app_2.py --server.port 8501