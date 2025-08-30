.PHONY: install run test build-image

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest -q

build-image:
	docker build -t doc-qa-system:latest .
