.PHONY: install test monitor clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

monitor:
	python src/monitor.py --data-path data/bank-additional-full.csv

monitor-synthetic:
	python src/monitor.py

clean:
	rm -rf reports/ data/production_traffic.parquet __pycache__
	find . -name "*.pyc" -delete

help:
	@echo "Commands:"
	@echo "  make install           Install dependencies"
	@echo "  make test              Run 16 drift detection tests"
	@echo "  make monitor           Run monitoring on real data"
	@echo "  make monitor-synthetic Run monitoring on synthetic data"
