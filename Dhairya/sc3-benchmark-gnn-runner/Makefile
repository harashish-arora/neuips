PYTHON ?= python

.PHONY: install quick gcn gpu all hpo hpo-quick

install:
	$(PYTHON) -m pip install -r requirements.txt

quick:
	$(PYTHON) scripts/run_gnn_baselines.py --quick

gcn:
	$(PYTHON) scripts/run_gnn_baselines.py --gnn_type GCN

gpu:
	$(PYTHON) scripts/run_gnn_baselines.py --gpu

all:
	$(PYTHON) scripts/run_gnn_baselines.py --gnn_type GCN GAT GIN

hpo:
	$(PYTHON) scripts/run_gnn_hyperparam_search.py

hpo-quick:
	$(PYTHON) scripts/run_gnn_hyperparam_search.py --quick
