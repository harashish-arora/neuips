PYTHON ?= python

.PHONY: install quick fastprop fastsolv mlp all hpo hpo-quick

install:
	$(PYTHON) -m pip install -r requirements.txt

quick:
	$(PYTHON) scripts/run_descriptor_baselines.py --quick

fastprop:
	$(PYTHON) scripts/run_descriptor_baselines.py --model_type fastprop

fastsolv:
	$(PYTHON) scripts/run_descriptor_baselines.py --model_type fastsolv

mlp:
	$(PYTHON) scripts/run_descriptor_baselines.py --model_type mlp

all:
	$(PYTHON) scripts/run_descriptor_baselines.py --model_type fastprop fastsolv mlp

hpo:
	$(PYTHON) scripts/run_descriptor_hyperparam_search.py

hpo-quick:
	$(PYTHON) scripts/run_descriptor_hyperparam_search.py --quick
