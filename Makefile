PY ?= python

.PHONY: test synthetic smoke clean-smoke

test:            ## unit tests (no GPU, no ultralytics needed)
	$(PY) -m pytest -q

synthetic:       ## tiny synthetic dataset for the smoke run
	$(PY) scripts/make_synthetic.py --out data/synthetic

smoke: synthetic ## full loop on synthetic data, CPU, a few minutes
	$(PY) -m selftrain.loop --config configs/smoke.yaml

clean-smoke:
	rm -rf runs/smoke data/synthetic
