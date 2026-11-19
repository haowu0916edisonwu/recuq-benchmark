PYTHON ?= python

.PHONY: validate reproduce reproduce-full paired-control-positive paper-tables figures test

validate:
	$(PYTHON) scripts/validate_release.py

reproduce:
	$(PYTHON) scripts/reproduce_release.py --mode quick

reproduce-full:
	$(PYTHON) scripts/reproduce_release.py --mode full

paired-control-positive:
	$(PYTHON) scripts/evaluate_source_response_interventions.py \
		--base_predictions artifacts/base_predictions/llama_fixed_rankings.jsonl.gz \
		--uq_predictions artifacts/uq_outputs/llama/*.jsonl.gz \
		--output_dir outputs/paired_control/both_positive \
		--require_both_positive_effects \
		--write_instance_details

paper-tables:
	$(PYTHON) scripts/assemble_paper_tables.py

figures:
	MPLBACKEND=Agg $(PYTHON) scripts/make_main_paper_figures.py
	MPLBACKEND=Agg $(PYTHON) scripts/make_case_study_figure.py --formats pdf svg png
	MPLBACKEND=Agg $(PYTHON) scripts/make_signed_effect_figure.py

test:
	$(PYTHON) -m pytest -q
