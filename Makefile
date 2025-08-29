SHELL := /bin/zsh

.PHONY: help env install data_prep tokenize train evaluate test api streamlit demo quantize all

help:
	@echo "make env        # create & activate venv, install deps"
	@echo "make data_prep  # run data exploration, cleaning, pseudo-labeling, split"
	@echo "make tokenize   # tokenize datasets"
	@echo "make train      # fine-tune model and save to ./final_model"
	@echo "make evaluate   # evaluate on test set"
	@echo "make test       # run hybrid pipeline test cases"
	@echo "make api        # start Flask API"
	@echo "make streamlit  # start Streamlit dashboard"
	@echo "make demo       # start Gradio demo"
	@echo "make quantize   # dynamic quantization of final model"
	@echo "make all        # env -> data_prep -> tokenize -> train -> evaluate -> test"

env:
	python3 -m venv nlp_env && \
	source nlp_env/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

install: env

data_prep:
	source nlp_env/bin/activate && \
	python data_prep_scripts/data_exploration.py && \
	python data_prep_scripts/data_cleaning.py && \
	python data_prep_scripts/pseudo_labeling.py && \
	python data_prep_scripts/dataset_preparation.py

tokenize:
	source nlp_env/bin/activate && \
	python training_scripts/tokenization.py

train:
	source nlp_env/bin/activate && \
	python -c "from training_scripts.trainer_setup import get_trainer; t=get_trainer(); t.train(); t.save_model('./final_model'); t.tokenizer.save_pretrained('./final_model')"

evaluate:
	source nlp_env/bin/activate && \
	PYTHONPATH=. python -m evaluation_scripts.evaluation

test:
	source nlp_env/bin/activate && \
	PYTHONPATH=. python -m hybrid_pipeline.pipeline_testing

api:
	source nlp_env/bin/activate && \
	python hybrid_pipeline/api_interface.py

streamlit:
	source nlp_env/bin/activate && \
	streamlit run hybrid_pipeline/app.py

demo:
	source nlp_env/bin/activate && \
	python hybrid_pipeline/demo_interface.py

quantize:
	source nlp_env/bin/activate && \
	python performance/quantization.py

all: install data_prep tokenize train evaluate test
	@echo "✅ Full pipeline completed"


