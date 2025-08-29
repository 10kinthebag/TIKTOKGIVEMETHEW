SHELL := /bin/zsh

.PHONY: help env install data_prep tokenize tokenize_hybrid tokenize_ground_truth tokenize_weighted train train_progressive evaluate test api streamlit demo quantize all

help:
	@echo "Training Data Options:"
	@echo "make tokenize                # tokenize with pseudo labels only"  
	@echo "make tokenize_ground_truth   # tokenize with ground truth only"
	@echo "make tokenize_hybrid         # tokenize with both datasets combined"
	@echo "make tokenize_weighted       # tokenize with weighted approach"
	@echo ""
	@echo "Training Options:"
	@echo "make train                   # standard training"
	@echo "make train_progressive       # progressive training (ground truth → pseudo)"
	@echo ""
	@echo "Testing Trained Model:"
	@echo "make test_model              # quick test of trained model"
	@echo "make demo_model              # interactive demo with trained model"
	@echo "make use_model               # example usage of trained model"
	@echo ""
	@echo "Other Commands:"
	@echo "make env                     # create & activate venv, install deps"
	@echo "make data_prep               # run data exploration, cleaning, pseudo-labeling"
	@echo "make evaluate                # evaluate on test set"
	@echo "make test                    # run hybrid pipeline test cases"
	@echo "make api                     # start Flask API"
	@echo "make streamlit               # start Streamlit dashboard"
	@echo "make demo                    # start Gradio demo"
	@echo "make quantize                # dynamic quantization of final model"

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
	python training_scripts/tokenization.py pseudo

tokenize_ground_truth:
	source nlp_env/bin/activate && \
	python training_scripts/tokenization.py ground_truth

tokenize_hybrid:
	source nlp_env/bin/activate && \
	python training_scripts/tokenization.py hybrid

tokenize_weighted:
	source nlp_env/bin/activate && \
	python training_scripts/weighted_tokenization.py

train_progressive:
	source nlp_env/bin/activate && \
	python training_scripts/progressive_training.py

test_model:
	source nlp_env/bin/activate && \
	python quick_test_model.py

demo_model:
	source nlp_env/bin/activate && \
	python hybrid_pipeline/demo_interface.py

use_model:
	source nlp_env/bin/activate && \
	python use_trained_model.py

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


