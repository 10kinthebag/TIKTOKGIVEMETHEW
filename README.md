# 10kinthebag — Filtering the Noise: ML for Trustworthy Location Reviews

## Setup
```bash
python3 -m venv nlp_env
source nlp_env/bin/activate
pip install -r requirements.txt
```

## Data prep
```bash
python data_prep_scripts/data_exploration.py
python data_prep_scripts/data_cleaning.py
python data_prep_scripts/pseudo_labeling.py
python data_prep_scripts/dataset_preparation.py
```

## Tokenize and train
```bash
python training_scripts/tokenization.py
python training.py
```

## Evaluate
```bash
python evaluation.py
```

## Hybrid pipeline quick test
```bash
python pipeline_testing.py
```

## API (Flask)
```bash
python api_interface.py
# POST /classify  JSON: {"text": "Great food..."}
# POST /batch_classify  JSON: {"texts": ["...", "..."]}
```

## Demo (Gradio)
```bash
python demo_interface.py
```

## Streamlit dashboard
```bash
streamlit run app.py
```

