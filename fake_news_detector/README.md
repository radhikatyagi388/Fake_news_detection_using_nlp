# Fake News Detector — Streamlit + NLP

This project follows the BRD/FRD/PRD for the Fake News Detection project.

## 1. Folder structure

```text
fake_news_detector/
│
├── app.py
├── train_model.py
├── text_utils.py
├── requirements.txt
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── fake_news_pipeline.pkl
│
└── reports/
    ├── model_comparison.csv
    └── metrics.json
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
venv\Scripts\activate
```

## 3. Put dataset files

Copy `Fake.csv` and `True.csv` into `data/`.

## 4. Train

```bash
python train_model.py
```

This creates the saved model and evaluation metrics.

## 5. Start Streamlit

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Important

The model is a pattern classifier, not a fact-checker. It should not be treated as proof that an article is true or false.
