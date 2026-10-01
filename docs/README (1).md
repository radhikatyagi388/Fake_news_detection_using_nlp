# Fake News Detection Using NLP

An end-to-end Fake News Detection application built with Natural Language Processing, TF-IDF and Machine Learning.

## Features

- Fake / Real news classification
- NLP text preprocessing
- TF-IDF vectorization
- Unigram + Bigram features
- Logistic Regression model
- Prediction confidence
- Low-confidence warning
- Model performance metrics
- Article statistics
- Sample news articles
- Interactive Streamlit interface
- Prediction history during the session

> **Disclaimer:** This system does not independently fact-check news using external sources. It predicts linguistic patterns learned from the training dataset. Human verification is recommended for important decisions.

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| NLP | NLTK |
| Vectorization | TF-IDF |
| ML | Scikit-learn |
| Model | Logistic Regression |
| Data Processing | Pandas, NumPy |
| Model Saving | Joblib |
| Dataset | ISOT Fake and Real News Dataset |

## Project Structure

```text
fake_news_detector/
├── app.py
├── train_model.py
├── text_utils.py
├── requirements.txt
├── README.md
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── README.txt
├── models/
│   └── fake_news_pipeline.pkl
└── reports/
    ├── model_comparison.csv
    └── metrics.json
```

## Dataset

The project uses the ISOT Fake and Real News Dataset.

- `Fake.csv` → fake news articles
- `True.csv` → real news articles

The dataset contains approximately 44,000+ articles.

Large dataset files should normally be excluded from GitHub using `.gitignore`.

## Machine Learning Pipeline

```text
News Article
     ↓
Text Cleaning
     ↓
Stopword Removal
     ↓
Lemmatization
     ↓
TF-IDF
     ↓
Logistic Regression
     ↓
FAKE / REAL
     ↓
Confidence Score
```

## Text Preprocessing

The application performs:

1. Lowercasing
2. Source/Reuters tag removal
3. URL removal
4. HTML removal
5. Number removal
6. Punctuation removal
7. Stopword removal
8. Lemmatization
9. Extra whitespace removal

## TF-IDF Configuration

```text
Maximum Features: 30,000
N-grams: Unigrams + Bigrams
```

The TF-IDF vectorizer is fitted on training data and then used to transform test/user data.

## Models Compared

The training workflow compares:

1. Logistic Regression
2. Linear SVM
3. Naive Bayes
4. Passive Aggressive Classifier
5. Decision Tree
6. Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Training Time
- Classification Report
- Confusion Matrix

The deployed application uses Logistic Regression because it provides probability estimates for the confidence display.

## Model Performance

Current final model results:

| Metric | Result |
|---|---:|
| Accuracy | **98.53%** |
| F1 Score | **98.48%** |
| Features | **30,000** |
| Vectorizer | TF-IDF |
| Model | Logistic Regression |

These results are test-dataset metrics and do not guarantee real-world fact-checking accuracy.

## Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK resources

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Train the Model

Place the dataset files in:

```text
data/
├── Fake.csv
└── True.csv
```

Then run:

```bash
python train_model.py
```

This creates the trained model and evaluation reports.

## Run the Application

```bash
streamlit run app.py
```

The application will open at the local Streamlit URL displayed in the terminal.

## Application Flow

```text
Paste / Select News Article
          ↓
Article Validation
          ↓
Text Cleaning
          ↓
TF-IDF Transformation
          ↓
Logistic Regression
          ↓
Prediction + Probability
          ↓
Result + Confidence
```

## Example

```text
Prediction: REAL
Confidence: 97.42%

Real Probability: 97.42%
Fake Probability: 2.58%
```

## Data Leakage Prevention

- Source/Reuters tags are removed.
- The `subject` column is not used as a feature.
- TF-IDF is fitted only on training data.
- Test data uses the already-fitted vectorizer.
- Duplicate articles are removed before model training.

## Limitations

- No independent external fact-checking
- English-news focused
- Performance depends on training data
- New writing styles or unseen topics may affect predictions
- High confidence does not guarantee factual correctness

## Future Improvements

- DistilBERT / Transformer models
- Hindi and Hinglish support
- Batch CSV prediction
- FastAPI backend
- Cloud deployment
- Explainable AI dashboard
- Larger and more diverse datasets
- External fact-checking integration

## Author

**Radhika Tyagi**

B.Tech Computer Science & Engineering

## Disclaimer

This project is intended for educational and research purposes. The prediction is a machine-learning classification result, not a definitive fact-checking verdict.
