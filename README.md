<img width="1440" height="790" alt="Screenshot 2026-10-01 at 3 14 49 PM" src="https://github.com/user-attachments/assets/64467d51-f697-4a92-a3b3-0f79571b24ff" />
<img width="1440" height="786" alt="Screenshot 2026-10-01 at 3 15 10 PM" src="https://github.com/user-attachments/assets/8e4d6de1-53e1-4a56-81d6-c08257916b7c" />
<img width="1440" height="787" alt="Screenshot 2026-10-01 at 3 15 23 PM" src="https://github.com/user-attachments/assets/0a34dcd6-e2b7-46e6-8859-e34dae7565b5" />
<img width="1440" height="789" alt="Screenshot 2026-10-01 at 3 15 59 PM" src="https://github.com/user-attachments/assets/50874125-911a-4681-8040-a2cfd4f52878" />
<img width="1440" height="786" alt="Screenshot 2026-10-01 at 3 16 16 PM" src="https://github.com/user-attachments/assets/4d61357a-bb78-4009-91db-f03c246d015e" />
<img width="1440" height="787" alt="Screenshot 2026-10-01 at 3 16 32 PM" src="https://github.com/user-attachments/assets/85ddf4a1-29a9-4a7c-b576-21a1e7020bd1" />

# 📰 Fake News Detection Using NLP

A Machine Learning and Natural Language Processing project that classifies news articles as **Fake** or **Real** using text preprocessing, **TF-IDF feature extraction**, and **Logistic Regression**.

---

## 📌 Project Overview

The rapid spread of information on digital platforms makes it difficult to manually verify every news article.

This project aims to build an automated **Fake News Detection System** that analyzes the textual content of a news article and predicts whether it is:

* ✅ **Real News**
* ❌ **Fake News**

The project demonstrates a complete Machine Learning workflow:

```text
News Article
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Prediction
     ↓
REAL / FAKE
```

---

## 🎯 Objectives

* Understand how NLP can be applied to text classification.
* Clean and preprocess news article text.
* Convert text into numerical features using TF-IDF.
* Train a Machine Learning classification model.
* Predict whether an article is Real or Fake.
* Evaluate the model using classification metrics.

---

## 🛠️ Technologies Used

| Technology              | Purpose                 |
| ----------------------- | ----------------------- |
| 🐍 Python               | Programming language    |
| 📝 NLP                  | Text processing         |
| 📊 Pandas               | Data manipulation       |
| 🔢 NumPy                | Numerical operations    |
| 🤖 Scikit-learn         | Machine Learning        |
| 🔤 TF-IDF               | Text feature extraction |
| 📈 Logistic Regression  | Classification          |
| 📊 Matplotlib / Seaborn | Visualization           |

---

## 🔄 Project Workflow

### 1. Data Collection

The project uses a labeled news dataset containing examples of:

* Fake news
* Real news

Each article is assigned a corresponding class label.

---

### 2. Text Preprocessing

The raw news text is cleaned before being provided to the Machine Learning model.

Typical preprocessing steps include:

```text
Raw Text
   ↓
Lowercase Conversion
   ↓
Remove Unnecessary Characters
   ↓
Text Normalization
   ↓
Clean Text
```

This helps reduce noise and provides more consistent input to the model.

---

### 3. TF-IDF Vectorization

Machine Learning algorithms cannot directly understand raw text.

Therefore, **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert text into numerical vectors.

TF-IDF gives higher importance to words that are useful for distinguishing between documents.

Conceptually:

```text
Text
 ↓
TF-IDF
 ↓
Numerical Feature Vector
 ↓
Machine Learning Model
```

---

### 4. Model Training

The project uses **Logistic Regression** for binary classification.

The model learns patterns from the training data and predicts one of two classes:

```text
0 → Fake
1 → Real
```

The exact label mapping depends on the dataset used during training.

---

### 5. Prediction

After training, new/unseen news articles can be passed through the same preprocessing and TF-IDF pipeline.

Example:

```text
Input:
"Government announces a new economic policy..."

             ↓

      NLP Preprocessing

             ↓

        TF-IDF Vector

             ↓

     Logistic Regression

             ↓

       Prediction: REAL
```

---

## 📊 Model Evaluation

The model can be evaluated using several standard classification metrics.

### Accuracy

Measures the proportion of total predictions that are correct.

### Precision

Measures how many predicted positive examples are actually positive.

### Recall

Measures how many actual positive examples are correctly identified.

### F1 Score

The F1 score combines precision and recall using their harmonic mean.

```text
              Precision × Recall
F1 = 2 × ----------------------------
              Precision + Recall
```

These metrics provide a more complete view of model performance than accuracy alone.

---

## 📁 Project Structure

A typical project structure is:

```text
Fake_News_Detection_using_NLP/
│
├── archive/
│   ├── Fake.csv
│   └── True.csv
│
├── fake_news_detector/
│   ├── data/
│   ├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── README.md
├── requirements.txt
└── .gitignore
```

> Large dataset files may be excluded from GitHub using `.gitignore` to keep the repository lightweight.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/radhikatyagi388/Fake_news_detection_using_nlp.git
```

### 2. Move into the project directory

```bash
cd Fake_news_detection_using_nlp
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

After installing the dependencies:

1. Open the project in VS Code or Jupyter Notebook.
2. Load the dataset.
3. Run the preprocessing steps.
4. Train the TF-IDF + Logistic Regression pipeline.
5. Evaluate the model.
6. Provide a new news article for prediction.

---

## 🧠 Machine Learning Pipeline

The core approach can be represented as:

```text
                ┌─────────────────┐
                │   News Dataset  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Text Cleaning   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │    TF-IDF       │
                │  Vectorization  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Logistic        │
                │ Regression      │
                └────────┬────────┘
                         ↓
                 ┌───────────────┐
                 │ Prediction    │
                 └───────┬───────┘
                         ↓
                  REAL / FAKE
```

---

## 💡 Key Learnings

Through this project, I learned how to:

* Work with real-world text data.
* Perform NLP preprocessing.
* Convert text into numerical representations.
* Understand TF-IDF.
* Build a classification pipeline.
* Train Logistic Regression for text classification.
* Evaluate a Machine Learning model.
* Manage datasets and project files using Git/GitHub.
* Build an end-to-end NLP Machine Learning project.

---

## 🔮 Future Scope

The project can be extended by:

* Using **Support Vector Machines (SVM)**.
* Experimenting with **Random Forest** and other classifiers.
* Using **Word Embeddings**.
* Applying **BERT / Transformer models**.
* Using larger and more diverse datasets.
* Building a web-based prediction interface.
* Creating a real-time news analysis system.
* Adding explainability to show which textual features influenced a prediction.

---

## ⚠️ Limitations

A Machine Learning classifier does not independently verify whether a news claim is factually true.

Its prediction depends on:

* Quality of the training dataset
* Dataset size and diversity
* Text preprocessing
* Feature representation
* Model assumptions
* Similarity between training and unseen news

Therefore, predictions should be treated as **model classifications rather than definitive fact-checks**.

---

## 📚 Concepts Covered

```text
Natural Language Processing
        ↓
Text Preprocessing
        ↓
TF-IDF
        ↓
Feature Extraction
        ↓
Supervised Learning
        ↓
Logistic Regression
        ↓
Binary Classification
        ↓
Precision / Recall / F1
        ↓
Model Evaluation
```

---

## 👩‍💻 Author

**Radhika Tyagi**

Interested in:

**Artificial Intelligence • NLP • Machine Learning • RAG • Generative AI**

---

## ⭐ Project

If you find this project useful, feel free to explore the repository and learn from the implementation.

**Learning by building, one project at a time. 🚀**
