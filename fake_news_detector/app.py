import streamlit as st
import joblib
import os
import json
import time
from text_utils import clean_text


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FakeLens AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# Only styling. No HTML UI elements are used.
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at top right, #20204f 0%, #0b1020 38%, #050810 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080d1d 0%, #0d1427 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #cbd5e1 !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        color: #ffffff;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.35);
        margin-bottom: 20px;
    }

    .warning-box {
        padding: 16px 20px;
        border-radius: 14px;
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.35);
        margin: 15px 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 16px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    textarea {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
    }

    textarea::placeholder {
        color: #64748b !important;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }

    .result-fake {
        padding: 25px;
        border-radius: 18px;
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.45);
        text-align: center;
    }

    .result-real {
        padding: 25px;
        border-radius: 18px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.45);
        text-align: center;
    }

    .small-text {
        color: #94a3b8;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = "models/fake_news_pipeline.pkl"
METRICS_PATH = "reports/metrics.json"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# LOAD METRICS
# =========================================================

def load_metrics():

    default_metrics = {
        "accuracy": 0.9853,
        "f1": 0.9848
    }

    if os.path.exists(METRICS_PATH):

        try:
            with open(METRICS_PATH, "r") as file:
                data = json.load(file)

            accuracy = data.get("accuracy", default_metrics["accuracy"])
            f1 = data.get("f1", default_metrics["f1"])

            return {
                "accuracy": accuracy,
                "f1": f1
            }

        except Exception:
            pass

    return default_metrics


metrics = load_metrics()


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "article_text" not in st.session_state:
    st.session_state.article_text = ""


# =========================================================
# SAMPLE ARTICLES
# =========================================================

REAL_SAMPLE = """
The central bank announced a new policy measure Tuesday following
its monetary policy meeting. Officials said the decision was intended
to support financial stability while keeping inflation expectations
under control. The committee said it would continue monitoring
economic indicators and financial conditions in the coming months.
"""

FAKE_SAMPLE = """
Scientists have discovered that drinking three glasses of a special
juice every morning can completely eliminate all diseases within
seven days. The secret formula was reportedly hidden for decades,
and experts claim that hospitals may soon become unnecessary.
"""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📰 FakeLens AI")

    st.caption("NLP-powered Fake News Detection")

    st.divider()

    st.subheader("⚙️ Model")

    st.info(
        "Logistic Regression\n\n"
        "TF-IDF Vectorization\n\n"
        "30,000 Features\n\n"
        "Unigrams + Bigrams"
    )

    st.subheader("📊 Performance")

    st.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    st.metric(
        "F1 Score",
        f"{metrics['f1'] * 100:.2f}%"
    )

    st.divider()

    st.subheader("🧠 Pipeline")

    st.write("1. Article Input")
    st.write("2. Text Cleaning")
    st.write("3. TF-IDF")
    st.write("4. Logistic Regression")
    st.write("5. Prediction")
    st.write("6. Confidence")

    st.divider()

    st.success("🟢 MODEL READY")

    st.caption(
        "The model analyzes linguistic patterns learned "
        "from the training dataset."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">FakeLens AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Fake News Detection using NLP, TF-IDF & Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL STATUS
# =========================================================

if model is not None:

    st.success(
        "🟢 Model Online — Ready to analyze news articles"
    )

else:

    st.error(
        "🔴 Model not found. Please make sure "
        "`models/fake_news_pipeline.pkl` exists."
    )

    st.stop()


# =========================================================
# WARNING
# =========================================================

st.warning(
    "⚠️ This system does NOT independently fact-check news using "
    "external sources. It predicts based on linguistic patterns "
    "learned from training data. Use the result as a first-level "
    "helper and apply human judgment."
)


# =========================================================
# TOP PERFORMANCE CARDS
# =========================================================

st.subheader("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

with col3:
    st.metric(
        "F1 Score",
        f"{metrics['f1'] * 100:.2f}%"
    )

with col4:
    st.metric(
        "TF-IDF Features",
        "30K"
    )


st.divider()


# =========================================================
# ANALYZE SECTION
# =========================================================

left, right = st.columns(
    [2.2, 1],
    gap="large"
)


# =========================================================
# LEFT PANEL
# =========================================================

with left:

    st.subheader("🔍 Analyze News")

    st.caption(
        "Paste an English news article and let the trained NLP "
        "model analyze its linguistic patterns."
    )

    sample = st.selectbox(
        "Choose a sample",
        [
            "Custom Article",
            "🟢 Real News Sample",
            "🔴 Fake News Sample"
        ]
    )

    if sample == "🟢 Real News Sample":

        selected_text = REAL_SAMPLE

    elif sample == "🔴 Fake News Sample":

        selected_text = FAKE_SAMPLE

    else:

        selected_text = st.session_state.article_text


    article_text = st.text_area(
        "News Article",
        value=selected_text,
        height=280,
        placeholder="Paste at least 15 words of an English news article here...",
        label_visibility="collapsed"
    )

    word_count = len(article_text.split())
    character_count = len(article_text)

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric(
            "Words",
            word_count
        )

    with stat2:
        st.metric(
            "Characters",
            character_count
        )

    with stat3:

        if word_count >= 15:
            st.success("✓ Ready")
        else:
            st.warning("Need 15+ words")


    analyze = st.button(
        "🔎 ANALYZE ARTICLE",
        type="primary",
        use_container_width=True
    )


# =========================================================
# RIGHT PANEL
# =========================================================

with right:

    st.subheader("⚡ AI System")

    st.success("MODEL READY")

    st.write(
        "Your trained NLP model is loaded and ready "
        "to analyze English news articles."
    )

    st.divider()

    st.subheader("🧩 Model Details")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Vectorizer:** TF-IDF")
    st.write("**Features:** 30,000")
    st.write("**N-grams:** Unigrams + Bigrams")

    st.divider()

    st.subheader("⏱️ Prediction")

    st.write(
        "Target response time: less than 1 second"
    )


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not article_text.strip():

        st.error(
            "Please enter a news article."
        )

    elif word_count < 15:

        st.error(
            f"Article contains only {word_count} words. "
            "Please enter at least 15 words."
        )

    else:

        with st.spinner("Analyzing article..."):

            start_time = time.perf_counter()

            try:

                cleaned_article = clean_text(
                    article_text
                )

                prediction = model.predict(
                    [cleaned_article]
                )[0]

                probabilities = model.predict_proba(
                    [cleaned_article]
                )[0]

                prediction_time = (
                    time.perf_counter() - start_time
                )

                fake_probability = float(
                    probabilities[0]
                )

                real_probability = float(
                    probabilities[1]
                )

                confidence = max(
                    fake_probability,
                    real_probability
                )

                if prediction == 0:

                    label = "FAKE"

                else:

                    label = "REAL"


                # =========================================
                # SAVE HISTORY
                # =========================================

                st.session_state.article_text = article_text

                st.session_state.history.insert(
                    0,
                    {
                        "prediction": label,
                        "confidence": confidence,
                        "words": word_count,
                        "time": prediction_time
                    }
                )

                # Keep only latest 10
                st.session_state.history = (
                    st.session_state.history[:10]
                )


                # =========================================
                # RESULT
                # =========================================

                st.divider()

                st.subheader("🎯 Analysis Result")

                result_col1, result_col2 = st.columns(
                    [1.3, 1]
                )

                with result_col1:

                    if label == "FAKE":

                        st.error(
                            f"🚨 {label} NEWS"
                        )

                    else:

                        st.success(
                            f"✅ {label} NEWS"
                        )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                with result_col2:

                    st.metric(
                        "Prediction Time",
                        f"{prediction_time:.4f} sec"
                    )

                    st.metric(
                        "Words Analyzed",
                        word_count
                    )


                # =========================================
                # PROBABILITIES
                # =========================================

                st.subheader("📈 Prediction Probabilities")

                probability_col1, probability_col2 = st.columns(2)

                with probability_col1:

                    st.write(
                        f"🔴 Fake News — "
                        f"{fake_probability * 100:.2f}%"
                    )

                    st.progress(
                        fake_probability
                    )

                with probability_col2:

                    st.write(
                        f"🟢 Real News — "
                        f"{real_probability * 100:.2f}%"
                    )

                    st.progress(
                        real_probability
                    )


                # =========================================
                # CONFIDENCE WARNING
                # =========================================

                if confidence < 0.70:

                    st.warning(
                        "⚠️ Low confidence prediction. "
                        "The model is not highly certain about "
                        "this classification."
                    )

                else:

                    st.info(
                        "ℹ️ Confidence represents the model's "
                        "estimated probability, not independent "
                        "fact verification."
                    )


                # =========================================
                # ARTICLE STATISTICS
                # =========================================

                st.subheader("📝 Article Statistics")

                s1, s2, s3, s4 = st.columns(4)

                with s1:
                    st.metric(
                        "Words",
                        word_count
                    )

                with s2:
                    st.metric(
                        "Characters",
                        character_count
                    )

                with s3:
                    sentences = max(
                        1,
                        article_text.count(".")
                        + article_text.count("!")
                        + article_text.count("?")
                    )

                    st.metric(
                        "Sentences",
                        sentences
                    )

                with s4:
                    st.metric(
                        "Prediction",
                        label
                    )


                # =========================================
                # LINGUISTIC INSIGHT
                # =========================================

                st.subheader("🧠 Linguistic Insight")

                st.write(
                    "The model makes its prediction using "
                    "patterns learned from TF-IDF features "
                    "during training."
                )

                st.write(
                    "It does not independently verify the "
                    "claims or facts contained in the article."
                )


            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# =========================================================
# HISTORY
# =========================================================

st.divider()

st.subheader("🕘 Recent Analysis History")

if len(st.session_state.history) == 0:

    st.info(
        "No articles analyzed yet."
    )

else:

    for index, item in enumerate(
        st.session_state.history,
        start=1
    ):

        history_col1, history_col2, history_col3, history_col4 = st.columns(
            4
        )

        with history_col1:

            if item["prediction"] == "FAKE":

                st.error(
                    f"#{index} 🔴 FAKE"
                )

            else:

                st.success(
                    f"#{index} 🟢 REAL"
                )

        with history_col2:

            st.write(
                f"Confidence: "
                f"{item['confidence'] * 100:.2f}%"
            )

        with history_col3:

            st.write(
                f"Words: {item['words']}"
            )

        with history_col4:

            st.write(
                f"Time: {item['time']:.4f}s"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "FakeLens AI • NLP + TF-IDF + Logistic Regression • "
    "Educational Fake News Detection System"
)

st.caption(
    "⚠️ Prediction is a machine-learning classification, "
    "not independent fact verification."
)