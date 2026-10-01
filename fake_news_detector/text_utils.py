import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Clean ISOT news text without using subject/source metadata."""
    text = str(text).lower()

    # Remove common Reuters/source tags that can create leakage.
    text = re.sub(r"\(\s*reuters\s*\)", " ", text, flags=re.I)

    # Remove URLs, HTML, numbers and punctuation.
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = []
    for word in text.split():
        if len(word) < 3 or word in STOP_WORDS:
            continue
        words.append(LEMMATIZER.lemmatize(word))

    return " ".join(words)
