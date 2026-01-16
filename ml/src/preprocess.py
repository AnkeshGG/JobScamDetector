# ml/src/preprocess.py
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")
STOP = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\$?\d+(?:,\d{3})*(?:\.\d+)?", " money ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def lemmatize(text: str) -> str:
    doc = nlp(text)
    tokens = []
    for t in doc:
        if t.is_stop or t.is_punct or t.like_num:
            continue
        lemma = t.lemma_.strip()
        if lemma and lemma not in STOP:
            tokens.append(lemma)
    return " ".join(tokens)

def preprocess_text(text: str) -> str:
    return lemmatize(clean_text(text))
