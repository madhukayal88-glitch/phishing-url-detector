import pickle
import re
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200
MODEL_PATH = "phishing_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
st.set_page_config(page_title="Phishing URL Detector", layout="centered")
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer
def clean_url(u):
    u = str(u).strip()
    u = re.sub(r"\s+", "", u)
    return u.lower()
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "confirm",
                     "signin", "banking", "alert", "suspend", "password"]
BAD_TLDS = [".xyz", ".tk", ".top", ".club", ".work", ".loan", ".gq", ".ml"]
def explain(url):
    host = url.replace("https://", "").replace("http://", "").split("/")[0]
    notes = []
    if not url.startswith("https://"):
        notes.append("No HTTPS")
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", host):
        notes.append("Domain is a raw IP address")
    if "@" in url:
        notes.append("Contains '@' redirection trick")
    if len(url) > 60:
        notes.append(f"Unusually long URL ({len(url)} chars)")
    if host.count("-") >= 2:
        notes.append("Multiple hyphens in domain")
    if any(host.endswith(t) for t in BAD_TLDS):
        notes.append("High-risk top-level domain")
    if any(w in url.lower() for w in SUSPICIOUS_WORDS):
        notes.append("Contains urgency/login-style keywords")
    return notes
st.title("🎣 Phishing URL Detector")
st.caption("Character-level CNN + BiLSTM model — paste a URL to check it.")
try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error(
        "Couldn't load the trained model. Run `python train_model.py --data data.csv` "
        "first so phishing_model.h5 and tokenizer.pkl exist in this folder."
    )
    st.stop()
url_input = st.text_input("Enter a URL", placeholder="e.g. paypa1-login-alert.xyz/verify")
if st.button("Check URL") and url_input.strip():
    cleaned = clean_url(url_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    is_phishing = prob >= 0.5
    confidence = prob if is_phishing else (1 - prob)
    if is_phishing:
        st.error(f"Phishing suspected — {confidence*100:.1f}% confidence")
    else:
        st.success(f"Looks safe — {confidence*100:.1f}% confidence")
    notes = explain(cleaned)
    if notes:
        st.subheader("Signals noticed")
        for n in notes:
            st.write(f"- {n}")
    else:
        st.write("No obvious red-flag signals in the URL structure.")
st.divider()
st.caption(
    "Model trained on labelled phishing/legitimate URLs using a character-level "
    "CNN-BiLSTM hybrid. This is a project demo, not a production security tool."
)
