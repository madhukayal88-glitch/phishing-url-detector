import argparse
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dense, Dropout, GlobalMaxPooling1D, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
MAX_LEN = 200          # max characters per URL considered
EMBED_DIM = 32
BATCH_SIZE = 64
EPOCHS = 12
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42
URL_COL_CANDIDATES = ["url", "urls", "URL", "Domain", "domain", "website"]
LABEL_COL_CANDIDATES = ["label", "Label", "status", "Result", "type", "Type"]
POSITIVE_TOKENS = {"1", "bad", "phishing", "malicious", "yes", "-1"}
NEGATIVE_TOKENS = {"0", "good", "legitimate", "benign", "no", "1_safe", "safe"}
def find_column(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a {kind} column. Columns present: {list(df.columns)}. "
        f"Rename your {kind} column to one of {candidates}."
    )
def normalize_label(value):
    v = str(value).strip().lower()
    if v in {t.lower() for t in POSITIVE_TOKENS}:
        return 1
    if v in {t.lower() for t in NEGATIVE_TOKENS}:
        return 0
    try:
        return int(float(v))
    except ValueError:
        raise ValueError(f"Unrecognised label value: {value!r}")
def clean_url(u):
    u = str(u).strip()
    u = re.sub(r"\s+", "", u)
    return u.lower()
def load_dataset(path):
    df = pd.read_csv(path)
    url_col = find_column(df, URL_COL_CANDIDATES, "URL")
    label_col = find_column(df, LABEL_COL_CANDIDATES, "label")
    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]
    df["url"] = df["url"].apply(clean_url)
    df["label"] = df["label"].apply(normalize_label)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    return df
def build_model(vocab_size):
    inp = Input(shape=(MAX_LEN,), name="url_input")
    x = Embedding(input_dim=vocab_size + 1, output_dim=EMBED_DIM, input_length=MAX_LEN)(inp)
    conv = Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv_out = GlobalMaxPooling1D()(conv)
    lstm_out = Bidirectional(LSTM(48))(x)
    merged = Concatenate()([conv_out, lstm_out])
    d = Dense(64, activation="relu")(merged)
    d = Dropout(0.4)(d)
    out = Dense(1, activation="sigmoid")(d)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model
def main(data_path, model_out, tokenizer_out):
    print(f"Loading dataset from {data_path} ...")
    df = load_dataset(data_path)
    print(f"Loaded {len(df)} URLs. Class balance:\n{df['label'].value_counts()}")
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(df["url"])
    vocab_size = len(tokenizer.word_index)
    print(f"Character vocabulary size: {vocab_size}")
    sequences = tokenizer.texts_to_sequences(df["url"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df["label"].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_STATE, stratify=y
    )
    rel_test = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    model = build_model(vocab_size)
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    print("\nEvaluating on held-out test set...")
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))
    model.save(model_out)
    with open(tokenizer_out, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"\nSaved model to {model_out}")
    print(f"Saved tokenizer to {tokenizer_out}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", help="Path to training CSV")
    parser.add_argument("--model_out", default="phishing_model.h5")
    parser.add_argument("--tokenizer_out", default="tokenizer.pkl")
    args = parser.parse_args()
    main(args.data, args.model_out, args.tokenizer_out)
