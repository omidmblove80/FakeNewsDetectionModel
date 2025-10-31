# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:08:22 2025

@author: Amirhosein
"""

import os, re, joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# TRAINING PHASE
# -------------------------------

def clean(x):
    if pd.isna(x): return ""
    x = re.sub(r"http\\S+", "", str(x))
    return re.sub(r"\\s+", " ", x.strip())

def train_and_save():
    # Load dataset (expects Fake.csv and True.csv in same folder)
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake['label'] = 1
    true['label'] = 0

    df = pd.concat([fake[['title','text','label']], true[['title','text','label']]], ignore_index=True)
    df['title'] = df['title'].astype(str).apply(clean)
    df['text'] = df['text'].astype(str).apply(clean)
    df['full'] = df['title'] + " " + df['text']

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words="english")
    X = vectorizer.fit_transform(df['full'])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
  #  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["True","Fake"], yticklabels=["True","Fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save artifacts
    joblib.dump(clf, "model_en.pkl")
    joblib.dump(vectorizer, "vectorizer_tfidf.pkl")
    print("Artifacts saved: model_en.pkl, vectorizer_tfidf.pkl")

# Train only if artifacts don’t exist
if not os.path.exists("model_en.pkl") or not os.path.exists("vectorizer_tfidf.pkl"):
    train_and_save()

# -------------------------------
# STREAMLIT APP
# -------------------------------

@st.cache_resource
def load_artifacts():
    clf = joblib.load("model_en.pkl")
    vectorizer = joblib.load("vectorizer_tfidf.pkl")
    return clf, vectorizer

clf, vectorizer = load_artifacts()

st.set_page_config(page_title="English Fake News Detector", layout="centered")
st.title("📰 English Fake News Detector")
st.write("Enter a news **title** and **body** to classify as Fake or True.")

title = st.text_area("News Title")
body = st.text_area("News Body")

if st.button("Classify"):
    text = clean(title + " " + body)
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0,1]
    pred = 1 if proba >= 0.5 else 0
    if pred == 1:
        st.error(f"⚠️ Fake News (probability {proba:.2f})")
    else:
        st.success(f"✅ True News (probability {1-proba:.2f})")
