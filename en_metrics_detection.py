# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 11:13:38 2025

@author: Amirhosein
"""

import nltk
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import textstat
import os

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Mock dataset (replace with actual PolitiFact/BuzzFeed dataset)
data = {
    'title': [
        "Hillary's ISIS Email Just Leaked & It's Worse Than Anyone Could Have Imagined",
        "Obama Injured in Explosion, Stock Market Crashes",
        "Scientists Discover New Planet in Our Solar System",
        "New Law Passed to Improve Healthcare Access"
    ],
    'body': [
        "Huffington Post is really running with this story from The Washington Post about the CIA confirming Russian interference in the presidential election. They're saying if 100% true, the courts can PUT HILLARY IN THE WHITE HOUSE!",
        "A fake report claimed Barack Obama was injured in an explosion, causing a $130B wipeout in stock value. This is absolutely shocking and unbelievable!",
        "Astronomers have confirmed the discovery of a new planet beyond Pluto, with potential for life. NASA is planning a mission to explore it.",
        "Congress passed a new law today aimed at improving healthcare access for millions of Americans, focusing on affordability and coverage."
    ],
    'label': [1, 1, 0, 0]  # 1: Fake, 0: True
}
df = pd.DataFrame(data)

# Validation function to compute credit score and metrics
def validate_input(uploaded_file, title, body):
    metrics = {}
    scores = []

    # Picture metrics
    metrics['picture'] = {}
    metrics['picture']['presence'] = 1 if uploaded_file else 0
    metrics['picture']['file_size_kb'] = 0
    if uploaded_file:
        # Get file size in KB
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell() / 1024  # Bytes to KB
        uploaded_file.seek(0)  # Reset file pointer
        metrics['picture']['file_size_kb'] = round(file_size, 2)
    # Picture score: +10 if present, +5 if size > 100KB (heuristic)
    picture_score = 10 * metrics['picture']['presence'] + 5 * (metrics['picture']['file_size_kb'] > 100)
    scores.append(picture_score / 15 * 20)  # Normalize to 0-20

    # Title metrics
    metrics['title'] = {}
    title_words = word_tokenize(title.lower())
    metrics['title']['length'] = len(title_words)
    # Sentiment polarity
    title_sentiment = sid.polarity_scores(title)
    metrics['title']['sentiment_polarity'] = title_sentiment['compound']
    # Sensationalist words
    sensational_words = ['shocking', 'unbelievable', 'leaked', 'secret', 'scandal', 'exposed']
    metrics['title']['sensationalist_count'] = sum(1 for word in title_words if word in sensational_words)
    # Title score: +10 if length > 5, -5 per sensational word, +5 if neutral sentiment
    title_score = 10 * (metrics['title']['length'] > 5) - 5 * metrics['title']['sensationalist_count']
    title_score += 5 * (abs(metrics['title']['sentiment_polarity']) < 0.3)
    title_score = max(0, min(15, title_score))  # Clamp to 0-15
    scores.append(title_score / 15 * 30)  # Normalize to 0-30

    # Body metrics
    metrics['body'] = {}
    body_words = word_tokenize(body.lower())
    metrics['body']['length'] = len(body_words)
    # Readability (Flesch-Kincaid grade level)
    metrics['body']['readability_fk'] = textstat.flesch_kincaid_grade(body)
    # Rhetorical markers proportion
    sentences = nltk.sent_tokenize(body)
    causation_count = len(re.findall(r'\bbecause\b|\bdue to\b', body, re.IGNORECASE))
    condition_count = len(re.findall(r'\bif\b|\bwhether\b', body, re.IGNORECASE))
    attribution_count = len(re.findall(r'\bsaid\b|\bstated\b|\bclaimed\b', body, re.IGNORECASE))
    total_words = len(body_words)
    metrics['body']['rhetorical_proportion'] = (causation_count + condition_count + attribution_count) / total_words if total_words > 0 else 0
    # Body score: +15 if length > 100, +10 if readability 6-12, +10 if rhetorical proportion 0.01-0.1
    body_score = 15 * (metrics['body']['length'] > 100)
    body_score += 10 * (6 <= metrics['body']['readability_fk'] <= 12)
    body_score += 10 * (0.01 <= metrics['body']['rhetorical_proportion'] <= 0.1)
    body_score = max(0, min(35, body_score))  # Clamp to 0-35
    scores.append(body_score / 35 * 50)  # Normalize to 0-50

    # Compute credit score (weighted sum)
    credit_score = sum(scores)
    credit_score = round(max(0, min(100, credit_score)), 2)

    return credit_score, metrics

# Function to extract lexicon-level features (standardized BOW)
def extract_lexicon_features(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    word_counts = Counter(tokens)
    total_words = sum(word_counts.values())
    if total_words == 0:
        return [0] * 10
    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform([text]).toarray()[0]
    standardized = [x / total_words for x in X]
    return standardized

# Function to extract discourse-level features (simplified)
def extract_discourse_features(text):
    sentences = nltk.sent_tokenize(text)
    causation_count = len(re.findall(r'\bbecause\b|\bdue to\b', text, re.IGNORECASE))
    condition_count = len(re.findall(r'\bif\b|\bwhether\b', text, re.IGNORECASE))
    attribution_count = len(re.findall(r'\bsaid\b|\bstated\b|\bclaimed\b', text, re.IGNORECASE))
    total_sentences = len(sentences)
    if total_sentences == 0:
        return [0, 0, 0]
    return [
        causation_count / total_sentences,
        condition_count / total_sentences,
        attribution_count / total_sentences
    ]

# Combine features
def extract_features(title, body):
    text = title + " " + body
    lexicon_features = extract_lexicon_features(text)
    discourse_features = extract_discourse_features(text)
    return lexicon_features + discourse_features

# Train model
def train_model():
    X = [extract_features(row['title'], row['body']) for _, row in df.iterrows()]
    y = df['label'].values
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

# Streamlit app
st.title("Fake News Detection")
st.write("Upload a thumbnail, enter a title and body text to check if the news is Fake or True.")

# Inputs
uploaded_file = st.file_uploader("Upload thumbnail image", type=["jpg", "png"])
title = st.text_input("News Title")
body = st.text_area("News Body Text")

# Display thumbnail
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Thumbnail", width=200)

# Prediction and validation
if st.button("Predict"):
    if title and body:
        # Extract features and predict
        features = extract_features(title, body)
        clf = train_model()
        prediction = clf.predict([features])[0]
        label = "Fake" if prediction == 1 else "True"
        st.write(f"Prediction: **{label}**")

        # Validate input and compute credit score
        credit_score, metrics = validate_input(uploaded_file, title, body)
        st.write(f"Credit Score: **{credit_score}/100** (Higher is more credible)")

        # Display metrics
        st.subheader("Validation Metrics")
        st.write("**Picture Metrics**")
        st.write(f"- Image Present: {'Yes' if metrics['picture']['presence'] else 'No'}")
        st.write(f"- File Size: {metrics['picture']['file_size_kb']} KB")

        st.write("**Title Metrics**")
        st.write(f"- Word Count: {metrics['title']['length']}")
        st.write(f"- Sentiment Polarity: {metrics['title']['sentiment_polarity']:.2f} (-1 to 1)")
        st.write(f"- Sensationalist Words: {metrics['title']['sensationalist_count']}")

        st.write("**Body Metrics**")
        st.write(f"- Word Count: {metrics['body']['length']}")
        st.write(f"- Flesch-Kincaid Grade Level: {metrics['body']['readability_fk']:.2f}")
        st.write(f"- Rhetorical Markers Proportion: {metrics['body']['rhetorical_proportion']:.4f}")
    else:
        st.error("Please provide both title and body text.")