import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from preprocess import clean_text
import os

def train_model():
    # Load dataset
    df = pd.read_csv("data/data.csv")
    df = df.dropna()

    # Balance the dataset
    min_count = df['Sentiment'].value_counts().min()

    balanced_df = (
        df.groupby('Sentiment')
        .apply(lambda x: x.sample(min_count, random_state=42))
        .reset_index(drop=True)
    )

    balanced_df['clean'] = balanced_df['Sentence'].apply(clean_text)
    print("✅ Balanced dataset shape:", balanced_df.shape)
    print("🟰 New label distribution:\n", balanced_df['Sentiment'].value_counts())
    print("✅ Dataset loaded with shape:", df.shape)
    print("🧾 Labels:", df['Sentiment'].value_counts())
    print("🔍 Class distribution:\n", df['Sentiment'].value_counts())


    # Split
    X_train, X_test, y_train, y_test = train_test_split(balanced_df['clean'], balanced_df['Sentiment'], test_size=0.2, random_state=42)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    print("=== Evaluation Report ===")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    print("Model trained and saved to models/model.pkl")

train_model()