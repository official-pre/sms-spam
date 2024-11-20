import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vectorized)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the model and vectorizer
    joblib.dump(model, 'spam_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

def predict_spam(message):
    # Load the saved model and vectorizer
    model = joblib.load('spam_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')

    # Vectorize the input message
    message_vectorized = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_vectorized)
    return "spam" if prediction[0] == 1 else "ham"

if __name__ == "__main__":
    train_model()