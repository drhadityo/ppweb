import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('punkt', download_dir='C:/nltk_data')

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Main function to run the Streamlit app
def main():
    st.title("Text Preprocessing and Classification App")
    st.write("Aplikasi ini merupakan menggunakan preprocesses text dan membuat prediksi menggunakan pre-trained Logistic Regression Model.")

    # Load the pre-trained Logistic Regression model
    with open('app/logistic_regression_model.pkl', 'rb') as f:
        logistic_regression_model = pickle.load(f)

    # Load the pre-trained TfidfVectorizer
    with open('app/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Get user input
    text_input = st.text_input("Enter some news text:")
    if text_input:
        # Preprocess the text
        preprocessed_text = preprocess_text(text_input)

        # Vectorize the preprocessed text
        X_text = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = logistic_regression_model.predict(X_text)

        # Display prediction
        if prediction[0] == 1:
            st.write("Prediksi Berita: Travel News")
        else:
            st.write("Prediksi Berita: Health News")

# Run the app
if __name__ == '__main__':
    main()
