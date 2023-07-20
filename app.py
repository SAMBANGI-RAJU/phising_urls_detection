import streamlit as st
from joblib import load
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import pandas as pd
import numpy as np


def main():
    
    st.title("Phishing URL Detection")
    st.write("Enter the URL to check if it is a phishing URL or not:")

    url_inp = st.text_input("URL")

    if st.button("Check"):
        if not url_inp:
            st.warning("Please enter a URL")
        else:
            result = detect_phishing_url(url_inp)
            if result:
                st.error("This URL is identified as a phishing URL!")
            else:
                st.success("This URL is safe.")


def detect_phishing_url(url_inp):
    # Preprocessing the input URL
    url_inp = re.sub(r"/|\.", " ", url_inp)
    url_inp = re.sub(r":", " ", url_inp)

    # Loading the dataset
    df = pd.read_csv("Datathon_train.csv")
    url_list = df['url']

    # Loading the trained model
    model = pickle.load(open("Mb.pkl", "rb"))

    # Vectorizing the URL
    vectorizer = TfidfVectorizer()
    vectorizer.fit(url_list)
    transformed_url = vectorizer.transform([url_inp])

    # Predicting using the loaded model
    prediction = model.predict(transformed_url)

    return prediction[0]


if __name__ == "__main__":
    main()
