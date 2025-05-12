# to run...$ streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import joblib
import nltk
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# Load model and preprocessing tools
model = load_model("loan_model.keras")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

def preprocess(text):
    text = text.lower() 
    text = text.strip()
    text = re.compile(r'[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d',' ',text) 
    return text

def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

wl = WordNetLemmatizer()
 
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) 
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in word_pos_tags] 
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))



# ------------------- User Interface ---------------------- #

st.title("Loan Approval Prediction App")

# add a qualifier!!! - format is a blue box
st.info("""**Disclaimer**: This model was trained on a publicly available Kaggle dataset.
        It may not generalize well to real-world loan applications and should not be used for actual financial decisions.""")

st.write("Fill out the form below to see if your loan is likely to be approved.")


text_input = st.text_area("What is this loan for?")
income = st.number_input("Income", min_value=0.0, value=10000.00)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=700)
loan_amount = st.number_input("Loan Amount", min_value=0.00, value=1500.00)
dti_ratio = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, value=20.0)
employment_status = st.selectbox("Employment Status", ["employed", "unemployed"])

if credit_score < 550:
    st.warning("Warning: Applicants with credit scores below 550 are rarely approved.")

if loan_amount > 120000:
    st.warning("Warning: Loan amounts above $120,000 are rarely approved.")

if dti_ratio > 50:
    st.warning("Warning: Applicants with DTI ratio of more than 50 are rarely approved.")

if employment_status == "unemployed":
    st.warning("Warning: Applicants who are unemployed are rarely approved")


if st.button("Predict Loan Approval"):
    # Preprocess text
    cleaned_text = finalpreprocess(text_input)
    tfidf_input = vectorizer.transform([cleaned_text]).toarray()

    # Prepare and scale numeric input
    num_input_raw = np.array([[income, credit_score, loan_amount, dti_ratio, 1 if employment_status == "employed" else 0]])
    num_input_scaled = scaler.transform(num_input_raw)

    # Model prediction
    prediction = model.predict([tfidf_input, num_input_scaled])[0][0]
    approved = prediction > 0.5

    # Output
    st.subheader("Prediction Result")
    st.write("**Approved**" if approved else "**Rejected**")
    st.write(f"**Confidence Score:** `{prediction:.3f}`")