import pandas as pd
import nltk
import string
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

data = pd.read_csv("loan_data.csv")
data['Approval'].replace(['Approved', 'Rejected'], [1,0], inplace=True)
data['Employment_Status'].replace(['employed', 'unemployed'], [1,0], inplace=True)

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

data['clean_text'] = data['Text'].apply(lambda x: finalpreprocess(x))

X_text = data['clean_text']
y = data['Approval']
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

numeric_cols = ['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status']
X_train_num = data.loc[X_train.index][numeric_cols]
X_test_num = data.loc[X_test.index][numeric_cols]

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

input_text = Input(shape=(X_train_tfidf.shape[1],))
input_num = Input(shape=(X_train_num_scaled.shape[1],))

text_branch = Dense(64, activation='sigmoid')(input_text)
text_branch = Dense(32, activation='sigmoid')(text_branch) 

num_branch = Dense(32, activation='relu')(input_num)
num_branch = Dense(16, activation='relu')(num_branch)

combined = Concatenate()([text_branch, num_branch])
combined = Dense(16, activation='sigmoid')(combined)
output = Dense(1, activation='sigmoid')(combined)

model_tf = Model(inputs=[input_text, input_num], outputs=output)
model_tf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_tf.fit([X_train_tfidf, X_train_num_scaled], y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=1)

y_pred_tf = (model_tf.predict([X_test_tfidf, X_test_num_scaled]) > 0.5).astype(int)

X_test_num_reset = X_test_num.reset_index(drop=True)
for i, row in X_test_num_reset.iterrows():
    if row['Loan_Amount'] >= 120000 or row['Credit_Score'] < 550 or row['DTI_Ratio'] > 50:
        y_pred_tf[i] = 0

# print("\nTensorFlow Classification Report:")
# print(classification_report(y_test, y_pred_tf))

accuracy_tf = accuracy_score(y_test, y_pred_tf)
print(f"TensorFlow Accuracy: {accuracy_tf:.3f}")

roc_auc = roc_auc_score(y_test, model_tf.predict([X_test_tfidf, X_test_num_scaled]))
print(f"ROC AUC: {roc_auc:.3f}")

model_tf.save("loan_model.keras")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training complete. Saving model and preprocessing tools...")