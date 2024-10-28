# Import Modules

import nltk  # natural language toolkit used for tasks related to natural language processing
# nltk.download("popular")  # Downloads popular datasets and pre-trained models for NLP tasks
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle

# Load Dataset

data=pd.read_csv("C:\\TrueScript\\dataset.csv")
# print(data.head())

# Checking distribution of label function

print(data['label'].value_counts())

# Clean Text

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text
data["source_text"] = data["source_text"].apply(preprocess_text)
data["plagiarized_text"] = data["plagiarized_text"].apply(preprocess_text)

# Vectorisation

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data["source_text"] + " " + data["plagiarized_text"])
y = data["label"]

# print(X)
# print(y)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix")
print(cm)

# Random Forest Model

model = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate classification report
classification_rep = classification_report(y_test, y_pred)
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Print results
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

# Naive Bayes -this algorithm is particularly suited for text classification problems

model = MultinomialNB()
# Fit the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate classification report
classification_rep = classification_report(y_test, y_pred)
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Print results
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

# SVM -used for classification by finding the hyperplane that best separates the two classes (plagiarized vs. non-plagiarized texts)

model = SVC(kernel='linear', random_state=42)
# Fit the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate classification report
classification_rep = classification_report(y_test, y_pred)
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

# Save SVM and Vectoriser 
pickle.dump(model,open("model.pkl",'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl','wb'))

# Load Model and Vectorizer
model = pickle.load(open('model.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

# Detection System
def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarim Detected" if result[0] == 1 else "No Plagiarism"

# example ( it is a plagarized text)
input_text = 'Researchers have discovered a new species of butterfly in the Amazon rainforest.'
print(detect(input_text))

# example 
input_text = 'In recent years, artificial intelligence (AI) has revolutionized many fields, including healthcare, finance, and even entertainment. AI systems are now able to diagnose illnesses, forecast market trends, and generate music or art. However, these advancements also bring up significant ethical concerns, such as the risk of job loss and the difficulty in ensuring fairness and openness. As AI technology advances, society must balance innovation with accountability, aiming to make the benefits of AI accessible while reducing any negative impacts.'
print(detect(input_text))
