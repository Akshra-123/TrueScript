# Import Modules

import nltk  # natural language toolkit used for tasks related to natural language processing
# nltk.download("popular")  # Downloads popular datasets and pre-trained models for NLP tasks
import pandas as pd
import string
from nltk.corpus import stopwords

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

print(data['source_text'])
print(data["plagiarized_text"])
