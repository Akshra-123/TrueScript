# Import Modules

import nltk  # natural language toolkit used for tasks related to natural language processing
# nltk.download("popular")  # Downloads popular datasets and pre-trained models for NLP tasks
import pandas as pd

# Load Dataset
data=pd.read_csv("C:\\TrueScript\\dataset.csv")
# print(data.head())

# Checking distribution of label function
print(data['label'].value_counts())
