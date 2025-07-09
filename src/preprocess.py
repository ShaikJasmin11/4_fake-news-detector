# src/preprocess.py

import pandas as pd
import string
import re

def load_and_label_data(fake_path='data/fake.csv', true_path='data/true.csv'):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0  # fake
    true_df['label'] = 1  # real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df[['text', 'label']]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text
