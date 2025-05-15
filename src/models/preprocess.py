import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


download_nltk_data()

def clean_text_advanced(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_data(input_path="data/processed/train.csv", output_path="data/processed/train_cleaned.csv"):
    df = pd.read_csv(input_path)
    df["cleaned_review"] = df["review_text"].apply(clean_text_advanced)
    df = df[["cleaned_review", "label"]]
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_data()