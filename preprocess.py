import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    words = text.strip().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
