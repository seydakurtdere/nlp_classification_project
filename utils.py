import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def clean_text(text):
    """
    Metni temizlemek için kullanılan fonksiyon:
    1. Küçük harfe dönüştürme
    2. Noktalama işaretlerini kaldırma
    3. Gereksiz boşlukları temizleme
    """
    text = text.lower()  # Küçük harfe dönüştür
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları temizle
    return text

def generate_ngrams(text, n):
    """
    Verilen bir metin için n-gram'ları oluşturur.
    """
    words = text.split()  # Metni kelimelere ayır
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def build_vocabulary(train_data, n=1):
    """
    Eğitim verisinden n-gram'ları çıkararak bir vocabulary oluşturur.
    """
    vocabulary = set()
    for _, row in train_data.iterrows():
        ngrams = generate_ngrams(row['Cleaned Description'], n)
        vocabulary.update(ngrams)
    return vocabulary

def generate_ngrams_from_corpus(corpus, n):
    ngrams = []
    for text in corpus:
        generated = generate_ngrams(text, n)
        ngrams.extend(generated)
    return ngrams

