import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def create_ann_model(X_train, y_train, hidden_layer_sizes=(10,)):
    """
    Basit bir yapay sinir ağı (ANN) modeli oluşturur ve eğitir.
    """
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_tfidf(train_texts, test_texts, max_features):
    """
    TF-IDF çevirimi: Eğitim ve test verileri için TF-IDF vektörleri oluşturur.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer
