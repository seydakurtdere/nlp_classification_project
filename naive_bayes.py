from collections import Counter, defaultdict
import math
from utils import generate_ngrams

#Naive Bayes, olasılık temelli bir makine öğrenimi algoritmasıdır. 
#Amacı, bir veri noktasını sınıflandırmak (örneğin, bir metni bir kategoriye atamak) için sınıf olasılıklarını hesaplamaktır. 

def train_naive_bayes(train_data, n, smoothing=1):
    """
    Naïve Bayes modelini eğitmek için n-gram'ları hesaplar ve model oluşturur.
    """
    class_ngram_counts = defaultdict(Counter)
    class_totals = defaultdict(int)

    for idx, row in train_data.iterrows():
        class_label = row['Class Index']
        ngrams = generate_ngrams(row['Cleaned Description'], n) #Her metin için n-gramlar oluşturulur
        for ngram in ngrams:
            class_ngram_counts[class_label][ngram] += 1
            class_totals[class_label] += 1

    return class_ngram_counts, class_totals

def classify_naive_bayes(test_data, class_ngram_counts, class_totals, n, smoothing=1):
    """
    Naïve Bayes modelini kullanarak test verilerini sınıflandırır.
    """
    predictions = [] #Sınıf tahminlerini saklamak için boş bir liste oluşturulur.
    vocab_size = sum(len(class_ngram_counts[class_label]) for class_label in class_ngram_counts) #Tüm sınıflarda farklı n gramların toplam sayısı hesaplanır.

    for _, row in test_data.iterrows():
        ngrams = generate_ngrams(row['Cleaned Description'], n) #Her bir test verisi işlenir ve metin, n-gram'lara bölünür.
        class_probs = {}

        for class_label in class_ngram_counts.keys(): 
            log_prob = 0 
            total_count = class_totals[class_label] 

            for ngram in ngrams:
                count = class_ngram_counts[class_label][ngram] + smoothing
                log_prob += math.log(count / (total_count + smoothing * vocab_size))

            class_probs[class_label] = log_prob

        predictions.append(max(class_probs, key=class_probs.get)) #En yüksek olasılığa sahip sınıf seçilir ve tahmin listesine eklenir.

    return predictions
