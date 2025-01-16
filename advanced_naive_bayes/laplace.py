import math
from collections import defaultdict, Counter
from utils import generate_ngrams

#Laplace düzeltmesi, sıfır olasılık problemini çözmek için Naive Bayes algoritmasında kullanılan bir yöntemdir.

def laplace_smoothing(class_ngram_counts, class_totals, vocab_size, alpha=1):
    """
    Daha önce görülmeyen n-gram'lar için sıfır olmayan bir olasılık atar.
    """
    smoothed_probs = defaultdict(Counter)
    for class_label, ngram_counts in class_ngram_counts.items():
        total_count = class_totals[class_label]
        for ngram, count in ngram_counts.items():
            smoothed_probs[class_label][ngram] = (count + alpha) / (total_count + alpha * vocab_size) #laplace düzeltmesi hesaplaması
        smoothed_probs[class_label]["<unseen>"] = alpha / (total_count + alpha * vocab_size) #görülmeyen n-gram olasılıkları
    return smoothed_probs
def classify_with_laplace(test_data, laplace_probs, class_totals, n=1):
    """
    Laplace düzeltmesi uygulanmış Naive Bayes modelini kullanarak test verilerini sınıflandırır.
    """
    predictions = []
    for _, row in test_data.iterrows():
        ngrams = generate_ngrams(row['Cleaned Description'], n)
        class_probs = defaultdict(float)
        for class_label in laplace_probs.keys():
            log_prob = 0
            total_count = class_totals[class_label]
            for ngram in ngrams:
                prob = laplace_probs[class_label].get(ngram, laplace_probs[class_label]["<unseen>"])
                log_prob += math.log(prob)
            class_probs[class_label] = log_prob
        predictions.append(max(class_probs, key=class_probs.get)) #Tüm sınıflar arasında en yüksek logaritmik olasılığa sahip olan sınıf seçilir ve tahminler listesine eklenir.
    return predictions

