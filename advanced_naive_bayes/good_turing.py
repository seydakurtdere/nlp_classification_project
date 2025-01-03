import math
from collections import defaultdict, Counter
from utils import generate_ngrams

#Good-Turing Fonksiyonu (Sadece Görülmemiş N-gramlar için)
def good_turing_smoothing(class_ngram_counts, class_totals, vocab_size, ngrams):
    smoothed_probs = defaultdict(Counter)

    for class_label, ngram_counts in class_ngram_counts.items():
        total_count = class_totals[class_label]

        for ngram in ngrams:
            count = ngram_counts.get(ngram, 0)
            if count == 0:  # Sadece görülmemişler için
                smoothed_probs[class_label][ngram] = 1 / (total_count + vocab_size)
            else:
                smoothed_probs[class_label][ngram] = count / total_count

    return smoothed_probs

#Test Setini Sınıflandırma Fonksiyonu
def classify_with_good_turing(test_data, good_turing_probs, class_totals, n=1):
    predictions = []
    for _, row in test_data.iterrows():
        ngrams = generate_ngrams(row['Cleaned Description'], n)
        class_probs = defaultdict(float)

        for class_label in good_turing_probs.keys():
            log_prob = 0
            for ngram in ngrams:
                prob = good_turing_probs[class_label].get(ngram, 1e-7)
                log_prob += math.log(prob)
            class_probs[class_label] = log_prob

        predictions.append(max(class_probs, key=class_probs.get))
    return predictions

