import math
from collections import defaultdict, Counter
from utils import generate_ngrams

def laplace_smoothing(class_ngram_counts, class_totals, vocab_size, alpha=1):
    smoothed_probs = defaultdict(Counter)
    for class_label, ngram_counts in class_ngram_counts.items():
        total_count = class_totals[class_label]
        for ngram, count in ngram_counts.items():
            smoothed_probs[class_label][ngram] = (count + alpha) / (total_count + alpha * vocab_size)
        smoothed_probs[class_label]["<unseen>"] = alpha / (total_count + alpha * vocab_size)
    return smoothed_probs

# Test Setini Sınıflandırma
def classify_with_laplace(test_data, laplace_probs, class_totals, n=1):
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
        predictions.append(max(class_probs, key=class_probs.get))
    return predictions

