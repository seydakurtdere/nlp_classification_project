from collections import defaultdict
import math
from utils import generate_ngrams

#Interpolasyon, farklı düzeylerdeki (örneğin, unigram, bigram, trigram) n-gram'ların 
#olasılıklarını birleştirerek daha güçlü bir dil modeli oluşturur.

def interpolate_ngrams(class_ngram_counts_1, class_ngram_counts_2, class_ngram_counts_3, 
                       class_totals_1, class_totals_2, class_totals_3, weights):
    if sum(weights) != 1: #Verilen ağırlıkların toplamının 1 olması gerekir.
        raise ValueError("Weights must sum to 1.")
        
    interpolated_probs = defaultdict(lambda: defaultdict(float)) #Her sınıf için interpolasyon yapılmış olasılıkları saklayan bir sözlük.
    
    all_classes = class_ngram_counts_1.keys()
    
    for class_label in all_classes:
        all_ngrams = (set(class_ngram_counts_1[class_label].keys()) | 
                      set(class_ngram_counts_2[class_label].keys()) | 
                      set(class_ngram_counts_3[class_label].keys()))
        
        for ngram in all_ngrams: #interpolasyon hesaplamaları
            prob_1 = class_ngram_counts_1[class_label].get(ngram, 1e-7) / class_totals_1[class_label]
            prob_2 = class_ngram_counts_2[class_label].get(ngram, 1e-7) / class_totals_2[class_label]
            prob_3 = class_ngram_counts_3[class_label].get(ngram, 1e-7) / class_totals_3[class_label]
            
            interpolated_probs[class_label][ngram] = (
                weights[0] * prob_1 + 
                weights[1] * prob_2 + 
                weights[2] * prob_3
            )
        
        interpolated_probs[class_label]["<unseen>"] = 1e-7 #Daha önce görülmemiş n-gram'lar için çok küçük bir olasılık atanır.

    return interpolated_probs


def classify_with_interpolation(test_data, interpolated_probs, class_labels, n=1):
    predictions = []
    
    for _, row in test_data.iterrows():
        ngrams = generate_ngrams(row['Cleaned Description'], n)
        class_probs = defaultdict(float)
        
        for class_label in class_labels:
            log_prob = 0
            for ngram in ngrams:
                prob = interpolated_probs[class_label].get(ngram, interpolated_probs[class_label]["<unseen>"])
                log_prob += math.log(prob)
            class_probs[class_label] = log_prob
        
        predictions.append(max(class_probs, key=class_probs.get))
    
    return predictions

