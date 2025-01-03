import pandas as pd
from utils import clean_text, generate_ngrams,build_vocabulary, generate_ngrams_from_corpus
from naive_bayes import train_naive_bayes, classify_naive_bayes
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from advanced_naive_bayes.laplace import laplace_smoothing, classify_with_laplace
from advanced_naive_bayes.good_turing import good_turing_smoothing, classify_with_good_turing
from interpolation import interpolate_ngrams, classify_with_interpolation
from itertools import product
from ann_model import create_ann_model, prepare_tfidf

# Verileri yükle
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Metin temizleme
train_data['Cleaned Description'] = train_data['Description'].apply(clean_text)
test_data['Cleaned Description'] = test_data['Description'].apply(clean_text)

# İlk 5 temizlenmiş metni görüntüle
print("Eğitim Verisinden Temizlenmiş Örnekler:")
print(train_data[['Description', 'Cleaned Description']].head())

# Naïve Bayes için n-gram'ları hesaplayarak modeli eğit
print("1-gram modeli eğitiliyor...")
class_ngram_counts_1, class_totals_1 = train_naive_bayes(train_data, n=1)
print("2-gram modeli eğitiliyor...")
class_ngram_counts_2, class_totals_2 = train_naive_bayes(train_data, n=2)
print("3-gram modeli eğitiliyor...")
class_ngram_counts_3, class_totals_3 = train_naive_bayes(train_data, n=3)

# Test setini sınıflandır
print("1-gram ile sınıflandırma yapılıyor...")
predictions_1 = classify_naive_bayes(test_data, class_ngram_counts_1, class_totals_1, n=1)
print("2-gram ile sınıflandırma yapılıyor...")
predictions_2 = classify_naive_bayes(test_data, class_ngram_counts_2, class_totals_2, n=2)
print("3-gram ile sınıflandırma yapılıyor...")
predictions_3 = classify_naive_bayes(test_data, class_ngram_counts_3, class_totals_3, n=3)

# Sonuçları değerlendirme
print("1-gram değerlendirme:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_1))
print("Recall:", recall_score(test_data['Class Index'], predictions_1, average=None))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_1))

print("2-gram değerlendirme:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_2))
print("Recall:", recall_score(test_data['Class Index'], predictions_2, average=None))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_2))

print("3-gram değerlendirme:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_3))
print("Recall:", recall_score(test_data['Class Index'], predictions_3, average=None))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_3))

#Laplace Yaklaşımı ile sınıflandırma
vocabulary_1 = build_vocabulary(train_data, n=1)
vocabulary_2 = build_vocabulary(train_data, n=2)
vocabulary_3 = build_vocabulary(train_data, n=3)
ngrams_1 = generate_ngrams_from_corpus(test_data['Cleaned Description'], n=1)
ngrams_2 = generate_ngrams_from_corpus(test_data['Cleaned Description'], n=2)
ngrams_3 = generate_ngrams_from_corpus(test_data['Cleaned Description'], n=3)

# Laplace Smoothing Hesaplama
laplace_probs_1 = laplace_smoothing(class_ngram_counts_1, class_totals_1, len(vocabulary_1))
laplace_probs_2 = laplace_smoothing(class_ngram_counts_2, class_totals_2, len(vocabulary_2))
laplace_probs_3 = laplace_smoothing(class_ngram_counts_3, class_totals_3, len(vocabulary_3))

# **5. Good-Turing Smoothing Hesaplama 
good_turing_probs_1 = good_turing_smoothing(class_ngram_counts_1, class_totals_1, len(vocabulary_1), ngrams_1)
good_turing_probs_2 = good_turing_smoothing(class_ngram_counts_2, class_totals_2, len(vocabulary_2), ngrams_2)
good_turing_probs_3 = good_turing_smoothing(class_ngram_counts_3, class_totals_3, len(vocabulary_3), ngrams_3)

#Laplace ile Sınıflandırma ve Sonuç Değerlendirme
predictions_laplace_1 = classify_with_laplace(test_data, laplace_probs_1, class_totals_1, n=1)
predictions_laplace_2 = classify_with_laplace(test_data, laplace_probs_2, class_totals_2, n=2)
predictions_laplace_3 = classify_with_laplace(test_data, laplace_probs_3, class_totals_3, n=3)

#Good-Turing ile Sınıflandırma ve Sonuç Değerlendirme
predictions_good_turing_1 = classify_with_good_turing(test_data, good_turing_probs_1, class_totals_1, n=1)
predictions_good_turing_2 = classify_with_good_turing(test_data, good_turing_probs_2, class_totals_2, n=2)
predictions_good_turing_3 = classify_with_good_turing(test_data, good_turing_probs_3, class_totals_3, n=3)

#Laplace Sonuçları Değerlendirme
print("Laplace Smoothing Sonuçları - 1-gram:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_laplace_1))
print("Recall:", recall_score(test_data['Class Index'], predictions_laplace_1, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_laplace_1))

print("\nLaplace Smoothing Sonuçları - 2-gram:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_laplace_2))
print("Recall:", recall_score(test_data['Class Index'], predictions_laplace_2, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_laplace_2))

print("\nLaplace Smoothing Sonuçları - 3-gram:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_laplace_3))
print("Recall:", recall_score(test_data['Class Index'], predictions_laplace_3, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_laplace_3))

#Good-Turing Sonuçları Değerlendirme
print("Good-Turing Smoothing Sonuçları - 1-gram:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_good_turing_1))
print("Recall:", recall_score(test_data['Class Index'], predictions_good_turing_1, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_good_turing_1))

print("\nGood-Turing Smoothing Sonuçları - 2-gram:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_good_turing_2))
print("Recall:", recall_score(test_data['Class Index'], predictions_good_turing_2, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_good_turing_2))

print("\nGood-Turing Smoothing Sonuçları - 3-gram:")
print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_good_turing_3))
print("Recall:", recall_score(test_data['Class Index'], predictions_good_turing_3, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_good_turing_3))


#Interpolasyon Hesaplamaları İçin Ağırlıklar
weights = [0.4, 0.3, 0.3]

#Interpolasyon Sonuçları
for n in [1, 2, 3]:
        interpolated_probs = interpolate_ngrams(
            class_ngram_counts_1, class_ngram_counts_2, class_ngram_counts_3,
            class_totals_1, class_totals_2, class_totals_3, weights
        )

        class_labels = list(class_ngram_counts_1.keys())
        predictions_interpolated = classify_with_interpolation(test_data, interpolated_probs, class_labels, n=n)

        print(f"Interpolasyon Sonuçları - {n}-gram:")
        print("Accuracy:", accuracy_score(test_data['Class Index'], predictions_interpolated))
        print("Recall:", recall_score(test_data['Class Index'], predictions_interpolated, average='macro'))
        print("Confusion Matrix:\n", confusion_matrix(test_data['Class Index'], predictions_interpolated))


# Metin verisini ayırma ve etiketleri alma
train_texts = train_data['Cleaned Description'].tolist()
train_labels = train_data['Class Index'].tolist()
test_texts = test_data['Cleaned Description'].tolist()
test_labels = test_data['Class Index'].tolist()

# TF-IDF Dönüştürme (5, 10 ve 15 kelimelik çantalar)
for max_features in [5, 10, 15]:
    print(f"\n{max_features} Kelime Çantası ile ANN Testi Başlıyor...")
    
    X_train, X_test, vectorizer = prepare_tfidf(train_texts, test_texts, max_features)
    ann_model = create_ann_model(X_train, train_labels)

    # Tahminleme ve Sonuçlar
    predictions = ann_model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions, average='macro')
    conf_matrix = confusion_matrix(test_labels, predictions)

    print(f"\n{max_features} Kelime Çantası için Sonuçlar:")
    print(f"Doğruluk (Accuracy): {accuracy:.2f}")
    print(f"Hatırlama (Recall): {recall:.2f}")
    print(f"Karmaşıklık Matrisi:\n{conf_matrix}")