import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def PreProcessText(text : str):
    # 轉成小寫
    text = text.lower()
    
    # 將 not 後的字詞與 not 合併成新字詞 not_xxx
    NEGATIVES = ['not', 'don\'t', 'doesn\'t', 'didn\'t']
    for negative in NEGATIVES:
        negative_plus_space = ' ' + negative + ' '
        text = text.replace(negative_plus_space, ' not_')
        
    return text

# 顯示 (Accuracy, Precision, Recall, F1)
def ShowRates(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    print(f"Accuracy  : {accuracy * 100 :.2f}%")
    print(f"Precision : {precision * 100 :.2f}%")
    print(f"Recall    : {recall * 100 :.2f}%")
    print(f"F1        : {f1 * 100 :.2f}%")
    
# 分類
def Predict(classifier, train_data, test_data):
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data['text'])
    test_vectors = vectorizer.transform(test_data['text'])
    classifier.fit(train_vectors, train_data['label'])
    pred_labels = classifier.predict(test_vectors)
    return pred_labels

def Main():
    # 讀取資料
    train_data = pd.read_csv('data/train_150k.txt', sep='\t', header=None, names=['label', 'text'])
    test_data = pd.read_csv('data/test_62k.txt', sep='\t', header=None, names=['label', 'text'])
    train_data['text'] = train_data['text'].map(PreProcessText)
    test_data['text'] = test_data['text'].map(PreProcessText)

    # Naive Bayes
    print("Naive Bayes Result:")
    pred_labels = Predict(MultinomialNB(), train_data, test_data)
    ShowRates(test_data['label'], pred_labels)

    # LinearSVC
    print("LinearSVC Result:")
    pred_labels = Predict(LinearSVC(), train_data, test_data)
    ShowRates(test_data['label'], pred_labels)
    
Main()