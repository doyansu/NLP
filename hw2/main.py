import nltk
from pprint import pprint
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags

data_folder = "./data/"
train_file_names = [data_folder + "a.conll", data_folder + "b.conll", data_folder + "f.conll", data_folder + "e.conll", data_folder + "g.conll"]
test_file_names = [data_folder + "h.conll"]

# 讀取 conll 文件
def read_conll_file(file_name):
    current_item = []
    with open(file_name, encoding='utf-8') as conll:
        for line in conll:
            line = line.strip()
            if line and len(line.split()) == 2: # 排除有問題的資料
                word, tag_class = line.split()
                if word.find(":") > 0:
                    word = word.replace(":", "/")
                current_item.append((word, tag_class))
    return current_item

# 預處理-詞性標記
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent) #詞性標記
    return sent

# NLTK 預處理
def nltk_preprocess(data):
    data_without_second_value = [(t[0]) for t in data]
    data_tostrig = ' '.join(data_without_second_value)
    data_preprocess = preprocess(data_tostrig)
    pattern = 'NP: {<DT>?<JJ>*<NN>}' #正規表達式提取NP短語
    cp = nltk.RegexpParser(pattern)
    data_cs = cp.parse(data_preprocess)
    data_iob_tagged = tree2conlltags(data_cs)
    return data_iob_tagged

# 將 NLTK 取出的 NP 短語特徵值插入原始數據
def nltk_insert_data(nltk_preprocess_data, data_o):
    i = 0
    result = []
    for data in data_o:
        temp_l = list(data)
        temp_l.insert(1, nltk_preprocess_data[i][1])
        temp_t = tuple(temp_l)
        result.append(temp_t)
        if (i < len(data_o)-1):
            i+=1
    return result

# 整個預處理過程
def all_preprocess(file_name):
    data_o = read_conll_file(file_name)
    nltk_preprocess_result = nltk_preprocess(data_o)
    nltk_insert_data_result = nltk_insert_data(nltk_preprocess_result, data_o)
    return nltk_insert_data_result

# 處理完的數據
train_sents_f = []
for file_name in train_file_names:
    train_sents = all_preprocess(file_name)
    train_sents_f.append(train_sents)

test_sents_f = []
for file_name in test_file_names:
    test_sents = all_preprocess(file_name)
    test_sents_f.append(test_sents)

# 特徵提取函数
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

# 特徵提取函數-應用於訓練和測試數據
def extract_features(sentences):
    X = []
    for sent in sentences:
        for i in range(len(sent)):
            X.append(word2features(sent, i))
    return X

# 標籤提取函數
def extract_labels(sentences):
    y = []
    for sent in sentences:
        for i in range(len(sent)):
            y.append(sent[i][-1])
    return y

# 特徵和標籤提取
X_train = extract_features(train_sents_f)
y_train = extract_labels(train_sents_f)
X_test = extract_features(test_sents_f)
y_test = extract_labels(test_sents_f)

# 特徵向量化
vec = DictVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 感知模型訓練和預測
clf = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 報告結果輸出
print(classification_report(y_test, y_pred))

