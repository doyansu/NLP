from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
import os
import pandas as pd

# 載入預訓練的詞向量模型
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 讀取WordSim-353資料集
Humans_mean = []
def read_wordsim_dataset(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            word1, word2, score = line.strip().split('\t')
            if (score == "Human (mean)"):
                continue
            pairs.append((word1, word2, float(score)))
            Humans_mean.append(float(score))
    return pairs

wordsim_pairs = read_wordsim_dataset('wordsim353/combined.txt')

# 計算單詞相似度
similarities = []
for pair in wordsim_pairs:
    word1, word2, _ = pair
    similarity = word_vectors.similarity(word1, word2)
    similarities.append(similarity)
    
# 輸出 correlation
correlation, p_value = pearsonr(Humans_mean, similarities)
print('Word similarity')
print("Correlation:", correlation)

# 讀取 BATS_3.0 資料並轉成 features 的形式
def read_BATS_dataset():
    folder1_path = 'BATS_3.0'
    folder2_paths = []
    datas = pd.DataFrame(columns=['features', 'label'])

    if os.path.exists(folder1_path) and os.path.isdir(folder1_path):
        for file_name in os.listdir(folder1_path):
            folder2_paths.append(os.path.join(folder1_path, file_name))
    else:
        print('No such folder')

    for folder_name in folder2_paths:
        if not(os.path.exists(folder_name) and os.path.isdir(folder_name)):
            continue
        for file_name in os.listdir(folder_name):
            label = file_name.split(']')[0].split('[', 1)[1]
            temp = []
            data = pd.read_csv(os.path.join(folder_name, file_name), sep='\t', header=None, names=['word1', 'word2'])
            for index, row in data.iterrows():
                if '/' in row['word2']:
                    for word2 in row['word2'].split('/'):
                        try:
                            features = word_vectors[row['word1']] - word_vectors[word2]
                            temp.append(pd.Series({'features' : features, 'label' : label}))
                        except:
                            pass
                else:
                    try:
                        features = word_vectors[row['word1']] - word_vectors[row['word2']]
                        temp.append(pd.Series({'features' : features, 'label' : label}))
                    except:
                        pass
            datas = pd.concat([datas, pd.DataFrame(temp)], ignore_index=True)
    return datas
            

datas = read_BATS_dataset()
# 分類
X_train, X_test, y_train, y_test = train_test_split(list(datas['features']), list(datas['label']), test_size=0.2, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print()
print('Analogy prediction')
print(classification_report(y_test, predictions))