import pandas as pd
import jieba
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import joblib

# 加载数据集
data = pd.read_csv('../data/Data.csv')

# 数据清洗和预处理
stop_words = set(stopwords.words('chinese'))

def preprocess_text(text):
    words = jieba.lcut(text)
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(words)

data['processed_text'] = data['文案内容'].apply(preprocess_text)

# 标签编码
le = LabelEncoder()
data['label'] = le.fit_transform(data['标签'])

# 保存预处理后的数据和标签编码器
data.to_csv('processed_data.csv', index=False)
joblib.dump(le, '../models/label_encoder.pkl')