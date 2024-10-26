import jieba
import string
import joblib
from nltk.corpus import stopwords
import numpy as np

# 加载模型、向量化器和标签编码器
best_model = joblib.load('../models/sgd_model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')
le = joblib.load('../models/label_encoder.pkl')

# 加载 classes
classes = np.load('../classes.npy')

stop_words = set(stopwords.words('chinese'))

def preprocess_text(text):
    words = jieba.lcut(text)
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(words)

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = best_model.predict(text_vec)
    label = le.inverse_transform(prediction)
    return label[0]

def update_model(text, true_label):
    # 预处理文本
    processed_text = preprocess_text(text)
    # 向量化文本
    text_vec = vectorizer.transform([processed_text])
    # 更新模型
    best_model.partial_fit(text_vec, [true_label], classes=classes)
    # 保存更新后的模型
    joblib.dump(best_model, '../models/sgd_model.pkl')