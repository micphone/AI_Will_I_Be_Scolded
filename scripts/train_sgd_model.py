import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# 加载预处理后的数据
data = pd.read_csv('../data/processed_data.csv')
X = data['processed_text']
y = data['标签']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 保存标签编码器和 classes
joblib.dump(le, '../models/label_encoder.pkl')
classes = le.transform(le.classes_)
np.save('../classes.npy', classes)

# 初始化向量化器并拟合
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
joblib.dump(vectorizer, '../models/vectorizer.pkl')

# 初始化并训练模型
best_model = SGDClassifier(loss='log_loss', max_iter=1000)
best_model.fit(X_vec, y_encoded)

# 保存模型和模型的 classes_
joblib.dump(best_model, '../models/sgd_model.pkl')
np.save('../model_classes.npy', best_model.classes_)