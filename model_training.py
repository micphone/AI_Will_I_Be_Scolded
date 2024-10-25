import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 加载预处理后的数据
data = pd.read_csv('processed_data.csv')

# 特征和标签
X = data['processed_text']
y = data['label']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 向量化
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 模型训练
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 模型评估
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 保存初始模型和向量化器
joblib.dump(model, 'initial_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')