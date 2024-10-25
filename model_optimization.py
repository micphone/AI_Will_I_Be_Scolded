import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 加载数据和向量化器
data = pd.read_csv('processed_data.csv')
vectorizer = joblib.load('vectorizer.pkl')

X = data['processed_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 网格搜索进行超参数调优
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1')
grid.fit(X_train_vec, y_train)

# 最佳模型
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_best))

# 保存最佳模型
joblib.dump(best_model, 'best_model.pkl')