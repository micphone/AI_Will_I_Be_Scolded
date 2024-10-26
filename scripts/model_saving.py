import joblib

# 假设最佳模型和向量化器已经在之前的步骤中保存
# 这里只是确认一下保存的文件

# 检查保存的模型和向量化器
best_model = joblib.load('../models/best_model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')
le = joblib.load('../models/label_encoder.pkl')