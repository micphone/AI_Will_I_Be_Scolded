from flask import Flask, request, jsonify, render_template_string
from prediction_function import predict_sentiment, update_model
import joblib
import os

app = Flask(__name__)

# 加载模型、向量化器和标签编码器
best_model = joblib.load('sgd_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# 定义简单的 HTML 模板，添加了反馈功能
TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>文案预测</title>
</head>
<body>
    <h1>预测您的文案是否会被骂</h1>
    <form method="post">
        <textarea name="text" rows="10" cols="50" placeholder="请输入您的文案">{{ text if text else '' }}</textarea><br>
        <input type="submit" value="预测">
    </form>
    {% if result %}
    <h2>预测结果：{{ result }}</h2>
    <form method="post" action="/feedback">
        <input type="hidden" name="text" value="{{ text }}">
        <input type="hidden" name="prediction" value="{{ result }}">
        <label>该预测结果是否准确？</label><br>
        <input type="radio" name="feedback" value="正确" required> 正确<br>
        <input type="radio" name="feedback" value="错误" required> 错误<br>
        <input type="submit" value="提交反馈">
    </form>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '')
        if not text:
            return render_template_string(TEMPLATE, result='请输入文案内容。')
        result = predict_sentiment(text)
        # 直接传递预测结果，不添加前缀
        return render_template_string(TEMPLATE, result=result, text=text)
    return render_template_string(TEMPLATE)

@app.route('/feedback', methods=['POST'])
def feedback():
    text = request.form.get('text', '')
    prediction = request.form.get('prediction', '')
    feedback = request.form.get('feedback', '')

    # 将反馈数据保存到文件
    feedback_file = 'feedback_data.csv'
    header = 'text,prediction,feedback\n'
    data_line = f'"{text}","{prediction}","{feedback}"\n'

    # 如果文件不存在，先写入表头
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(data_line)
    else:
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(data_line)

    # 根据反馈更新模型
    # 判断真实标签
    if feedback == '正确':
        true_label = prediction
    else:
        # 如果预测错误，真实标签应为相反的类别
        true_label = '被骂' if prediction == '未被骂' else '未被骂'

    # 将真实标签转换为数值
    true_label_encoded = le.transform([true_label])[0]
    # 更新模型
    update_model(text, true_label_encoded)

    return render_template_string(TEMPLATE, result='感谢您的反馈！', text='')

if __name__ == '__main__':
    app.run(debug=True)