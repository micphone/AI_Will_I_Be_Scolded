# Predicting Negative Feedback for Debunking Video Scripts

This project is an AI application designed to predict whether a debunking video script is likely to receive negative feedback. It utilizes machine learning techniques such as text preprocessing, TF-IDF feature extraction, and a Stochastic Gradient Descent classifier.

**Note:** The dataset files (.csv) and model files (.pkl, .npy) are not uploaded to the repository due to size constraints and privacy considerations.

## Features

- **Text Preprocessing:** Tokenization, stopword removal, and text normalization using `jieba` and `nltk`.
- **Model Training:** Utilizes `SGDClassifier` for training with the ability to update based on user feedback.
- **Web Application:** A Flask-based web app (`app.py`) that allows users to input scripts and receive predictions.
- **Feedback Mechanism:** Users can provide feedback on predictions to improve the model over time.

## Requirements

- Python 3.x
- Libraries: `pandas`, `scikit-learn`, `nltk`, `jieba`, `Flask`, `joblib`, `numpy`

## Usage

1. Install the required libraries using `pip install -r requirements.txt`.
2. Run `train_sgd_model.py` to train the initial model.
3. Start the web application by running `app.py`.
4. Access the web interface at `http://localhost:5000/`.

**Note:** Since the data and model files are not provided, you will need to supply your own dataset and retrain the model accordingly.

## Training Data Format Requirements

### CSV File
- **Columns**:
  - **文案内容 (Script Content)**: The text of the video script.
  - **标签 (Label)**: The label indicating whether the script received negative feedback (被骂) or not (未被骂).

### Consistency
- Ensure that labels are consistent and only contain the specified categories.

### Preprocessing
- Data should be preprocessed appropriately before training, including steps such as tokenization and stopword removal.