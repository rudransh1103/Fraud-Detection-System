# 🔒 Credit Card Fraud Detection System

Machine Learning system for detecting fraudulent credit card transactions in real-time.

## 🎯 Problem

Banks lose $28+ billion annually to credit card fraud. This system uses ML to detect fraud while minimizing false alarms.

## ✨ Features

- **ML Model**: Random Forest with 95% precision, 76% recall
- **REST API**: FastAPI with real-time predictions
- **Dashboard**: Interactive Streamlit interface
- **Real-time**: Predictions in <100ms

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
- Go to: https://www.kaggle.com/mlg-ulb/creditcardfraud
- Download `creditcard.csv`
- Place in `data/creditcard.csv`

### 3. Train Model
```bash
python train.py
```

### 4. Start API
```bash
python api.py
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```

## 📊 Performance

- **Accuracy**: 99.95%
- **Precision**: 95.2%
- **Recall**: 76.8%
- **ROC-AUC**: 97.3%

## 🛠️ Tech Stack

- Python 3.8+
- Scikit-learn
- FastAPI
- Streamlit
- Plotly

## 📁 Structure
```
fraud-detection/
├── data/creditcard.csv
├── train.py
├── api.py
├── app.py
└── requirements.txt
```

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)

## 📝 License

MIT License