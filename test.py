import numpy as np
import pandas as pd
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    features = pickle.load(f)

# Load data
df = pd.read_csv('data/creditcard.csv')

# Test on 100 ACTUAL fraud transactions
frauds = df[df['Class'] == 1].head(100)

# Prepare
X = frauds.drop('Class', axis=1)
X['Hour'] = (X['Time'] / 3600) % 24
X['Amount_Log'] = np.log1p(X['Amount'])
X = X[features]

# Predict
predictions = model.predict(X)

caught = predictions.sum()
print(f"Out of 100 REAL frauds:")
print(f"Caught: {caught}")
print(f"Missed: {100-caught}")
print(f"Detection Rate: {caught}%")