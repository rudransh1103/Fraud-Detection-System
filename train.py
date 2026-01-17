"""
Credit Card Fraud Detection - Training Script
Simple training pipeline for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CREDIT CARD FRAUD DETECTION - TRAINING")
print("="*60)

# STEP 1: Load data
print("\n[1/6] Loading data...")
try:
    df = pd.read_csv('data/creditcard.csv')
    print(f"✓ Data loaded! Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: creditcard.csv not found in data/")
    exit()

# STEP 2: Explore
print("\n[2/6] Exploring data...")
print(f"  Total: {len(df):,}")
print(f"  Fraud: {df['Class'].sum():,} ({df['Class'].sum()/len(df)*100:.2f}%)")

# STEP 3: Feature engineering
print("\n[3/6] Creating features...")
df_processed = df.copy()
df_processed['Hour'] = (df_processed['Time'] / 3600) % 24
df_processed['Amount_Log'] = np.log1p(df_processed['Amount'])
print(f"✓ Created {df_processed.shape[1]-df.shape[1]} new features")

# STEP 4: Split data
print("\n[4/6] Splitting data...")
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train):,}, Test: {len(X_test):,}")

# STEP 5: Train model
print("\n[5/6] Training model...")
print("  (This takes 2-3 minutes...)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Training complete!")

# STEP 6: Evaluate
print("\n[6/6] Evaluating model...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Legitimate', 'Fraud'],
                          digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(f"              Legit   Fraud")
print(f"Legit  {tn:10d}  {fp:6d}")
print(f"Fraud  {fn:10d}  {tp:6d}")

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC: {roc_auc:.4f}")

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'model.pkl'")

feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Features saved")

print("\n" + "="*60)
print("TRAINING COMPLETE! 🎉")
print("="*60)
print("\nNext steps:")
print("  1. python api.py")
print("  2. streamlit run app.py")
print("="*60)