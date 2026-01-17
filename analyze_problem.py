"""
Credit Card Fraud Detection - Problem Analysis
Quick analysis to understand the problem before building the solution
"""

import pandas as pd
import numpy as np

print("="*70)
print(" "*15 + "FRAUD DETECTION - PROBLEM ANALYSIS")
print("="*70)

# Load data
print("\n[STEP 1] Loading dataset...")
print("-" * 70)

try:
    df = pd.read_csv('data/creditcard.csv')
    print("✓ Data loaded successfully!")
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   - {df.shape[0]:,} transactions")
    print(f"   - {df.shape[1]} features")
except FileNotFoundError:
    print("\n❌ ERROR: Dataset not found!")
    print("\n📥 To get the dataset:")
    print("   1. Go to: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("   2. Download 'creditcard.csv'")
    print("   3. Place it in: data/creditcard.csv")
    exit()

# Understand the problem
print("\n[STEP 2] Understanding the problem...")
print("-" * 70)

fraud_count = df['Class'].sum()
legit_count = len(df) - fraud_count
fraud_rate = (fraud_count / len(df)) * 100

print(f"\n📈 Class Distribution:")
print(f"   Legitimate: {legit_count:,} ({100-fraud_rate:.2f}%)")
print(f"   Fraudulent: {fraud_count:,} ({fraud_rate:.4f}%)")

print(f"\n⚠️  THE PROBLEM:")
print(f"   - Only {fraud_rate:.4f}% are fraud!")
print(f"   - This is EXTREMELY IMBALANCED")
print(f"   - Regular ML won't work well")

# Amount analysis
legit_amounts = df[df['Class'] == 0]['Amount']
fraud_amounts = df[df['Class'] == 1]['Amount']

print(f"\n💰 Transaction Amounts:")
print(f"\n   Legitimate:")
print(f"   - Average: ${legit_amounts.mean():.2f}")
print(f"   - Median:  ${legit_amounts.median():.2f}")

print(f"\n   Fraudulent:")
print(f"   - Average: ${fraud_amounts.mean():.2f}")
print(f"   - Median:  ${fraud_amounts.median():.2f}")

# Time analysis
df['Hour'] = (df['Time'] / 3600) % 24
print(f"\n⏰ Transactions occur throughout 24 hours")

# Financial impact
avg_fraud_amount = fraud_amounts.mean()
total_fraud_loss = fraud_count * avg_fraud_amount

print(f"\n💸 Financial Impact:")
print(f"   - Total fraud loss: ${total_fraud_loss:,.2f}")
print(f"   - Per 100K transactions: ${(total_fraud_loss/len(df))*100000:,.2f}")

print(f"\n✅ ML can help prevent this loss!")
print(f"\n" + "="*70)
print("Analysis complete! Ready to build the model.")
print("="*70)