"""
Credit Card Fraud Detection - FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

print("="*60)
print("Starting Fraud Detection API...")
print("="*60)

# Load model
print("\n[1/2] Loading model...")
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("✗ Error: Model files not found!")
    print("  Run 'python train.py' first")
    model = None
    feature_names = None
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    feature_names = None

# Create FastAPI app
print("\n[2/2] Initializing API...")
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection system",
    version="1.0.0"
)
print("✓ API initialized!")

# Simple transaction model
class Transaction(BaseModel):
    Time: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = 0.0

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

# Health check
@app.get("/health")
def health_check():
    """Check if API and model are working"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }

# Prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first."
        )
    
    try:
        # Convert to DataFrame
        data = transaction.dict()
        df = pd.DataFrame([data])
        
        # Add engineered features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        # Reorder columns
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = "CRITICAL"
        elif probability >= 0.6:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
        elif probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Create message
        is_fraud = bool(prediction == 1)
        if is_fraud:
            message = f"⚠️ FRAUD ALERT: High risk transaction ({probability*100:.1f}% confidence)"
        else:
            message = f"✓ Transaction appears legitimate (fraud risk: {probability*100:.1f}%)"
        
        # Return result
        return {
            "is_fraud": is_fraud,
            "fraud_probability": float(probability),
            "risk_level": risk_level,
            "message": message,
            "transaction_amount": float(transaction.Amount)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Simple test endpoint
@app.get("/test")
def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "API is working!",
        "model_status": "loaded" if model else "not loaded"
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("API SERVER STARTING")
    print("="*60)
    print("\n📍 Main URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("📖 Alt Docs: http://localhost:8000/redoc")
    print("❤️  Health:   http://localhost:8000/health")
    print("🧪 Test:     http://localhost:8000/test")
    print("\n" + "="*60)
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")