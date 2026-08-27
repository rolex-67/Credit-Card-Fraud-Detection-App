from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
import joblib
import numpy as np
from pydantic import BaseModel
import model_def  # Enables deserialization of FraudStackingEnsemble

app = FastAPI(
    title="AI Financial Fraud Detection API (Stacking Ensemble)",
    description="""State-of-the-Art Financial Fraud Detection API powered by a Stacking Ensemble of XGBoost, LightGBM/Histogram Gradient Boosting, and Logistic Regression Meta-Learner.""",
    version="2.0.0", 
    debug=True
)

# Load the trained Stacking Ensemble model
model = joblib.load('credit_fraud.pkl')

@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
AI Financial Fraud Detection API (XGBoost + LightGBM + Logistic Regression Stacking Ensemble) 🛡️

Explore interactive Swagger UI Docs at /docs or Redoc at /redoc
    """
    return note

favicon_path = 'favicon.png'
@app.get('/favicon.png', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)
																	
class fraudDetection(BaseModel):
    step: int
    types: int
    amount: float	
    oldbalanceorig: float	
    newbalanceorig: float	
    oldbalancedest: float	
    newbalancedest: float	
    isflaggedfraud: float

@app.post('/predict')
def predict(data: fraudDetection):
    features = np.array([[
        data.step, 
        data.types, 
        data.amount, 
        data.oldbalanceorig, 
        data.newbalanceorig, 
        data.oldbalancedest, 
        data.newbalancedest, 
        data.isflaggedfraud
    ]], dtype=np.float64)

    # Get probability and binary prediction
    proba = float(model.predict_proba(features)[0][1])
    is_fraud = int(model.predict(features)[0])

    status = "fraudulent" if is_fraud == 1 else "not fraudulent"
    
    # Return both list format for backwards compatibility and detailed dictionary
    return {
        "prediction": status,
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(proba, 4),
        "risk_percentage": round(proba * 100, 2),
        "0": status  # for backwards compatibility with resp[0]
    }
