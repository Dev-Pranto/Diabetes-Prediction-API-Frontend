import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .schemas import PredictionInput, PredictionOutput, HealthResponse

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and test data
try:
    model = joblib.load("app/diabetes_model.pkl")
    test_data = pd.read_csv("test_data.csv")
    X_test = test_data.drop('Outcome', axis=1)
    y_test = test_data['Outcome']
except Exception as e:
    print(f"Error loading model or test data: {e}")
    model = None
    X_test, y_test = None, None

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Diabetes Prediction API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        features = np.array([[
            input_data.Pregnancies,
            input_data.Glucose,
            input_data.BloodPressure,
            input_data.SkinThickness,
            input_data.Insulin,
            input_data.BMI,
            input_data.DiabetesPedigreeFunction,
            input_data.Age
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Calculate confidence
        confidence = float(probabilities[prediction])
        
        return PredictionOutput(
            prediction=int(prediction),
            result="Diabetic" if prediction == 1 else "Not Diabetic",
            confidence=round(confidence, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    if model is None or X_test is None or y_test is None:
        raise HTTPException(status_code=500, detail="Model or test data not loaded")
    
    try:
        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        precision = round(precision_score(y_test, y_pred), 2)
        recall = round(recall_score(y_test, y_pred), 2)
        f1 = round(f1_score(y_test, y_pred), 2)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")