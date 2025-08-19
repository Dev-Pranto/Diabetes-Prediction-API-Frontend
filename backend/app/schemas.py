from pydantic import BaseModel

class PredictionInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class PredictionOutput(BaseModel):
    prediction: int
    result: str
    confidence: float

class HealthResponse(BaseModel):
    status: str