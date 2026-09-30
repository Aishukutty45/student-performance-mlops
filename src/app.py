from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="Student Performance Prediction API"
)

model = joblib.load("models/student_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "predicted_final_score": round(float(prediction[0]), 2)
    }