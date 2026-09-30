import time
import joblib
import pandas as pd

print("========== MODEL MONITORING ==========")

model = joblib.load("models/student_model.pkl")

sample = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "A",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "teacher",
    "Fjob": "health",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 4
}

df = pd.DataFrame([sample])

start = time.time()

prediction = model.predict(df)

end = time.time()

latency = end - start

print("Predicted Score:", round(float(prediction[0]), 2))
print("Prediction Latency:", round(latency, 6), "seconds")

if prediction[0] < 10:
    print("Monitoring Alert: Low predicted performance")
else:
    print("Prediction within normal range")

print("Monitoring completed successfully.")
print("======================================")