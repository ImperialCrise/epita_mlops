import joblib
from fastapi import FastAPI
model = joblib.load('../regression.joblib')

app = FastAPI(host="0.0.0.0", port=8000)

@app.post("/predict")
def predict(size: float, nb_rooms: float, garden: float):
    val = model.predict([[size, nb_rooms, garden]])[0]
    return {
        "y_pred": val
    }

@app.get("/")
def home():
    return "Hello, World!d"
