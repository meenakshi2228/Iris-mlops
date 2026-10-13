import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

model = joblib.load("models/iris_model.joblib")

flower_names = ["Setosa", "Versicolor", "Virginica"]

class FlowerInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris ML API is running"}

@app.post("/predict")
def predict(data: FlowerInput):
    sample = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(sample)[0]

    return {
        "predicted_class": int(prediction),
        "predicted_flower": flower_names[int(prediction)]
    }