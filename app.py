"""FastAPI-сервис, который отдаёт предсказания ONNX-модели.

Запуск:  uvicorn app:app --reload
Документация: http://127.0.0.1:8000/docs
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.onnx"
CLASSES = ["setosa", "versicolor", "virginica"]

# Модель загружается один раз при старте сервиса
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

app = FastAPI(title="Iris ONNX API")


class Flower(BaseModel):
    sepal_length: float = Field(examples=[5.1])
    sepal_width: float = Field(examples=[3.5])
    petal_length: float = Field(examples=[1.4])
    petal_width: float = Field(examples=[0.2])


class Prediction(BaseModel):
    class_id: int
    class_name: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(flower: Flower):
    x = np.array(
        [[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]],
        dtype=np.float32,
    )
    labels, probs = session.run(None, {INPUT_NAME: x})
    class_id = int(labels[0])
    return Prediction(
        class_id=class_id,
        class_name=CLASSES[class_id],
        probabilities={name: round(float(p), 4) for name, p in zip(CLASSES, probs[0])},
    )