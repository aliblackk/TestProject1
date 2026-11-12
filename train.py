"""Обучает простую модель на датасете Iris и сохраняет её в model.onnx."""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx

MODEL_PATH = Path(__file__).parent / "model.onnx"

# 1. Данные: 4 признака цветка -> один из 3 видов ириса
X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)  # ONNX ждёт float32
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Модель: масштабирование + логистическая регрессия
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print(f"Точность на тесте: {model.score(X_test, y_test):.3f}")

# 3. Экспорт в ONNX.
# zipmap=False -> вероятности выдаются обычным массивом, а не списком словарей
onx = to_onnx(
    model,
    X_train[:1],
    options={id(model): {"zipmap": False}},
    target_opset=17,
)
MODEL_PATH.write_bytes(onx.SerializeToString())
print(f"Модель сохранена: {MODEL_PATH}")
