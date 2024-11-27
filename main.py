from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import uvicorn

app = FastAPI(
    title="Modelo de Potabilidad de Agua",
    description=(
        "Esta API utiliza un modelo optimizado de XGBoost para predecir si una medición de agua es potable "
    ),
    version="1.0"
)

class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

model_path = "./models/best_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("No se encontró el modelo en la ruta. Asegúrate de que esté entrenado y guardado.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Ruta de prueba (home)
@app.get("/")
async def home():
    return {
        "mensaje": "API de Potabilidad de Agua",
        "modelo": "XGBoost",
        "problema": "Clasificación binaria para determinar si el agua es potable o no",
        "entrada": [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ],
        "salida": {"potabilidad": "0 (no potable) o 1 (potable)"}
    }

@app.post("/potabilidad/")
async def predict_potabilidad(data: WaterQuality):
    try:
        input_data = np.array([[data.ph, data.Hardness, data.Solids, data.Chloramines,
                                data.Sulfate, data.Conductivity, data.Organic_carbon,
                                data.Trihalomethanes, data.Turbidity]])

        prediction = model.predict(input_data)

        return {"potabilidad": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
