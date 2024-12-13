from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
import joblib
import io
import os
from proyecto_final import utils


app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/complete_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo o pipeline: {e}")

class InputData(BaseModel):
    borrow_block_number: float
    borrow_timestamp: float
    wallet_address: str
    first_tx_timestamp: float
    last_tx_timestamp: float
    wallet_age: float
    incoming_tx_count: float
    outgoing_tx_count: float
    net_incoming_tx_count: float
    total_gas_paid_eth: float
    avg_gas_paid_per_tx_eth: float
    risky_tx_count: float
    risky_unique_contract_count: float
    risky_first_tx_timestamp: float
    risky_last_tx_timestamp: float
    risky_first_last_tx_timestamp_diff: float
    risky_sum_outgoing_amount_eth: float
    outgoing_tx_sum_eth: float
    incoming_tx_sum_eth: float
    outgoing_tx_avg_eth: float
    incoming_tx_avg_eth: float
    max_eth_ever: float
    min_eth_ever: float
    total_balance_eth: float
    risk_factor: float
    total_collateral_eth: float
    total_collateral_avg_eth: float
    total_available_borrows_eth: float
    total_available_borrows_avg_eth: float
    avg_weighted_risk_factor: float
    risk_factor_above_threshold_daily_count: float
    avg_risk_factor: float
    max_risk_factor: float
    borrow_amount_sum_eth: float
    borrow_amount_avg_eth: float
    borrow_count: float
    repay_amount_sum_eth: float
    repay_amount_avg_eth: float
    repay_count: float
    borrow_repay_diff_eth: float
    deposit_count: float
    deposit_amount_sum_eth: float
    time_since_first_deposit: float
    withdraw_amount_sum_eth: float
    withdraw_deposit_diff_if_positive_eth: float
    liquidation_count: float
    time_since_last_liquidated: float
    liquidation_amount_sum_eth: float
    market_adx: float
    market_adxr: float
    market_apo: float
    market_aroonosc: float
    market_aroonup: float
    market_atr: float
    market_cci: float
    market_cmo: float
    market_correl: float
    market_dx: float
    market_fastk: float
    market_fastd: float
    market_ht_trendmode: float
    market_linearreg_slope: float
    market_macd_macdext: float
    market_macd_macdfix: float
    market_macd: float
    market_macdsignal_macdext: float
    market_macdsignal_macdfix: float
    market_macdsignal: float
    market_max_drawdown_365d: float
    market_natr: float
    market_plus_di: float
    market_plus_dm: float
    market_ppo: float
    market_rocp: float
    market_rocr: float
    unique_borrow_protocol_count: float
    unique_lending_protocol_count: float

@app.post("/predict-manual")
def predict_manual(data: InputData):
    """
    Endpoint para predicción con entrada manual.
    """
    try:
        df = pd.DataFrame([data.dict()])

        transformed_data = pipeline.transform(df)

        prediction = model.predict(transformed_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

@app.post("/predict-file")
async def predict_file(file: UploadFile):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        transformed_data = pipeline.transform(df)

        predictions = model.predict(transformed_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")


