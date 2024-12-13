import gradio as gr
import requests

API_URL_MANUAL = "http://backend:8000/predict-manual"
API_URL_FILE = "http://backend:8000/predict-file"

FIELDS = [
    ("Borrow Block Number", "number"),
    ("Borrow Timestamp", "number"),
    ("Wallet Address", "textbox"),
    ("First TX Timestamp", "number"),
    ("Last TX Timestamp", "number"),
    ("Wallet Age", "number"),
    ("Incoming TX Count", "number"),
    ("Outgoing TX Count", "number"),
    ("Net Incoming TX Count", "number"),
    ("Total Gas Paid ETH", "number"),
    ("Avg Gas Paid Per TX ETH", "number"),
    ("Risky TX Count", "number"),
    ("Risky Unique Contract Count", "number"),
    ("Risky First TX Timestamp", "number"),
    ("Risky Last TX Timestamp", "number"),
    ("Risky First Last TX Timestamp Diff", "number"),
    ("Risky Sum Outgoing Amount ETH", "number"),
    ("Outgoing TX Sum ETH", "number"),
    ("Incoming TX Sum ETH", "number"),
    ("Outgoing TX Avg ETH", "number"),
    ("Incoming TX Avg ETH", "number"),
    ("Max ETH Ever", "number"),
    ("Min ETH Ever", "number"),
    ("Total Balance ETH", "number"),
    ("Risk Factor", "number"),
    ("Total Collateral ETH", "number"),
    ("Total Collateral Avg ETH", "number"),
    ("Total Available Borrows ETH", "number"),
    ("Total Available Borrows Avg ETH", "number"),
    ("Avg Weighted Risk Factor", "number"),
    ("Risk Factor Above Threshold Daily Count", "number"),
    ("Avg Risk Factor", "number"),
    ("Max Risk Factor", "number"),
    ("Borrow Amount Sum ETH", "number"),
    ("Borrow Amount Avg ETH", "number"),
    ("Borrow Count", "number"),
    ("Repay Amount Sum ETH", "number"),
    ("Repay Amount Avg ETH", "number"),
    ("Repay Count", "number"),
    ("Borrow Repay Diff ETH", "number"),
    ("Deposit Count", "number"),
    ("Deposit Amount Sum ETH", "number"),
    ("Time Since First Deposit", "number"),
    ("Withdraw Amount Sum ETH", "number"),
    ("Withdraw Deposit Diff If Positive ETH", "number"),
    ("Liquidation Count", "number"),
    ("Time Since Last Liquidated", "number"),
    ("Liquidation Amount Sum ETH", "number"),
    ("Market ADX", "number"),
    ("Market ADXR", "number"),
    ("Market APO", "number"),
    ("Market Aroonosc", "number"),
    ("Market Aroonup", "number"),
    ("Market ATR", "number"),
    ("Market CCI", "number"),
    ("Market CMO", "number"),
    ("Market Correl", "number"),
    ("Market DX", "number"),
    ("Market FastK", "number"),
    ("Market FastD", "number"),
    ("Market HT Trendmode", "number"),
    ("Market Linearreg Slope", "number"),
    ("Market MACD MACDext", "number"),
    ("Market MACD MACDfix", "number"),
    ("Market MACD", "number"),
    ("Market MACD Signal MACDext", "number"),
    ("Market MACD Signal MACDfix", "number"),
    ("Market MACD Signal", "number"),
    ("Market Max Drawdown 365d", "number"),
    ("Market NATR", "number"),
    ("Market Plus DI", "number"),
    ("Market Plus DM", "number"),
    ("Market PPO", "number"),
    ("Market ROCP", "number"),
    ("Market ROCR", "number"),
    ("Unique Borrow Protocol Count", "number"),
    ("Unique Lending Protocol Count", "number"),
]

def predict_manual(*args):
    data = {field[0].lower().replace(" ", "_"): value for field, value in zip(FIELDS, args)}
    try:
        response = requests.post(API_URL_MANUAL, json=data)
        response.raise_for_status()
        prediction = response.json()["prediction"]
        return f"Predicción: {prediction}"
    except Exception as e:
        return f"Error: {e}"

def predict_file(file):
    try:
        with open(file.name, "rb") as f:
            response = requests.post(API_URL_FILE, files={"file": f})
            response.raise_for_status()
            predictions = response.json()["predictions"]
        return f"Predicciones: {predictions}"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## Predicción con Gradio y FastAPI")
    with gr.Tab("Predicción Manual"):
        gr.Markdown("### Introduzca los datos manualmente:")
        manual_inputs = [gr.Number(label=field[0]) if field[1] == "number" else gr.Textbox(label=field[0]) for field in FIELDS]
        predict_button = gr.Button("Predecir")
        manual_output = gr.Textbox(label="Resultado")
        predict_button.click(predict_manual, inputs=manual_inputs, outputs=manual_output)
    with gr.Tab("Predicción con Archivo"):
        gr.Markdown("### Suba un archivo CSV:")
        file_input = gr.File(label="Cargar Archivo CSV")
        file_output = gr.Textbox(label="Resultado")
        file_input.change(predict_file, inputs=file_input, outputs=file_output)

demo.launch(server_name="0.0.0.0", server_port=7860)
