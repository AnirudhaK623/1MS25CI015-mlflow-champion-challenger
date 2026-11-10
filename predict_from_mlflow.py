
import mlflow
import pandas as pd
from mlflow import MlflowClient
import os

# Connect to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

MODEL_NAME = "WhisperASR"
ALIAS = "champion"

# Load Champion information
client = MlflowClient()

champion_version = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias=ALIAS
)

# Load the registered Champion model
model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
champion_model = mlflow.pyfunc.load_model(model_uri)

print("Champion model loaded successfully!")
print("Model version:", champion_version.version)
print("Hugging Face model: openai/whisper-base")

# Read new audio data
input_file = "data/new_data.csv"
new_data = pd.read_csv(input_file)

# Generate predictions through MLflow
predictions = champion_model.predict(
    new_data[["audio_path"]]
)

# Save results
new_data["prediction"] = predictions["prediction"].values

os.makedirs("outputs", exist_ok=True)

new_data.to_csv(
    "outputs/new_data_predictions.csv",
    index=False
)

print("Predictions completed!")
print("Saved: outputs/new_data_predictions.csv")
print(new_data[["audio_id", "prediction"]])
