
import os
import time
import mlflow
import pandas as pd
import jiwer
from transformers import pipeline

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Whisper_Tiny_vs_Base")

# Load evaluation data
df = pd.read_csv("data/eval_inputs.csv")

# Initialize models
models = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base"
}

def normalize_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

os.makedirs("outputs", exist_ok=True)

for short_name, model_id in models.items():

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1
    )

    predictions = []
    latencies = []

    for audio_path in df["audio_path"]:
        start = time.perf_counter()

        result = asr(
            audio_path,
            generate_kwargs={
                "language": "english",
                "task": "transcribe"
            }
        )

        latencies.append(time.perf_counter() - start)
        predictions.append(result["text"])

    references = [normalize_text(x) for x in df["reference"]]
    hypotheses = [normalize_text(x) for x in predictions]

    wer = jiwer.wer(references, hypotheses)
    avg_latency = sum(latencies) / len(latencies)
    total_time = sum(latencies)

    with mlflow.start_run(
        run_name=f"evaluation_{short_name}"
    ):
        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "automatic-speech-recognition",
            "sample_count": len(df),
            "device": "CPU"
        })

        mlflow.log_metrics({
            "wer": wer,
            "avg_latency_ms": avg_latency * 1000,
            "total_inference_s": total_time
        })

        output = pd.DataFrame({
            "audio_path": df["audio_path"],
            "reference": df["reference"],
            "prediction": predictions,
            "latency_seconds": latencies
        })

        output.to_csv(
            f"outputs/{short_name}_reproduced.csv",
            index=False
        )

        mlflow.log_artifact(
            f"outputs/{short_name}_reproduced.csv"
        )

    print(
        f"{short_name}: WER={wer:.4f}, "
        f"Average latency={avg_latency:.2f}s"
    )

print("Evaluation completed.")
