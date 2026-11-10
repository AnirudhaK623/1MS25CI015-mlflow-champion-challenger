
import mlflow
from mlflow import MlflowClient

# Connect to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

client = MlflowClient()

MODEL_NAME = "WhisperASR"

# Source URIs of the two logged MLflow models
models = {
    "champion": "models:/m-5fa39acf72184d88ba468dc41b3de0b3",
    "challenger": "models:/m-65a64666876547638d5ff6b7f79f0bac"
}

# Register a model only if its source is not already registered
existing_versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

for alias, source_uri in models.items():

    matching_version = next(
        (
            version for version in existing_versions
            if version.source == source_uri
        ),
        None
    )

    if matching_version is None:
        registered = mlflow.register_model(
            model_uri=source_uri,
            name=MODEL_NAME
        )
        version_number = registered.version
    else:
        version_number = matching_version.version

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version_number
    )

    print(f"{alias}: version {version_number}")

print("\nModel registration and aliases verified!")
