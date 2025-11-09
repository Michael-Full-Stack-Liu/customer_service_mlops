# train_model.py
# Script for fine-tuning SetFit model on intent classification data using MLflow for tracking.
# Assumes data/train.csv with columns: 'utterance' (text), 'intent' (label).
# Exports to ONNX and registers in MLflow.
# Supports remote MLflow tracking via MLFLOW_TRACKING_URI env var for Docker readiness.

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import pickle
from sentence_transformers.losses import CosineSimilarityLoss
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load data
print("Loading data...")
df = pd.read_csv("data/train.csv")


# Split: 80/20 train/val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df[['query', 'label']])
val_dataset = Dataset.from_pandas(val_df[['query', 'label']])

# Log data stats to MLflow
num_samples = len(train_df)
num_classes = df['label'].nunique()
print(f"Will Train on {num_samples} samples, {num_classes} classes.")

# Step 2: Configure MLflow for remote tracking (local fallback)
# Set tracking URI via env var (e.g., export MLFLOW_TRACKING_URI=http://mlflow:5000 for Docker)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")  # Default to local server
mlflow.set_tracking_uri(tracking_uri)

# Optional: Verify URI connectivity (enterprise-grade robustness)
try:
    mlflow.search_experiments()  # Light probe; raises if unreachable
    print(f"MLflow tracking URI set to: {tracking_uri}")
except Exception as e:
    print(f"Warning: MLflow URI check failed ({e}). Falling back to local file store.")
    mlflow.set_tracking_uri("file:///")  # Fallback to local mlruns dir

# Proceed with experiment
mlflow.set_experiment("intent-classification1")
with mlflow.start_run(run_name="setfit-minilm-v1"):
    # Log params
    mlflow.log_param("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    mlflow.log_param("num_iterations", 20)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("num_epochs", 3)
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("train_samples", num_samples)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("loss_class", "CosineSimilarityLoss")

    # Step 3: Initialize and train SetFit model
    print("Initializing SetFit model...")
    checkpoint_path = "./checkpoints/checkpoint-327" if os.path.exists("./checkpoints/checkpoint-327") else "sentence-transformers/all-MiniLM-L6-v2"
    model = SetFitModel.from_pretrained(checkpoint_path, num_classes=num_classes)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_class=CosineSimilarityLoss,  # SOTA contrastive loss
        batch_size=32,
        num_iterations=1,  # Few-shot iterations
        num_epochs=0,
        column_mapping={"query": "text", "label": "label"},
    )

    print("Starting fine-tuning...")
    trainer.train()
    
    # Step 4: Evaluate on validation set
    preds = trainer.model(val_dataset["query"])
    true_labels = val_dataset["label"]
    accuracy = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Log metrics
    mlflow.log_metric("val_accuracy", accuracy)
    # Note: SetFit doesn't have direct train eval; approximate via subset if needed
    train_preds = trainer.model(train_dataset["query"][:len(preds)])  # Subset for approx
    train_accuracy = accuracy_score(train_dataset["label"][:len(preds)], train_preds)
    mlflow.log_metric("train_accuracy", train_accuracy)

    # Step 5: Save SetFit model as PyTorch checkpoint (native, no ONNX needed)
    print("Saving SetFit model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/setfit_intent"

    # Save full SetFit model (embedder + head + tokenizer)
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    class SetFitWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model

        def predict(self, context, model_input):
            if isinstance(model_input, list):
                return self.model(model_input)
            elif "text" in model_input.columns:
                return self.model(model_input["text"].tolist())
            else:
                raise ValueError("Expected a 'text' column or list of strings.")
    
    # Log to MLflow
    mlflow.log_artifacts(model_path)
    mlflow.pyfunc.log_model(
        name="setfit_model",
        python_model=SetFitWrapper(model),
        registered_model_name="intent-model-v1-string"
    )
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/setfit_model"
    print(f"Model saved and registered as 'intent-model-v1-string'. Path: {model_path}. URI: {model_uri}")

    # Step 6: Verify load and predict (strings direct)
    loaded_model = mlflow.pyfunc.load_model("models:/intent-model-v1-string/1")
    test_input = ["I want to cancel my order"]
    test_pred_str = loaded_model.predict(test_input)  # Direct strings!
    print(f"Load test: Input '{test_input[0]}' -> Output '{test_pred_str[0]}' (string success).")

print("Training complete. Run 'mlflow ui' to view experiments.")