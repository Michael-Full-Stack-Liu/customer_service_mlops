# service.py
# BentoML service for intent classification and slot extraction using MLflow PyFunc (Day 3 compatible)
# Non-LLM design: Rule-based slots for zero hallucination. Switch to ONNX for optimization if needed.

import re
import json
import bentoml
import mlflow.pyfunc
import numpy as np
from typing import Dict, Any
import os

# Load MLflow PyFunc model (from Day 3: update URI to your run)
MODEL_URI = "models/setfit_intent"  # Replace with actual (e.g., "runs:/abc123/model")
# Or local path: MODEL_URI = "./mlruns/0/<run_id>/artifacts/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Intent labels from training (match Day 3: update based on your train.csv unique intents)
INTENTS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund"
]  # Ensure order matches model's label encoder

@bentoml.service(resources={"cpu": "1"})
class CustomerBotService:
    # API endpoint for chat: POST /chat with JSON {"utterance": "I want to cancel order 123"}
    @bentoml.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
    async def classify_intent_and_slots(self, parsed_data: Dict[str, str]) -> Dict[str, Any]:
        utterance = parsed_data.get("utterance", "")
        if not utterance:
            return {"error": "No utterance provided"}

        # Step 1: Intent classification with MLflow PyFunc (SetFit)
        prediction = model.predict([utterance])  # Returns [{'labels': [list of intents], 'scores': [probs]}]
        scores = np.array(prediction[0]['scores'])  # Softmax probs
        intent_idx = np.argmax(scores)
        intent = prediction[0]['labels'][intent_idx]  # Top-1 intent
        confidence = float(scores[intent_idx])

        # Step 2: Slot extraction with regex (rule-based, zero hallucination)
        slots = {}
        # Extract order ID (e.g., "order 12345" -> {"order_id": "12345"})
        order_match = re.search(r'order\s+(\d+)', utterance, re.IGNORECASE)
        if order_match:
            slots["order_id"] = order_match.group(1)
        # Extract date (e.g., "11/08/2025" or "last month" – extend for relative dates if needed)
        date_match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', utterance)
        if date_match:
            slots["date"] = date_match.group(0)
        # Add more regex for other slots (e.g., email: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
            "utterance": utterance
        }

    # Step 3: Health check (unchanged)
    @bentoml.api(input=bentoml.io.Text(), output=bentoml.io.Text())
    async def healthz(self) -> str:
        return "OK"

# For local testing: Run `bentoml serve service:CustomerBotService --reload`
# Optional: Switch to ONNX (uncomment below, install onnxruntime, export in Day 3)
# import onnxruntime as ort
# session = ort.InferenceSession("models/intent.onnx")
# def onnx_predict(utterance): ...  # Implement if switching