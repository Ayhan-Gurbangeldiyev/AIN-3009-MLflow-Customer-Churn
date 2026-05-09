"""Send a sample prediction request to the MLflow model serving endpoint."""

from __future__ import annotations

import argparse
import json

import requests

from data_preprocessing import load_dataset, split_features_target


def build_payload(data_path: str, sample_index: int = 0) -> dict[str, object]:
    """Build a serving payload in MLflow's dataframe_split format."""

    df = load_dataset(data_path)
    x, _ = split_features_target(df)
    sample = x.iloc[[sample_index]]
    return {
        "dataframe_split": {
            "columns": sample.columns.tolist(),
            "data": sample.values.tolist(),
        }
    }


def send_prediction(url: str, data_path: str, sample_index: int) -> dict[str, object]:
    """Call the model endpoint and print a readable response."""

    payload = build_payload(data_path, sample_index=sample_index)
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    print("Request payload:")
    print(json.dumps(payload, indent=2))
    print("\nPrediction response:")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an MLflow model serving endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations", help="MLflow serving URL.")
    parser.add_argument("--data", default="data/telco_churn.csv", help="Path to the churn CSV file.")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index to send for prediction.")
    args = parser.parse_args()
    send_prediction(args.url, args.data, args.sample_index)
