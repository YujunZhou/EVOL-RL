#!/usr/bin/env python3
"""
Fast evaluation metrics calculation script
Used to parse verl training output validation metrics and calculate pass@k, maj@k and other metrics
"""

import re
import sys
import json
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from itertools import combinations


def parse_validation_output(log_content: str) -> Dict[str, Dict[str, float]]:
    """
    Parse validation output logs and extract metrics for each dataset

    Args:
        log_content: Log content

    Returns:
        Dictionary of dataset name -> metric name -> metric value
    """
    metrics = defaultdict(dict)

    # Match various formats of metric lines
    patterns = [
        # New format: "val-aux/MATH-TTT/acc/maj@16/mean:0.723"
        r'val-(?:aux|core)/([^/]+)/acc/([^:]+):([\d.]+)',
        # Old format: "val/AIME-TTT/acc/best@16/mean: 0.8542"
        r'val/([^/]+)/acc/([^:]+):\s*([\d.]+)',
        # Other possible formats
        r'val-([^/]+)/acc/([^:]+):\s*([\d.]+)',
        # Negative log likelihood metric format: "val_core/MATH-TTT/neg_log_likelihood/mean: 1.234"
        r'val_core/([^/]+)/neg_log_likelihood/([^:]+):\s*([\d.]+)',
        r'val-(?:aux|core)/([^/]+)/neg_log_likelihood/([^:]+):\s*([\d.]+)',
        r'val/([^/]+)/neg_log_likelihood/([^:]+):\s*([\d.]+)',
    ]

    for line in log_content.split('\n'):
        # For each line, find all matching metrics
        for pattern in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                dataset = match[0]
                metric_name = match[1]
                metric_value = float(match[2])

                # Compatible with logs printed in Python dictionary format where keys are wrapped in quotes:
                # For example: 'val-aux/bbeh/acc/mean@1': 0.0002
                # In this case, regex will capture metric_name as "mean@1'", need to remove trailing/leading quotes.
                dataset = dataset.strip().strip("'\"")
                metric_name = metric_name.strip().strip("'\"")

                # When appearing multiple times, keep the larger value to avoid subsequent low-precision printing (like 0.000) overriding previous precise non-zero values
                prev = metrics[dataset].get(metric_name)
                if prev is None or metric_value > prev:
                    metrics[dataset][metric_name] = metric_value

    return dict(metrics)


def _fallback_metric(dataset_metrics: Dict[str, float], kind: str, target_k: int) -> float:
    """
    Robustly select appropriate metrics from log keys:
    - kind in {"mean", "maj", "best"}
    - For mean use keys like mean@K
    - For maj/best use keys like maj@K/mean, best@K/mean
    Strategy: Prefer K<=target_k maximum K; if not available, use maximum available K; otherwise return 0.0.
    """
    candidates = []
    for key, val in dataset_metrics.items():
        if kind == "mean":
            # mean@K
            if key.startswith("mean@"):
                try:
                    k = int(key.split("@")[1])
                    candidates.append((k, val))
                except Exception:
                    continue
        elif kind in ("maj", "best"):
            # maj@K/mean 或 best@K/mean
            if key.startswith(f"{kind}@") and key.endswith("/mean"):
                try:
                    mid = key.split("@")[1]
                    k = int(mid.split("/")[0])
                    candidates.append((k, val))
                except Exception:
                    continue
    if not candidates:
        return 0.0
    # First select maximum K <= target_k
    leq = [cv for cv in candidates if cv[0] <= target_k]
    if leq:
        return max(leq, key=lambda x: x[0])[1]
    # Otherwise take maximum K
    return max(candidates, key=lambda x: x[0])[1]


def _map_metrics_to_pass_style(dataset_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Map original metrics to pass@k style, supporting fallback for low sampling scenarios like n=1.
    - pass@1 ≈ mean@K, prefer K=32/16/1, if none available use maximum K from all available mean@K
    - maj@16 ≈ maj@K/mean fallback to available K
    - pass@16 ≈ best@K/mean fallback to available K
    - best@32 ≈ best@K/mean fallback to available K
    - H_ttrl directly take 'mean' parsed from neg_log_likelihood/mean (if available)
    """
    # pass@1: prefer mean@32,16,1
    pass1 = dataset_metrics.get("mean@32")
    if pass1 is None:
        pass1 = dataset_metrics.get("mean@16")
    if pass1 is None:
        pass1 = dataset_metrics.get("mean@1")
    if pass1 is None:
        pass1 = _fallback_metric(dataset_metrics, kind="mean", target_k=1)

    maj16 = dataset_metrics.get("maj@16/mean")
    if maj16 is None:
        maj16 = _fallback_metric(dataset_metrics, kind="maj", target_k=16)

    pass16 = dataset_metrics.get("best@16/mean")
    if pass16 is None:
        pass16 = _fallback_metric(dataset_metrics, kind="best", target_k=16)

    best32 = dataset_metrics.get("best@32/mean")
    if best32 is None:
        best32 = _fallback_metric(dataset_metrics, kind="best", target_k=32)

    h_ttrl = dataset_metrics.get("mean", 0.0)

    return {
        "pass@1": float(pass1 or 0.0),
        "maj@16": float(maj16 or 0.0),
        "pass@16": float(pass16 or 0.0),
        "best@32": float(best32 or 0.0),
        "H_ttrl": float(h_ttrl or 0.0),
    }


def calculate_bootstrap_metrics(scores: List[float], n_samples: int = 16, n_bootstrap: int = 1000) -> Dict[str, float]:
    """
    Calculate bootstrap metrics
    
    Args:
        scores: Score list
        n_samples: Number of samples
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Metrics dictionary
    """
    if len(scores) < n_samples:
        return {}
    
    np.random.seed(42)
    
    # Pass@k: probability of at least one correct
    pass_at_k_scores = []
    # Best@k: best score
    best_at_k_scores = []
    # Mean@k: average score
    mean_at_k_scores = []
    
    for _ in range(n_bootstrap):
        # Randomly sample n_samples scores
        sampled_scores = np.random.choice(scores, size=n_samples, replace=True)
        
        # Pass@k: at least one 1.0
        pass_at_k_scores.append(1.0 if np.max(sampled_scores) >= 1.0 else 0.0)
        
        # Best@k: maximum value
        best_at_k_scores.append(np.max(sampled_scores))
        
        # Mean@k: average value
        mean_at_k_scores.append(np.mean(sampled_scores))
    
    return {
        f'pass@{n_samples}': np.mean(pass_at_k_scores),
        f'best@{n_samples}': np.mean(best_at_k_scores),
        f'mean@{n_samples}': np.mean(mean_at_k_scores),
    }


def calculate_majority_voting(predictions: List[str], scores: List[float], n_samples: int = 16, n_bootstrap: int = 1000) -> float:
    """
    Calculate majority voting metrics
    
    Args:
        predictions: Prediction result list
        scores: Corresponding score list
        n_samples: Number of samples
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Average accuracy of majority voting
    """
    if len(predictions) != len(scores) or len(predictions) < n_samples:
        return 0.0
    
    np.random.seed(42)
    maj_scores = []
    
    for _ in range(n_bootstrap):
        # Randomly sample n_samples predictions and scores
        indices = np.random.choice(len(predictions), size=n_samples, replace=True)
        sampled_preds = [predictions[i] for i in indices]
        sampled_scores = [scores[i] for i in indices]
        
        # Calculate majority voting
        pred_counts = defaultdict(int)
        pred_to_score = {}
        
        for pred, score in zip(sampled_preds, sampled_scores):
            pred_counts[pred] += 1
            if pred not in pred_to_score:
                pred_to_score[pred] = score
        
        # Find majority prediction
        majority_pred = max(pred_counts, key=pred_counts.get)
        maj_scores.append(pred_to_score[majority_pred])
    
    return np.mean(maj_scores)


def format_metrics_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Format metrics table - using pass@k style metric names

    Args:
        metrics: Metrics dictionary

    Returns:
        Formatted table string
    """
    if not metrics:
        return "No valid metrics data found"

    # Table header - using pass@k style names, add log-likelihood column
    table = "\n" + "="*90 + "\n"
    table += f"{'Dataset':<15} {'pass@1':<10} {'maj@16':<10} {'pass@16':<10} {'best@32':<10} {'H_ttrl':<12}\n"
    table += "="*90 + "\n"

    # Data rows
    for dataset, dataset_metrics in metrics.items():
        mapped = _map_metrics_to_pass_style(dataset_metrics)
        table += f"{dataset:<15} {mapped['pass@1']:<10.4f} {mapped['maj@16']:<10.4f} {mapped['pass@16']:<10.4f} {mapped['best@32']:<10.4f} {mapped['H_ttrl']:<12.4f}\n"

    table += "="*90 + "\n"
    table += "\nMetric explanation (verl metrics → pass@k style):"
    table += "\n- pass@1 ≈ mean@32: Average success rate of single attempt"
    table += "\n- maj@16 = maj@16/mean: Majority voting success rate of 16 attempts"
    table += "\n- pass@16 ≈ best@16/mean: Best result success rate of 16 attempts"
    table += "\n- best@32: Best success rate of 32 attempts (reference)"
    table += "\n- H_ttrl: Negative log likelihood (Ĥ_ttrl(π_θ) = -1/N ∑log π_θ(y*))"

    return table


def save_results_to_json(metrics: Dict[str, Dict[str, float]], json_file: str, model_name: str = None):
    """
    Save results to JSON file, supporting accumulation of multiple test results

    Args:
        metrics: Metrics dictionary
        json_file: JSON file path
        model_name: Model name
    """
    import datetime
    import os

    # If file already exists, read existing data first
    existing_data = {}
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"📂 Found existing result file, will merge datasets...")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️  Unable to read existing file, will create new file...")
            existing_data = {}

    # Build result data
    result_data = {
        "model_name": model_name or existing_data.get("model_name", "unknown"),
        "evaluation_time": datetime.datetime.now().isoformat(),
        "first_evaluation_time": existing_data.get("first_evaluation_time", datetime.datetime.now().isoformat()),
        "datasets": existing_data.get("datasets", {})
    }

    # Extract key metrics for each dataset - using pass@k style names, convert to percentage
    new_datasets = []
    updated_datasets = []

    for dataset, dataset_metrics in metrics.items():
        mapped = _map_metrics_to_pass_style(dataset_metrics)
        dataset_data = {
            "pass@1": round(mapped['pass@1'] * 100, 2),
            "maj@16": round(mapped['maj@16'] * 100, 2),
            "pass@16": round(mapped['pass@16'] * 100, 2),
            "best@32": round(mapped['best@32'] * 100, 2),
            "H_ttrl": round(mapped['H_ttrl'], 4),
            "evaluation_time": datetime.datetime.now().isoformat(),
            "all_metrics": dataset_metrics,
        }

        if dataset in result_data["datasets"]:
            updated_datasets.append(dataset)
        else:
            new_datasets.append(dataset)

        result_data["datasets"][dataset] = dataset_data

    # Print update information
    if new_datasets:
        print(f"🆕 New datasets: {', '.join(new_datasets)}")
    if updated_datasets:
        print(f"🔄 Updated datasets: {', '.join(updated_datasets)}")

    # Calculate overall average metrics - based on all datasets (including existing and new)
    all_datasets = result_data["datasets"]
    if all_datasets:
        dataset_count = len(all_datasets)
        result_data["summary"] = {
            "total_datasets": dataset_count,
            "avg_pass@1": round(sum(data["pass@1"] for data in all_datasets.values()) / dataset_count, 2),
            "avg_maj@16": round(sum(data["maj@16"] for data in all_datasets.values()) / dataset_count, 2),
            "avg_pass@16": round(sum(data["pass@16"] for data in all_datasets.values()) / dataset_count, 2),
            "avg_best@32": round(sum(data["best@32"] for data in all_datasets.values()) / dataset_count, 2),
            "avg_H_ttrl": round(sum(data["H_ttrl"] for data in all_datasets.values()) / dataset_count, 4),
        }

    # Save to JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # Display save information
    total_datasets = len(result_data["datasets"])
    print(f"✅ Results saved to: {json_file}")
    print(f"📊 Current total: {total_datasets} datasets")

    if new_datasets or updated_datasets:
        print(f"📈 This evaluation: {len(new_datasets)} new datasets, {len(updated_datasets)} updated datasets")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Parse TTRL validation metrics')
    parser.add_argument('log_file', help='Log file path, use - to read from stdin')
    parser.add_argument('--save-json', help='Save results to JSON file')
    parser.add_argument('--model-name', help='Model name')

    args = parser.parse_args()

    log_file = args.log_file
    
    # Read log content
    if log_file == '-':
        log_content = sys.stdin.read()
    else:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found {log_file}")
            sys.exit(1)
    
    # Parse metrics
    metrics = parse_validation_output(log_content)
    
    if not metrics:
        sys.exit(1)
    
    # Display results
    print("\n🎯 TTRL Fast Evaluation Results")
    print(format_metrics_table(metrics))

    # Display detailed metrics
    print("\n📊 Detailed metrics:")
    for dataset, dataset_metrics in metrics.items():
        print(f"\n=== {dataset} ===")
        for metric_name, metric_value in sorted(dataset_metrics.items()):
            print(f"  {metric_name}: {metric_value:.4f}")

    # Save JSON results
    if args.save_json:
        save_results_to_json(metrics, args.save_json, args.model_name)

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
