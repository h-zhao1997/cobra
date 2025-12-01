#!/usr/bin/env python3
"""
Compare scratchpad results across different numbers of passes (1, 2, 3, 4).
"""
import json
from pathlib import Path
from collections import defaultdict

# Find all result files
results_dir = Path("results/scratchpad_passes_comparison")
result_files = sorted(results_dir.glob("run_*/scratchpad/results_scratchpad*.json"))

print("=" * 80)
print("Scratchpad Passes Comparison Summary")
print("=" * 80)

all_results = {}
for result_file in result_files:
    # Extract pass count from directory structure or metadata
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Try to get pass count from config
    config = data.get("meta", {}).get("config", {})
    passes = config.get("scratchpad_passes", "unknown")
    
    all_results[passes] = data
    
    print(f"\n{passes} Pass(es):")
    print(f"  File: {result_file}")
    
    agg_metrics = data.get("aggregate_metrics", {})
    if agg_metrics:
        print(f"  Aggregate Metrics:")
        for metric, value in sorted(agg_metrics.items()):
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")

# Compare metrics across passes
print("\n" + "=" * 80)
print("Metric Progression Across Passes")
print("=" * 80)

# Get all metric names
all_metrics = set()
for data in all_results.values():
    all_metrics.update(data.get("aggregate_metrics", {}).keys())

# Sort passes numerically
sorted_passes = sorted([p for p in all_results.keys() if isinstance(p, int)])

if sorted_passes:
    print(f"\n{'Metric':<20} " + " ".join([f"{p} Pass{'es' if p > 1 else ''}:>10" for p in sorted_passes]))
    print("-" * 80)
    
    for metric in sorted(all_metrics):
        if metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BERTScore-F1", "BERTScore-Precision", "BERTScore-Recall"]:
            values = []
            for passes in sorted_passes:
                value = all_results[passes].get("aggregate_metrics", {}).get(metric, "N/A")
                if isinstance(value, (int, float)):
                    values.append(f"{value:.4f}")
                else:
                    values.append("N/A")
            print(f"{metric:<20} " + " ".join([f"{v:>10}" for v in values]))

# Show sample comparison for first image
print("\n" + "=" * 80)
print("Sample Output Comparison (First Image)")
print("=" * 80)

for passes in sorted_passes:
    results = all_results[passes].get("results", [])
    if results:
        first_result = results[0]
        img_id = first_result.get("image_id", "?")
        caption = first_result.get("generated_caption", "")
        reasoning = first_result.get("reasoning_trace", "")
        
        print(f"\n{passes} Pass{'es' if passes > 1 else ''} (Image {img_id}):")
        print(f"  Caption: {caption[:150]}{'...' if len(caption) > 150 else ''}")
        if reasoning:
            print(f"  Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")

print("\n" + "=" * 80)

