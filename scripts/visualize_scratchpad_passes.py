#!/usr/bin/env python3
"""
Create visualization charts comparing scratchpad performance across different numbers of passes.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Find all result files
results_dir = Path("results/scratchpad_passes_comparison")
result_files = sorted(results_dir.glob("run_*/scratchpad/results_scratchpad*.json"))

# Load all results
all_results = {}
for result_file in result_files:
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    config = data.get("meta", {}).get("config", {})
    passes = config.get("scratchpad_passes", "unknown")
    
    if isinstance(passes, int):
        all_results[passes] = data

# Sort passes
sorted_passes = sorted(all_results.keys())

if not sorted_passes:
    print("No results found!")
    exit(1)

# Extract metrics
metrics_data = {
    "BLEU-1": [],
    "BLEU-2": [],
    "BLEU-3": [],
    "BLEU-4": [],
    "BERTScore-F1": [],
    "BERTScore-Precision": [],
    "BERTScore-Recall": []
}

for passes in sorted_passes:
    agg_metrics = all_results[passes].get("aggregate_metrics", {})
    for metric in metrics_data.keys():
        value = agg_metrics.get(metric, 0.0)
        metrics_data[metric].append(value)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Scratchpad Performance Across Different Numbers of Passes', fontsize=16, fontweight='bold')

# Plot 1: BLEU Scores
ax1 = axes[0, 0]
ax1.plot(sorted_passes, metrics_data["BLEU-1"], marker='o', linewidth=2, markersize=8, label='BLEU-1')
ax1.plot(sorted_passes, metrics_data["BLEU-2"], marker='s', linewidth=2, markersize=8, label='BLEU-2')
ax1.plot(sorted_passes, metrics_data["BLEU-3"], marker='^', linewidth=2, markersize=8, label='BLEU-3')
ax1.plot(sorted_passes, metrics_data["BLEU-4"], marker='d', linewidth=2, markersize=8, label='BLEU-4')
ax1.set_xlabel('Number of Passes', fontsize=12)
ax1.set_ylabel('BLEU Score', fontsize=12)
ax1.set_title('BLEU Scores vs Number of Passes', fontsize=14, fontweight='bold')
ax1.set_xticks(sorted_passes)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_ylim(bottom=0)

# Plot 2: BERTScore Metrics
ax2 = axes[0, 1]
ax2.plot(sorted_passes, metrics_data["BERTScore-F1"], marker='o', linewidth=2, markersize=8, label='BERTScore-F1', color='green')
ax2.plot(sorted_passes, metrics_data["BERTScore-Precision"], marker='s', linewidth=2, markersize=8, label='BERTScore-Precision', color='orange')
ax2.plot(sorted_passes, metrics_data["BERTScore-Recall"], marker='^', linewidth=2, markersize=8, label='BERTScore-Recall', color='purple')
ax2.set_xlabel('Number of Passes', fontsize=12)
ax2.set_ylabel('BERTScore', fontsize=12)
ax2.set_title('BERTScore Metrics vs Number of Passes', fontsize=14, fontweight='bold')
ax2.set_xticks(sorted_passes)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_ylim([0.5, 0.8])

# Plot 3: BLEU Scores Bar Chart
ax3 = axes[1, 0]
x = np.arange(len(sorted_passes))
width = 0.2
ax3.bar(x - 1.5*width, metrics_data["BLEU-1"], width, label='BLEU-1', alpha=0.8)
ax3.bar(x - 0.5*width, metrics_data["BLEU-2"], width, label='BLEU-2', alpha=0.8)
ax3.bar(x + 0.5*width, metrics_data["BLEU-3"], width, label='BLEU-3', alpha=0.8)
ax3.bar(x + 1.5*width, metrics_data["BLEU-4"], width, label='BLEU-4', alpha=0.8)
ax3.set_xlabel('Number of Passes', fontsize=12)
ax3.set_ylabel('BLEU Score', fontsize=12)
ax3.set_title('BLEU Scores Comparison (Bar Chart)', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{p} Pass{"es" if p > 1 else ""}' for p in sorted_passes])
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(bottom=0)

# Plot 4: Combined Metric Summary
ax4 = axes[1, 1]
# Normalize metrics to 0-1 scale for comparison
normalized_bleu4 = [m / max(metrics_data["BLEU-4"]) if max(metrics_data["BLEU-4"]) > 0 else 0 for m in metrics_data["BLEU-4"]]
normalized_bert_f1 = [(m - 0.5) / 0.3 for m in metrics_data["BERTScore-F1"]]  # Scale from 0.5-0.8 to 0-1

ax4.plot(sorted_passes, normalized_bleu4, marker='o', linewidth=2, markersize=8, label='BLEU-4 (normalized)', color='red')
ax4.plot(sorted_passes, normalized_bert_f1, marker='s', linewidth=2, markersize=8, label='BERTScore-F1 (normalized)', color='blue')
ax4.set_xlabel('Number of Passes', fontsize=12)
ax4.set_ylabel('Normalized Score', fontsize=12)
ax4.set_title('Key Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
ax4.set_xticks(sorted_passes)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best')
ax4.set_ylim([0, 1.1])

plt.tight_layout()

# Save the figure
output_path = results_dir / "scratchpad_passes_comparison_charts.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Charts saved to: {output_path}")

# Also create a detailed metrics table visualization
fig2, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
headers = ['Metric'] + [f'{p} Pass{"es" if p > 1 else ""}' for p in sorted_passes]

for metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BERTScore-F1", "BERTScore-Precision", "BERTScore-Recall"]:
    row = [metric]
    for passes in sorted_passes:
        value = all_results[passes].get("aggregate_metrics", {}).get(metric, 0.0)
        row.append(f"{value:.4f}")
    table_data.append(row)

# Find best value for each metric (for highlighting)
colors = []
for i, metric in enumerate(["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BERTScore-F1", "BERTScore-Precision", "BERTScore-Recall"]):
    row_colors = ['white'] * (len(sorted_passes) + 1)
    values = [all_results[p].get("aggregate_metrics", {}).get(metric, 0.0) for p in sorted_passes]
    if values:
        best_idx = values.index(max(values))
        row_colors[best_idx + 1] = '#90EE90'  # Light green for best
    colors.append(row_colors)

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                colWidths=[0.25] + [0.15] * len(sorted_passes))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Highlight best values
for i in range(len(table_data)):
    for j in range(1, len(headers)):
        if colors[i][j] == '#90EE90':
            table[(i+1, j)].set_facecolor('#90EE90')
            table[(i+1, j)].set_text_props(weight='bold')

# Style header
for j in range(len(headers)):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(weight='bold', color='white')

plt.title('Scratchpad Metrics Comparison Table\n(Green = Best Value)', fontsize=14, fontweight='bold', pad=20)

output_path2 = results_dir / "scratchpad_passes_comparison_table.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Table saved to: {output_path2}")

plt.close('all')
print("\nVisualization complete!")

