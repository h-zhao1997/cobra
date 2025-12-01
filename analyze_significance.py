import json
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict
from tqdm import tqdm

# Ensure we can import from cobra_eval
import sys
sys.path.append(os.getcwd())

from cobra_eval.metrics.bleu import BLEUMetric

def load_data(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['images']

def prepare_inputs(images: List[Dict]):
    refs = {}
    preds = {}
    for img in images:
        # Use string ID as key, consistent with json
        key = img['id']
        refs[key] = img['reference_captions']
        # generated_caption is a string, wrap in list for consistency if needed, 
        # but usually predictions are List[str] for 1 candidate or just list of tokens?
        # BLEUMetric implementation: compute_score(references, predictions)
        # pycocoevalcap expects {id: [sent]}
        preds[key] = [img['generated_caption']]
    return refs, preds

def main():
    data_path = 'cobra_eval/data/baseline_caption_COCO_output.json'
    print(f"Loading data from {data_path}...")
    images = load_data(data_path)
    total_images = len(images)
    print(f"Total images: {total_images}")

    # Initialize Metric
    bleu_metric = BLEUMetric()

    # 1. Calculate True Score (Full Dataset)
    print("Calculating score for full dataset...")
    all_refs, all_preds = prepare_inputs(images)
    # Computing on 40k might take a moment but should be fast for BLEU
    full_results = bleu_metric.compute(all_refs, all_preds)
    true_bleu4 = full_results['BLEU-4']
    print(f"Full Dataset BLEU-4: {true_bleu4:.4f}")

    # 2. Bootstrap Sampling
    sample_size = 1000
    num_iterations = 100
    print(f"\nRunning bootstrap analysis: {num_iterations} iterations of sample_size={sample_size}...")
    
    sample_scores = []
    
    for i in tqdm(range(num_iterations)):
        # Random sample
        sample_indices = random.sample(range(total_images), sample_size)
        sample_images = [images[i] for i in sample_indices]
        
        refs, preds = prepare_inputs(sample_images)
        results = bleu_metric.compute(refs, preds)
        sample_scores.append(results['BLEU-4'])

    # 3. Analysis
    sample_scores = np.array(sample_scores)
    mean_score = np.mean(sample_scores)
    std_score = np.std(sample_scores)
    min_score = np.min(sample_scores)
    max_score = np.max(sample_scores)
    
    # 95% Confidence Interval
    ci_95 = 1.96 * std_score
    
    print("\n--- Results for 1k Sample Size ---")
    print(f"True BLEU-4: {true_bleu4:.4f}")
    print(f"Sampled Mean BLEU-4: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")
    print(f"Min Sample Score: {min_score:.4f}")
    print(f"Max Sample Score: {max_score:.4f}")
    print(f"95% CI: +/- {ci_95:.4f} (Range: {mean_score - ci_95:.4f} - {mean_score + ci_95:.4f})")
    
    relative_error = (ci_95 / true_bleu4) * 100
    print(f"Relative Margin of Error (95% Confidence): {relative_error:.2f}%")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(sample_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black', label='1k Samples')
    plt.axvline(true_bleu4, color='red', linestyle='dashed', linewidth=2, label=f'True Score ({true_bleu4:.4f})')
    plt.axvline(mean_score, color='green', linestyle='dotted', linewidth=2, label=f'Sample Mean ({mean_score:.4f})')
    
    plt.title(f'Distribution of BLEU-4 Scores (Sample Size: {sample_size})')
    plt.xlabel('BLEU-4 Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = 'bleu4_significance_1k.png'
    plt.savefig(output_plot)
    print(f"\nVisualization saved to {output_plot}")

if __name__ == "__main__":
    main()

