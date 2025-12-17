# Scratchpad Multipass Compare Mode

## Overview

The `scratchpad_compare` mode runs scratchpad inference with multiple pass counts (1, 2, 3, 4) in a single evaluation and compares the results. This allows you to see how performance changes across different numbers of passes without running separate evaluations.

## Usage

### Command Line

```bash
# Compare 1, 2, 3, 4 passes (default)
python -m cobra_eval --method scratchpad_compare --num_samples 10

# Compare up to 5 passes (1, 2, 3, 4, 5)
python -m cobra_eval --method scratchpad_compare --num_samples 10 --scratchpad_passes 5
```

### Python API

```python
from cobra_eval.registry import Registry
from cobra_eval.generators import scratchpad_compare

# Compare 1, 2, 3, 4 passes
generator = Registry.get_generator("scratchpad_compare")(vlm, scratchpad_passes=4)

# Compare specific pass counts
generator = Registry.get_generator("scratchpad_compare")(
    vlm, 
    scratchpad_passes=4,
    compare_passes=[1, 2, 4]  # Only compare these pass counts
)

result = generator.generate(image, prompt)
```

## Output Structure

### Result Metadata

Each result includes comparison data for all pass counts:

```json
{
  "image_id": 0,
  "generated_caption": "...",  // Best pass caption
  "reasoning_trace": "...",     // Best pass reasoning
  "metadata": {
    "method": "scratchpad_compare",
    "compare_passes": [1, 2, 3, 4],
    "best_pass": 4,
    "comparison": {
      "1": {
        "caption": "Caption from 1 pass",
        "reasoning_trace": "Reasoning from 1 pass",
        "caption_length": 15,
        "reasoning_length": 120
      },
      "2": { ... },
      "3": { ... },
      "4": { ... }
    },
    "all_results": {
      "1": {"caption": "...", "reasoning": "..."},
      "2": {"caption": "...", "reasoning": "..."},
      "3": {"caption": "...", "reasoning": "..."},
      "4": {"caption": "...", "reasoning": "..."}
    }
  },
  "metrics": {
    "BLEU-1": 0.40,
    "BLEU-4": 0.07,
    "comparison_metrics": {
      "pass_1": {"BLEU-1": 0.40, "BLEU-4": 0.07},
      "pass_2": {"BLEU-1": 0.36, "BLEU-4": 0.07},
      "pass_3": {"BLEU-1": 0.33, "BLEU-4": 0.08},
      "pass_4": {"BLEU-1": 0.32, "BLEU-4": 0.00}
    }
  }
}
```

### Aggregate Metrics

The aggregate metrics include comparison metrics for each pass count:

```json
{
  "aggregate_metrics": {
    "BLEU-1": 0.40,
    "BLEU-4": 0.07,
    "comparison_by_passes": {
      "pass_1": {
        "BLEU-1": 0.40,
        "BLEU-2": 0.24,
        "BLEU-3": 0.13,
        "BLEU-4": 0.07
      },
      "pass_2": { ... },
      "pass_3": { ... },
      "pass_4": { ... }
    }
  }
}
```

## Analysis

### Comparing Pass Counts

The results allow you to:

1. **See performance degradation**: Compare BLEU scores across pass counts
2. **Identify optimal pass count**: Find which number of passes performs best
3. **Analyze caption evolution**: See how captions change with more passes
4. **Measure verbosity**: Track caption and reasoning length changes

### Example Analysis

```python
import json

# Load results
with open("results/run_XXX/scratchpad_compare/results_scratchpad_compare_XXX.json") as f:
    data = json.load(f)

# Extract comparison metrics
comparison = data["aggregate_metrics"]["comparison_by_passes"]

# Find best pass count for each metric
for metric in ["BLEU-1", "BLEU-4", "BERTScore-F1"]:
    best_pass = max(comparison.keys(), 
                   key=lambda p: comparison[p].get(metric, 0))
    best_score = comparison[best_pass][metric]
    print(f"{metric}: Best at {best_pass} with {best_score:.4f}")
```

## Benefits

1. **Single Run**: Compare all pass counts in one evaluation
2. **Consistent Conditions**: Same images, same model, same parameters
3. **Detailed Metrics**: Per-pass metrics for comprehensive analysis
4. **Efficient**: Reuses model loading and setup

## Limitations

1. **Slower**: Runs multiple pass configurations per image
2. **Memory**: Stores all pass results in memory
3. **Visualization**: Current visualization shows only best pass (can be extended)

## Future Enhancements

1. **Visualization**: Create charts showing metric progression across passes
2. **Best Pass Selection**: Automatically select best pass based on metrics
3. **Selective Comparison**: Compare only specific pass counts
4. **Parallel Execution**: Run different pass counts in parallel

## Example Output

```
Running Method: scratchpad_compare
Initialized scratchpad_compare generator
Comparing pass counts: [1, 2, 3, 4]
Loading dataset (10 samples)...
Starting generation at 10:00:00...
[10:00:15] Processed 1/10 samples (4 passes per sample)...
[10:00:30] Processed 2/10 samples (4 passes per sample)...
...
Computing global metrics...
Computing comparison metrics for each pass count...
Results saved to results/run_XXX/scratchpad_compare/results_scratchpad_compare_XXX.json
```

The results file will contain:
- Main metrics (from best pass)
- Comparison metrics for each pass count
- Full comparison data in metadata

