# Cobra Evaluation System (`cobra_eval`)

A modular, extensible benchmarking system for the Cobra VLM, supporting multiple inference strategies (Baseline, Scratchpad), comprehensive metrics (BLEU, BERTScore), and structured JSON output.

## Installation

Ensure you have the project dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

Run the benchmark module directly from the command line:

```bash
python -m cobra_eval [OPTIONS]
```

### Quick Start

**Run baseline evaluation on 10 samples:**
```bash
python -m cobra_eval --method baseline --num_samples 10
```

**Run scratchpad reasoning evaluation:**
```bash
python -m cobra_eval --method scratchpad --num_samples 100
```

**Run comparison (Baseline vs Scratchpad):**
```bash
python -m cobra_eval --method both --num_samples 50
```

**Resume from latest run:**
```bash
python -m cobra_eval --method scratchpad --load_results latest
```

### Command Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Method & Data** | | | |
| `--method` | str | `baseline` | Inference strategy: `baseline`, `scratchpad`, or `both`. |
| `--num_samples` | int | `100` | Number of images to evaluate from the dataset. |
| `--scratchpad_passes` | int | `1` | Number of reasoning passes (only for `scratchpad` method). |
| `--no_streaming` | flag | `False` | If set, disables streaming mode (downloads full dataset). |
| **Model Config** | | | |
| `--model_id` | str | `cobra+3b` | HuggingFace Model ID or local path. |
| `--hf_token` | str | `.hf_token` | Path to file containing HuggingFace token. |
| **Generation** | | | |
| `--temperature` | float | `0.4` | Sampling temperature for generation. |
| `--max_new_tokens` | int | `512` | Maximum new tokens for final caption. |
| `--reasoning_max_tokens` | int | `256` | Maximum new tokens for reasoning traces. |
| `--repetition_penalty` | float | `1.2` | Penalty for repetition to reduce loops. |
| **System & Output** | | | |
| `--output_dir` | str | `results` | Directory to save JSON results and visualizations. |
| `--clear_cache` | flag | `False` | Clear GPU cache before loading the model. |
| `--min_free_gb` | float | `8.0` | Minimum free GPU memory (GB) required to run. |
| **Input/Resume** | | | |
| `--load_results` | str | `None` | Path to results file to resume from (or "latest"). |

## Output Structure

Results are organized by run timestamp in the `results/` directory:
`results/run_<timestamp>/<method>/results_<method>.json`

When running `--method both`, a comparison summary and visualization are also generated in the run directory:
- `comparison_summary_<timestamp>.json`: Win rates and metric differences.
- `comparison_baseline_vs_scratchpad_<timestamp>.png`: Side-by-side visualization.

**JSON Structure (Individual Results):**
```json
{
  "meta": {
    "timestamp": "2023-11-25T12:00:00",
    "config": { ... }
  },
  "aggregate_metrics": {
    "BLEU-1": 0.75,
    "BLEU-4": 0.32,
    "BERTScore-F1": 0.88
  },
  "results": [
    {
      "image_id": 12345,
      "reference_captions": ["A cat sitting on a mat.", ...],
      "generated_caption": "A black cat resting on a blue rug.",
      "reasoning_trace": "Step 1: Identify objects...",
      "metrics": { "BLEU-4": 0.45 },
      "metadata": { "passes": 1 }
    },
    ...
  ]
}
```

## Extensibility

### Adding a New Generator
1. Create a new file in `cobra_eval/generators/`.
2. Inherit from `BaseGenerator`.
3. Decorate with `@Registry.register_generator("my_method")`.

```python
from cobra_eval.registry import Registry
from cobra_eval.interfaces import BaseGenerator, GenerationResult

@Registry.register_generator("my_method")
class MyGenerator(BaseGenerator):
    def generate(self, image, prompt, **kwargs):
        # Implementation
        return GenerationResult(caption="...")
```

### Adding a New Metric
1. Create a new file in `cobra_eval/metrics/`.
2. Inherit from `BaseMetric`.
3. Decorate with `@Registry.register_metric("my_metric")`.

```python
from cobra_eval.registry import Registry
from cobra_eval.interfaces import BaseMetric

@Registry.register_metric("my_metric")
class MyMetric(BaseMetric):
    def compute(self, references, predictions):
        return {"MyScore": 0.95}
```

