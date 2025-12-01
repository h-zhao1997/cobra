# External Models Testing Guide

This guide explains how to test against external models (GPT-5, Gemini, Claude, and Llama) using the evaluation system.

## Setup

### 1. Install Required Packages

The external model packages are optional dependencies. Install them as needed:

```bash
pip install openai anthropic google-generativeai
```

Note: `transformers` and `torch` are already in requirements.txt for Llama support.

### 2. Set Up API Keys

You can provide API keys in three ways:

1. **Environment Variables** (recommended):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GEMINI_API_KEY="your-gemini-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export HF_TOKEN="your-huggingface-token"  # For Llama
   ```

2. **Command Line Arguments**:
   ```bash
   python -m cobra_eval --method gpt5 --openai_api_key "your-key"
   ```

3. **Key Files**:
   ```bash
   python -m cobra_eval --method gpt5 --openai_api_key_file ".openai_key"
   ```

## Usage Examples

### Test Individual Models

**GPT-5 (or GPT-4o):**
```bash
python -m cobra_eval --method gpt5 --num_samples 10 --gpt5_model "gpt-4o"
```

**Gemini:**
```bash
python -m cobra_eval --method gemini --num_samples 10 --gemini_model "gemini-1.5-pro"
```

**Claude:**
```bash
python -m cobra_eval --method claude --num_samples 10 --claude_model "claude-3-5-sonnet-20241022"
```

**Llama:**
```bash
python -m cobra_eval --method llama --num_samples 10 --llama_model_id "meta-llama/Llama-3.2-11B-Vision-Instruct"
```

### Test All External Models at Once

```bash
python -m cobra_eval --method external --num_samples 10
```

This will run all four external models (GPT-5, Gemini, Claude, Llama) and generate comparison visualizations.

### Compare External Models with Cobra Methods

You can manually run multiple methods and compare results:

```bash
# Run baseline
python -m cobra_eval --method baseline --num_samples 10

# Run GPT-5
python -m cobra_eval --method gpt5 --num_samples 10

# Compare results manually or use the comparison visualization
```

## Model Options

### GPT-5/OpenAI Models
- `gpt-4o` (default) - Latest GPT-4o model
- `gpt-4o-mini` - Faster, cheaper option
- `gpt-5` - When available

### Gemini Models
- `gemini-1.5-pro` (default) - Most capable
- `gemini-1.5-flash` - Faster option

### Claude Models
- `claude-3-5-sonnet-20241022` (default) - Latest Sonnet
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fastest

### Llama Models
- `meta-llama/Llama-3.2-11B-Vision-Instruct` (default)
- Any HuggingFace vision-language model ID

## Output

Results are saved in the same format as other methods:
- JSON results: `results/run_YYYYMMDD_HHMMSS/{method}/results_{method}_YYYYMMDD_HHMMSS.json`
- Visualizations: `results/run_YYYYMMDD_HHMMSS/{method}/viz_results_{method}_YYYYMMDD_HHMMSS.png`
- Comparison visualizations (when using `--method external`): `results/run_YYYYMMDD_HHMMSS/comparison_all_methods_YYYYMMDD_HHMMSS.png`

## Notes

- External models don't require loading a local VLM, so they're faster to start
- API costs apply for GPT-5, Gemini, and Claude (check your provider's pricing)
- Llama runs locally and requires GPU memory
- All models use the same evaluation metrics (BLEU, BERTScore) for fair comparison

