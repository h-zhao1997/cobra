# LLaVA-CoT Structured Reasoning Implementation

This document explains how to use the LLaVA-CoT structured reasoning method with multiple scratchpad passes in your benchmark system.

## Overview

LLaVA-CoT implements structured, multi-stage reasoning inspired by the [LLaVA-CoT paper](https://github.com/PKU-YuanGroup/LLaVA-CoT). The method generates responses with four distinct stages:

1. **SUMMARY**: Problem understanding and approach
2. **CAPTION**: Visual interpretation from the image
3. **REASONING**: Step-by-step logical analysis
4. **CONCLUSION**: Final answer/conclusion

## Usage

### Command Line Interface

**Run LLaVA-CoT with 1 pass:**
```bash
python -m cobra_eval --method llava_cot --num_samples 100 --scratchpad_passes 1
```

**Run LLaVA-CoT with 2 passes (iterative refinement):**
```bash
python -m cobra_eval --method llava_cot --num_samples 100 --scratchpad_passes 2
```

**Compare all methods (Baseline, Scratchpad, LLaVA-CoT):**
```bash
python -m cobra_eval --method all --num_samples 100 --scratchpad_passes 2
```

### Python API

```python
from cobra_eval.registry import Registry
from cobra_eval.generators import llava_cot  # Import to register

# Initialize generator with multiple passes
generator = Registry.get_generator("llava_cot")(vlm, scratchpad_passes=2)

# Generate structured reasoning
result = generator.generate(
    image,
    "Please carefully observe the image and come up with a caption for the image.",
    temperature=0.4,
    max_new_tokens=512,
    reasoning_max_tokens=512,
    repetition_penalty=1.2
)

# Access results
print(f"Final Caption: {result.caption}")
print(f"Reasoning Trace: {result.reasoning_trace}")

# Access structured stages
if result.metadata.get("stages"):
    stages = result.metadata["stages"]
    print(f"Summary: {stages.get('summary')}")
    print(f"Caption Stage: {stages.get('caption')}")
    print(f"Reasoning: {stages.get('reasoning')}")
    print(f"Conclusion: {stages.get('conclusion')}")
```

## How Multiple Passes Work

When `scratchpad_passes > 1`, the generator performs iterative refinement:

1. **First Pass**: Generates structured reasoning with all four stages
2. **Subsequent Passes**: Reviews previous output and refines each stage
3. **Final Output**: Uses the most refined pass (last pass)

The generator automatically stops early if a valid conclusion is reached.

## Output Structure

The `GenerationResult` contains:

- **`caption`**: Final caption (extracted from CONCLUSION stage)
- **`reasoning_trace`**: Full structured reasoning (SUMMARY + CAPTION + REASONING)
- **`metadata`**: 
  - `method`: "llava_cot"
  - `passes`: Number of requested passes
  - `actual_passes`: Number of passes actually used
  - `stages`: Dictionary with extracted stages (summary, caption, reasoning, conclusion)
  - `all_passes`: List of all passes with their stages

## Example Output

```
Final Caption: A black bicycle-shaped clock with a white face and black numbers.

Full Reasoning Trace:
SUMMARY: I need to analyze this image and identify the key objects and their relationships.

CAPTION: The image shows a bicycle-shaped object mounted on a white wall. The front wheel is replaced with a clock face.

REASONING: Step 1: Identify the main object - it's a bicycle frame. Step 2: Notice the front wheel is a clock. Step 3: The object is decorative, combining bicycle and clock elements.

Metadata:
{
  "method": "llava_cot",
  "passes": 2,
  "actual_passes": 2,
  "stages": {
    "summary": "...",
    "caption": "...",
    "reasoning": "...",
    "conclusion": "..."
  }
}
```

## Integration with Benchmark System

The LLaVA-CoT generator is fully integrated with the `cobra_eval` benchmark system:

- ✅ Supports all standard generation parameters
- ✅ Works with BLEU and BERTScore metrics
- ✅ Generates structured JSON output
- ✅ Compatible with visualization system
- ✅ Supports result caching and resuming

## Comparison with Other Methods

| Method | Approach | Stages | Multiple Passes |
|--------|----------|--------|-----------------|
| **Baseline** | Direct generation | None | No |
| **Scratchpad** | Two-pass (reasoning → caption) | 2 | Yes (iterative reasoning) |
| **LLaVA-CoT** | Structured single-pass with tags | 4 (SUMMARY, CAPTION, REASONING, CONCLUSION) | Yes (iterative refinement) |

## Notes

- The generator automatically extracts structured stages using regex patterns
- If structured format is not detected, it falls back to treating the entire response as reasoning
- Multiple passes allow the model to refine its analysis iteratively
- The final caption is extracted from the CONCLUSION stage if available

## References

- **LLaVA-CoT Paper**: Xu et al. (2025). "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step." ICCV 2025. arXiv:2411.10440
- **GitHub Repository**: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **Dataset**: [Xkev/LLaVA-CoT-100k on HuggingFace](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)

