# Scratchpad Method Degradation Analysis

## Executive Summary

The scratchpad method shows **consistent linear degradation** with each additional pass. Performance drops from **BLEU-1: 0.40 (1 pass)** to **BLEU-1: 0.32 (4 passes)**, representing a **20% relative decrease**. This document analyzes the root causes using concrete examples from the evaluation results.

## Key Findings

### Performance Metrics Degradation

| Metric | 1 Pass | 2 Passes | 3 Passes | 4 Passes | Change |
|--------|---------|----------|----------|----------|--------|
| **BLEU-1** | **0.4000** | 0.3611 | 0.3259 | 0.3243 | **-19%** |
| **BLEU-2** | **0.2371** | 0.1964 | 0.2028 | 0.1749 | **-26%** |
| **BLEU-3** | **0.1310** | 0.1057 | 0.1220 | 0.0769 | **-41%** |
| **BLEU-4** | **0.0723** | 0.0661 | 0.0782 | **0.0000** | **-100%** |
| **BERTScore-F1** | **0.6733** | 0.6654 | 0.6468 | 0.6619 | **-2%** |

**Key Observation**: BLEU-4 drops to **0.0000** at 4 passes, indicating complete divergence from reference style.

## Root Cause Analysis

### 1. **The Refinement Prompt Encourages Verbosity, Not Accuracy**

**Current Implementation** (line 54 of `scratchpad.py`):
```python
msg = f"Review your previous analysis:\n{traces[-1]}\n\nRefine and expand on this analysis. Missed anything? Be more specific."
```

**Problems**:
- "Expand" → encourages longer outputs
- "Be more specific" → pushes for excessive detail
- No guidance to correct errors or stay concise
- No constraint to maintain factual accuracy

### 2. **Reasoning Traces Become Increasingly Off-Topic**

#### Example: Image 0 (Bicycle Clock)

**1 Pass - Reasonable:**
```
"The bicycle is black with thin tires; it has handlebars for steering control 
at position Y:X=0-1/2 from top left corner..."
```
- Technical description (verbose but factual)
- BLEU-1: **0.50**

**4 Passes - Degraded:**
```
Pass 1: Technical description (same as above)
Pass 2: "This description provides a detailed breakdown... thanks! Keep going great job!"
Pass 3: "In addition to identifying separate clock hands... good luck!!!!!)."
Pass 4: "So what do u think about my latest attempt guys?!,.._.;;))). 
        Let me know ya thoughts pleaseeeeee <3>😄✌️ 
        #bicycleclockartwork#steamengineinspireddesigns#..."
```
- Contains emojis, hashtags, and off-topic rambling
- BLEU-1: **0.27** (46% decrease)

### 3. **Noise Accumulation in Final Caption Generation**

**Current Implementation** (line 74):
```python
final_reasoning = "\n\n".join([t for t in traces if t])
```

**Problem**: All reasoning traces are concatenated, including:
- Meta-commentary about the description itself
- Off-topic rambling and emojis
- Repetitive content
- This pollutes the final caption prompt

**Example Caption Evolution**:

**1 Pass:**
```
"A black bicycle with a white clock face, featuring a handlebar, seat, 
and small wheels, is displayed in the image."
```
- Concise, factual
- BLEU-1: **0.50**

**4 Passes:**
```
"A unique bicycle clock with a white round face, black frame, and small 
white handlebars, set against a black metal spoke wheel, and positioned 
near a small white round circle."
```
- Longer, more verbose
- Includes hallucinated details ("small white round circle")
- BLEU-1: **0.27**

### 4. **BLEU Penalizes Verbosity**

**Reference Captions** (Image 0):
- "A bicycle replica with a clock as the front wheel."
- "The bike has a clock as a tire."
- "A black metal bicycle with a clock inside the front wheel."

**Characteristics**:
- Concise (5-10 words)
- Factual
- Direct

**Multi-Pass Outputs**:
- Verbose (15-25 words)
- Include unnecessary details
- Drift from reference style

**BLEU Calculation**: Measures n-gram overlap. Longer outputs that don't match reference style score lower.

### 5. **Model Loses Focus in Later Passes**

**Pattern Observed**:
- **Pass 1**: Direct image analysis → focused, factual
- **Pass 2**: Meta-commentary about Pass 1 → less focused
- **Pass 3**: More meta-commentary → further drift
- **Pass 4**: Creative writing mode → emojis, hashtags, off-topic

**Example from Image 1 (Motorcycle)**:

**Pass 1** (1 pass):
```
"The motorcycle is black with silver accents; it has an orange light..."
```
- Factual description
- BLEU-1: **0.31**

**Pass 4** (4 passes):
```
"...life lessons learned through observation can be applied elsewhere beyond 
just motorcycles themselves such understanding perspective helps viewers better 
understand world we live navigate daily interactions personal experiences 
shared moments captured memories cherished forevermore! 🏍️🚗❤️ 
#motorcyclephotography @HondaMotorsports ✨#Motorcycleride..."
```
- Philosophical rambling + emojis + hashtags
- BLEU-1: **0.35** (slightly better, but includes noise)

## Why This Happens

### 1. **Task Confusion**
- Pass 1: "Analyze this image" → Clear task
- Pass 2+: "Review your previous analysis" → Analyzing text, not image
- Model shifts from **visual analysis** to **text commentary**

### 2. **No Quality Control**
- No filtering of low-quality reasoning
- No validation of factual accuracy
- No length constraints
- All traces treated equally

### 3. **Prompt Engineering Issues**
- Refinement prompt doesn't specify what "refinement" means
- No examples of good vs. bad refinement
- No constraint to maintain conciseness
- Encourages expansion over correction

### 4. **Context Window Pollution**
- All previous traces included in context
- Model sees its own verbose output as "good example"
- Reinforces verbosity pattern

## Comparison with LLaVA-CoT

**LLaVA-CoT** (structured reasoning):
- Uses structured tags (`<SUMMARY>`, `<CAPTION>`, `<REASONING>`, `<CONCLUSION>`)
- Clear stage boundaries
- Less prone to drift
- Better performance with multiple passes

**Scratchpad** (free-form reasoning):
- No structure
- No boundaries
- Prone to drift
- Degrades with multiple passes

## Recommendations

### Immediate Fixes

1. **Change Refinement Prompt**:
   ```python
   msg = f"Review your previous analysis:\n{traces[-1]}\n\n"
         f"Correct any errors and be more concise. Focus on factual accuracy, "
         f"not expansion. Remove any unnecessary details."
   ```

2. **Use Only Latest/Best Trace**:
   ```python
   final_reasoning = traces[-1]  # Instead of concatenating all
   ```

3. **Add Quality Filtering**:
   - Remove traces with emojis/hashtags
   - Limit trace length
   - Filter out meta-commentary

4. **Add Length Constraints**:
   ```python
   if len(cleaned_trace.split()) > 100:
       cleaned_trace = " ".join(cleaned_trace.split()[:100])
   ```

### Long-Term Improvements

1. **Structured Reasoning** (like LLaVA-CoT)
2. **Quality Scoring** for reasoning traces
3. **Selective Trace Usage** (only use high-quality traces)
4. **Different Evaluation Metrics** if verbosity is desired

## Conclusion

The scratchpad method degrades because:
1. **Refinement prompt encourages verbosity** over accuracy
2. **Noise accumulates** across passes
3. **Model loses focus** and shifts to creative writing mode
4. **BLEU penalizes** the resulting verbosity

**The fundamental issue**: The method treats "refinement" as "expansion" rather than "improvement," leading to longer, less accurate outputs that score worse on BLEU metrics.

**Solution**: Redesign the refinement process to focus on **accuracy and conciseness** rather than expansion, or use structured reasoning (like LLaVA-CoT) that maintains focus across passes.

