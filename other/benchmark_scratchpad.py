"""
Benchmark script for Cobra VLM with standard inference and two-pass scratchpad reasoning.

This script implements:
1. Standard single-pass inference (baseline)
2. Two-pass scratchpad reasoning mode

Usage:
    python benchmark_scratchpad.py --method baseline --num_samples 100
    python benchmark_scratchpad.py --method scratchpad --num_samples 100
    python benchmark_scratchpad.py --method both --num_samples 100
"""

import argparse
import gc
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image as PILImage
from pycocoevalcap.bleu.bleu import Bleu

from cobra import load


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - reserved
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB / {total:.2f} GB")
            return free
    return None


def check_gpu_memory(min_free_gb: float = 8.0) -> bool:
    """
    Check if there's enough free GPU memory.
    
    Args:
        min_free_gb: Minimum free memory required in GB
    
    Returns:
        True if enough memory is available, False otherwise
    """
    if not torch.cuda.is_available():
        return True
    
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    free = total - reserved
    
    print(f"GPU Memory Status:")
    print(f"  Total: {total:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Free: {free:.2f} GB")
    print(f"  Required: {min_free_gb:.2f} GB")
    
    if free < min_free_gb:
        print(f"\n⚠️  WARNING: Only {free:.2f} GB free, but {min_free_gb:.2f} GB recommended!")
        print("   Consider:")
        print("   1. Closing other processes using the GPU")
        print("   2. Running: nvidia-smi to check GPU usage")
        print("   3. Killing processes: kill <PID> (from nvidia-smi)")
        response = input(f"\nContinue anyway? (y/n): ")
        return response.lower() == 'y'
    
    return True


def generate_standard(
    vlm,
    image: PILImage.Image,
    prompt: str,
    **generation_kwargs
) -> str:
    """
    Standard single-pass inference (baseline method).
    
    Args:
        vlm: The Cobra VLM model
        image: PIL Image to caption
        prompt: Prompt for caption generation
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Generated caption string
    """
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=prompt)
    prompt_text = prompt_builder.get_prompt()
    
    # Filter out scratchpad-specific parameters
    filtered_kwargs = {k: v for k, v in generation_kwargs.items() 
                      if k not in ['reasoning_max_tokens', 'repetition_penalty']}
    
    generated_text = vlm.generate(
        image,
        prompt_text,
        use_cache=True,
        **filtered_kwargs
    )
    
    return generated_text


def detect_repetition(text: str, min_repeat_length: int = 20) -> int:
    """
    Detect repetitive loops in text and return the position to truncate.
    
    Args:
        text: Text to check for repetition
        min_repeat_length: Minimum length of repeated pattern to consider
    
    Returns:
        Position to truncate at, or -1 if no repetition found
    """
    words = text.split()
    if len(words) < min_repeat_length * 2:
        return -1
    
    # Check for repeating patterns at the end
    for i in range(len(words) - min_repeat_length, max(0, len(words) - 200), -1):
        pattern = words[i:i + min_repeat_length]
        if len(pattern) < min_repeat_length:
            continue
        
        # Check if this pattern repeats
        pattern_str = ' '.join(pattern)
        remaining = ' '.join(words[i + min_repeat_length:])
        
        if pattern_str in remaining:
            # Found repetition, truncate before it
            return i
    
    return -1


def clean_reasoning_trace(reasoning: str) -> str:
    """
    Clean up reasoning trace by removing repetitive loops and generic phrases.
    
    Args:
        reasoning: Raw reasoning trace
    
    Returns:
        Cleaned reasoning trace
    """
    # Remove generic unhelpful phrases
    generic_phrases = [
        "Step 1: Identify objects.",
        "Step-by-step analysis:",
        "Step 1:",
    ]
    
    cleaned = reasoning.strip()
    
    # Remove generic phrases at the start
    for phrase in generic_phrases:
        if cleaned.startswith(phrase):
            cleaned = cleaned[len(phrase):].strip()
    
    # Detect and truncate repetitive loops
    truncate_pos = detect_repetition(cleaned)
    if truncate_pos > 0:
        words = cleaned.split()
        cleaned = ' '.join(words[:truncate_pos])
    
    # If reasoning is too short or generic, return empty (will use baseline-like behavior)
    if len(cleaned.split()) < 5:
        return ""
    
    return cleaned.strip()


def generate_scratchpad(
    vlm,
    image: PILImage.Image,
    reasoning_prompt: str,
    caption_prompt: str,
    reasoning_max_tokens: int = 256,
    repetition_penalty: float = 1.2,
    **generation_kwargs
) -> Tuple[str, str]:
    """
    Two-pass scratchpad reasoning mode with improved prompts and repetition handling.
    
    First pass: Generate reasoning trace (step-by-step analysis)
    Second pass: Generate final caption using the reasoning trace
    
    Args:
        vlm: The Cobra VLM model
        image: PIL Image to caption
        reasoning_prompt: Prompt for first pass (reasoning analysis)
        caption_prompt: Prompt for second pass (final caption generation)
        reasoning_max_tokens: Max tokens for reasoning pass (default: 256)
        repetition_penalty: Penalty for repetition in reasoning (default: 1.2)
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Tuple of (reasoning_trace, final_caption)
    """
    # First pass: Generate reasoning trace with separate parameters
    prompt_builder_reasoning = vlm.get_prompt_builder()
    prompt_builder_reasoning.add_turn(role="human", message=reasoning_prompt)
    reasoning_prompt_text = prompt_builder_reasoning.get_prompt()
    
    # Use separate kwargs for reasoning pass
    reasoning_kwargs = generation_kwargs.copy()
    reasoning_kwargs['max_new_tokens'] = reasoning_max_tokens
    reasoning_kwargs['repetition_penalty'] = repetition_penalty
    
    reasoning_trace = vlm.generate(
        image,
        reasoning_prompt_text,
        use_cache=True,
        **reasoning_kwargs
    )
    
    # Clean up reasoning trace
    reasoning_trace = clean_reasoning_trace(reasoning_trace)
    
    # Second pass: Generate final caption using reasoning trace
    prompt_builder_caption = vlm.get_prompt_builder()
    
    # If reasoning is empty or too short, use a simpler prompt
    if not reasoning_trace or len(reasoning_trace.split()) < 5:
        # Fallback to baseline-like behavior
        combined_prompt = caption_prompt
    else:
        # Include the reasoning trace in the prompt
        combined_prompt = (
            f"{caption_prompt}\n\n"
            f"Based on the following detailed analysis:\n{reasoning_trace}\n\n"
            f"Now provide a concise, natural caption that summarizes the key elements:"
        )
    
    prompt_builder_caption.add_turn(role="human", message=combined_prompt)
    caption_prompt_text = prompt_builder_caption.get_prompt()
    
    final_caption = vlm.generate(
        image,
        caption_prompt_text,
        use_cache=True,
        **generation_kwargs
    )
    
    return reasoning_trace, final_caption


def run_benchmark(
    vlm,
    dataset,
    num_samples: int,
    method: str = "baseline",
    use_streaming: bool = True,
    store_images: bool = False,
    **generation_kwargs
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Optional[Dict[int, str]], Optional[Dict[int, PILImage.Image]]]:
    """
    Run benchmark evaluation on COCO dataset.
    
    Args:
        vlm: The Cobra VLM model
        dataset: COCO dataset (streaming or regular)
        num_samples: Number of samples to evaluate
        method: "baseline", "scratchpad", or "both"
        use_streaming: Whether dataset is in streaming mode
        store_images: Whether to store images for visualization
        **generation_kwargs: Generation parameters
    
    Returns:
        Tuple of (ref_captions, generated_captions, reasoning_traces, images)
        reasoning_traces is None if method is "baseline"
        images is None if store_images is False
    """
    # Standard prompts
    standard_prompt = "Please carefully observe the image and come up with a caption for the image."
    
    # Improved scratchpad prompts - more specific and structured
    reasoning_prompt = (
        "Analyze this image systematically. First, identify all major objects and their colors, "
        "sizes, and positions. Then describe the spatial relationships between objects (left, right, "
        "in front of, behind, etc.). Finally, note any distinctive features, actions, or context. "
        "Be specific and detailed. Do NOT write a caption - only provide your analytical observations."
    )
    
    ref_captions = {}
    generated_captions = {}
    reasoning_traces = {} if method in ["scratchpad", "both"] else None
    images = {} if store_images else None
    
    # Handle both streaming and non-streaming datasets
    is_streaming = (
        str(type(dataset).__name__) == 'IterableDataset' or
        (hasattr(dataset, '__iter__') and not hasattr(dataset, 'select'))
    )
    
    if is_streaming and use_streaming:
        dataset_iter = islice(dataset, num_samples)
    else:
        if hasattr(dataset, 'select'):
            dataset_iter = dataset.select(range(min(num_samples, len(dataset))))
        else:
            dataset_iter = islice(dataset, num_samples)
    
    print(f"Running benchmark with method: {method}")
    print(f"Processing {num_samples} samples...")
    
    for idx, example in enumerate(dataset_iter):
        image_id = idx
        image = example["image"]
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Store image if requested
        if store_images:
            images[image_id] = image.copy()
        
        caption = example["answer"]
        ref_captions[image_id] = caption
        
        # Generate caption based on method
        if method == "baseline":
            generated_text = generate_standard(
                vlm,
                image,
                standard_prompt,
                **generation_kwargs
            )
            generated_captions[idx] = [generated_text]
            
        elif method == "scratchpad":
            reasoning_trace, final_caption = generate_scratchpad(
                vlm,
                image,
                reasoning_prompt,
                standard_prompt,
                reasoning_max_tokens=generation_kwargs.get('reasoning_max_tokens', 256),
                repetition_penalty=generation_kwargs.get('repetition_penalty', 1.2),
                **{k: v for k, v in generation_kwargs.items() if k not in ['reasoning_max_tokens', 'repetition_penalty']}
            )
            generated_captions[idx] = [final_caption]
            reasoning_traces[idx] = reasoning_trace
            
        elif method == "both":
            # Run both methods for comparison
            # Filter out scratchpad-specific params for baseline
            baseline_kwargs = {k: v for k, v in generation_kwargs.items() 
                              if k not in ['reasoning_max_tokens', 'repetition_penalty']}
            baseline_caption = generate_standard(
                vlm,
                image,
                standard_prompt,
                **baseline_kwargs
            )
            
            reasoning_trace, scratchpad_caption = generate_scratchpad(
                vlm,
                image,
                reasoning_prompt,
                standard_prompt,
                reasoning_max_tokens=generation_kwargs.get('reasoning_max_tokens', 256),
                repetition_penalty=generation_kwargs.get('repetition_penalty', 1.2),
                **{k: v for k, v in generation_kwargs.items() if k not in ['reasoning_max_tokens', 'repetition_penalty']}
            )
            
            # Store both results (baseline first, then scratchpad)
            generated_captions[idx] = [baseline_caption, scratchpad_caption]
            reasoning_traces[idx] = reasoning_trace
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clear GPU cache periodically
        if (idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print(f"Processed {idx + 1}/{num_samples} images...")
    
    print(f"Completed processing {len(generated_captions)} images")
    return ref_captions, generated_captions, reasoning_traces, images


def compute_bleu_scores(
    ref_captions: Dict[int, List[str]],
    generated_captions: Dict[int, List[str]],
    method: str = "baseline"
) -> Tuple[Tuple[float, float, float, float], Dict]:
    """
    Compute BLEU scores for generated captions.
    
    Args:
        ref_captions: Reference captions dictionary
        generated_captions: Generated captions dictionary
        method: "baseline", "scratchpad", or "both"
    
    Returns:
        Tuple of (BLEU scores tuple, example scores dictionary)
    """
    bleu_scorer = Bleu(4)  # Calculate BLEU scores up to 4-grams
    
    # For "both" method, we need to handle multiple captions per image
    if method == "both":
        # Split into baseline and scratchpad results
        baseline_captions = {k: [v[0]] for k, v in generated_captions.items()}
        scratchpad_captions = {k: [v[1]] for k, v in generated_captions.items()}
        
        # Compute BLEU for baseline
        baseline_scores, baseline_examples = bleu_scorer.compute_score(ref_captions, baseline_captions)
        
        # Compute BLEU for scratchpad
        scratchpad_scores, scratchpad_examples = bleu_scorer.compute_score(ref_captions, scratchpad_captions)
        
        return {
            "baseline": (baseline_scores, baseline_examples),
            "scratchpad": (scratchpad_scores, scratchpad_examples)
        }
    else:
        bleu_scores, example_scores = bleu_scorer.compute_score(ref_captions, generated_captions)
        return (bleu_scores, example_scores)


def compute_per_image_bleu(
    ref_captions: Dict[int, List[str]],
    generated_captions: Dict[int, List[str]],
    method: str = "baseline"
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-image BLEU scores.
    
    Args:
        ref_captions: Reference captions dictionary
        generated_captions: Generated captions dictionary
        method: "baseline", "scratchpad", or "both"
    
    Returns:
        Dictionary mapping image_id to BLEU scores
    """
    bleu_scorer = Bleu(4)
    per_image_scores = {}
    
    if method == "both":
        baseline_captions = {k: [v[0]] for k, v in generated_captions.items()}
        scratchpad_captions = {k: [v[1]] for k, v in generated_captions.items()}
        
        for img_id in generated_captions.keys():
            # Compute baseline BLEU
            baseline_ref = {img_id: ref_captions[img_id]}
            baseline_gen = {img_id: baseline_captions[img_id]}
            baseline_scores, _ = bleu_scorer.compute_score(baseline_ref, baseline_gen)
            
            # Compute scratchpad BLEU
            scratchpad_ref = {img_id: ref_captions[img_id]}
            scratchpad_gen = {img_id: scratchpad_captions[img_id]}
            scratchpad_scores, _ = bleu_scorer.compute_score(scratchpad_ref, scratchpad_gen)
            
            per_image_scores[img_id] = {
                "baseline": {
                    "BLEU-1": baseline_scores[0],
                    "BLEU-2": baseline_scores[1],
                    "BLEU-3": baseline_scores[2],
                    "BLEU-4": baseline_scores[3],
                },
                "scratchpad": {
                    "BLEU-1": scratchpad_scores[0],
                    "BLEU-2": scratchpad_scores[1],
                    "BLEU-3": scratchpad_scores[2],
                    "BLEU-4": scratchpad_scores[3],
                }
            }
    else:
        for img_id in generated_captions.keys():
            ref = {img_id: ref_captions[img_id]}
            gen = {img_id: generated_captions[img_id]}
            scores, _ = bleu_scorer.compute_score(ref, gen)
            per_image_scores[img_id] = {
                "BLEU-1": scores[0],
                "BLEU-2": scores[1],
                "BLEU-3": scores[2],
                "BLEU-4": scores[3],
            }
    
    return per_image_scores


def create_visualization_table(
    images: Dict[int, PILImage.Image],
    ref_captions: Dict[int, List[str]],
    generated_captions: Dict[int, List[str]],
    reasoning_traces: Optional[Dict[int, str]],
    per_image_bleu: Dict[int, Dict],
    output_path: Path,
    method: str,
    max_images: int = 10
):
    """
    Create a visualization table with images, captions, and BLEU scores.
    
    Args:
        images: Dictionary of image_id -> PIL Image
        ref_captions: Reference captions
        generated_captions: Generated captions
        reasoning_traces: Reasoning traces (if available)
        per_image_bleu: Per-image BLEU scores
        output_path: Path to save the visualization
        method: Method used ("baseline", "scratchpad", or "both")
        max_images: Maximum number of images to display
    """
    if not images:
        print("No images available for visualization")
        return
    
    # Limit number of images
    image_ids = sorted(images.keys())[:max_images]
    n_images = len(image_ids)
    
    # Create figure with subplots - wider for better text readability
    # Layout: Each row has image on left (30%), text on right (70%)
    fig = plt.figure(figsize=(24, 6 * n_images))
    
    for idx, img_id in enumerate(image_ids):
        image = images[img_id]
        refs = ref_captions[img_id]
        baseline_cap = generated_captions[img_id][0] if method == "both" else generated_captions[img_id][0]
        scratchpad_cap = generated_captions[img_id][1] if method == "both" else None
        reasoning = reasoning_traces.get(img_id, "") if reasoning_traces else None
        
        # Image subplot (left) - 30% width
        ax_img = plt.subplot(n_images, 2, idx * 2 + 1)
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f'Image {img_id}', fontsize=16, fontweight='bold', pad=15)
        
        # Text subplot (right) - 70% width
        ax_text = plt.subplot(n_images, 2, idx * 2 + 2)
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        
        # Build text content with better spacing
        text_parts = []
        
        # Reference captions
        text_parts.append(("Reference Captions:", "bold", 13))
        for i, ref in enumerate(refs[:3]):  # Show first 3
            # Wrap long captions
            if len(ref) > 80:
                words = ref.split()
                lines = []
                current_line = []
                current_len = 0
                for word in words:
                    if current_len + len(word) + 1 > 80:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_len = len(word)
                    else:
                        current_line.append(word)
                        current_len += len(word) + 1
                if current_line:
                    lines.append(' '.join(current_line))
                for line in lines:
                    text_parts.append((f"  {i+1}. {line}", "normal", 11))
            else:
                text_parts.append((f"  {i+1}. {ref}", "normal", 11))
        if len(refs) > 3:
            text_parts.append((f"  ... ({len(refs) - 3} more)", "italic", 10))
        text_parts.append(("", "normal", 0))  # Spacing
        
        # BLEU scores
        if method == "both":
            bleu_b = per_image_bleu[img_id]["baseline"]
            bleu_s = per_image_bleu[img_id]["scratchpad"]
            text_parts.append(("BLEU Scores:", "bold", 13))
            text_parts.append(("  Baseline:", "bold", 12))
            text_parts.append((f"    BLEU-1: {bleu_b['BLEU-1']:.3f}  BLEU-2: {bleu_b['BLEU-2']:.3f}  BLEU-3: {bleu_b['BLEU-3']:.3f}  BLEU-4: {bleu_b['BLEU-4']:.3f}", "normal", 11))
            text_parts.append(("  Scratchpad:", "bold", 12))
            text_parts.append((f"    BLEU-1: {bleu_s['BLEU-1']:.3f}  BLEU-2: {bleu_s['BLEU-2']:.3f}  BLEU-3: {bleu_s['BLEU-3']:.3f}  BLEU-4: {bleu_s['BLEU-4']:.3f}", "normal", 11))
        else:
            bleu = per_image_bleu[img_id]
            text_parts.append(("BLEU Scores:", "bold", 13))
            text_parts.append((f"  BLEU-1: {bleu['BLEU-1']:.3f}  BLEU-2: {bleu['BLEU-2']:.3f}  BLEU-3: {bleu['BLEU-3']:.3f}  BLEU-4: {bleu['BLEU-4']:.3f}", "normal", 11))
        text_parts.append(("", "normal", 0))  # Spacing
        
        # Baseline caption
        text_parts.append(("Baseline Caption:", "bold", 13))
        # Wrap long captions
        if len(baseline_cap) > 80:
            words = baseline_cap.split()
            lines = []
            current_line = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > 80:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_len = len(word)
                else:
                    current_line.append(word)
                    current_len += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))
            for line in lines:
                text_parts.append((f"  {line}", "normal", 11))
        else:
            text_parts.append((f"  {baseline_cap}", "normal", 11))
        text_parts.append(("", "normal", 0))  # Spacing
        
        # Scratchpad (if available)
        if method in ["scratchpad", "both"]:
            if reasoning:
                text_parts.append(("Scratchpad Reasoning:", "bold", 13))
                # Truncate and wrap long reasoning
                reasoning_short = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                if len(reasoning_short) > 80:
                    words = reasoning_short.split()
                    lines = []
                    current_line = []
                    current_len = 0
                    for word in words:
                        if current_len + len(word) + 1 > 80:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_len = len(word)
                        else:
                            current_line.append(word)
                            current_len += len(word) + 1
                    if current_line:
                        lines.append(' '.join(current_line))
                    for line in lines:
                        text_parts.append((f"  {line}", "italic", 10))
                else:
                    text_parts.append((f"  {reasoning_short}", "italic", 10))
                text_parts.append(("", "normal", 0))  # Spacing
            
            if scratchpad_cap:
                text_parts.append(("Scratchpad Caption:", "bold", 13))
                # Wrap long captions
                if len(scratchpad_cap) > 80:
                    words = scratchpad_cap.split()
                    lines = []
                    current_line = []
                    current_len = 0
                    for word in words:
                        if current_len + len(word) + 1 > 80:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_len = len(word)
                        else:
                            current_line.append(word)
                            current_len += len(word) + 1
                    if current_line:
                        lines.append(' '.join(current_line))
                    for line in lines:
                        text_parts.append((f"  {line}", "normal", 11))
                else:
                    text_parts.append((f"  {scratchpad_cap}", "normal", 11))
        
        # Display text with better spacing and formatting
        y_pos = 0.98
        line_height_base = 0.045  # Base line height
        
        for text, style, fontsize in text_parts:
            if not text:
                y_pos -= 0.03  # Extra spacing for empty lines
                continue
            
            weight = 'bold' if style == "bold" else 'normal'
            style_attr = 'italic' if style == "italic" else 'normal'
            
            # Calculate line height based on font size
            line_height = line_height_base * (fontsize / 11)
            
            # Use text wrapping with proper width
            ax_text.text(0.02, y_pos, text, transform=ax_text.transAxes,
                        fontsize=fontsize, weight=weight, style=style_attr,
                        verticalalignment='top', horizontalalignment='left',
                        family='sans-serif',  # Changed from monospace for better readability
                        wrap=True,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7) if style == "bold" else None)
            
            # Adjust y position based on text length (estimate)
            num_lines = max(1, len(text) // 80 + (1 if len(text) % 80 > 0 else 0))
            y_pos -= line_height * num_lines + 0.01
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    plt.close()


def save_results(
    ref_captions: Dict[int, List[str]],
    generated_captions: Dict[int, List[str]],
    reasoning_traces: Optional[Dict[int, str]],
    bleu_scores,
    output_dir: Path,
    method: str,
    num_samples: int,
    images: Optional[Dict[int, PILImage.Image]] = None,
    per_image_bleu: Optional[Dict[int, Dict]] = None
):
    """
    Save benchmark results to files.
    
    Args:
        ref_captions: Reference captions
        generated_captions: Generated captions
        reasoning_traces: Reasoning traces (if available)
        bleu_scores: BLEU scores
        output_dir: Output directory
        method: Method used ("baseline", "scratchpad", or "both")
        num_samples: Number of samples evaluated
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Handle different output formats based on method
    if method == "both":
        baseline_scores, scratchpad_scores = bleu_scores["baseline"][0], bleu_scores["scratchpad"][0]
        
        # Save combined results file
        output_file = output_dir / f"bleu_scores_{method}_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write(f"Number of generated captions: {num_samples}\n")
            f.write(f"Method: {method}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("BASELINE RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"BLEU-1: {baseline_scores[0]:.4f}\n")
            f.write(f"BLEU-2: {baseline_scores[1]:.4f}\n")
            f.write(f"BLEU-3: {baseline_scores[2]:.4f}\n")
            f.write(f"BLEU-4: {baseline_scores[3]:.4f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("SCRATCHPAD RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"BLEU-1: {scratchpad_scores[0]:.4f}\n")
            f.write(f"BLEU-2: {scratchpad_scores[1]:.4f}\n")
            f.write(f"BLEU-3: {scratchpad_scores[2]:.4f}\n")
            f.write(f"BLEU-4: {scratchpad_scores[3]:.4f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("IMPROVEMENT (Scratchpad - Baseline)\n")
            f.write("=" * 60 + "\n")
            f.write(f"BLEU-1: {scratchpad_scores[0] - baseline_scores[0]:.4f}\n")
            f.write(f"BLEU-2: {scratchpad_scores[1] - baseline_scores[1]:.4f}\n")
            f.write(f"BLEU-3: {scratchpad_scores[2] - baseline_scores[2]:.4f}\n")
            f.write(f"BLEU-4: {scratchpad_scores[3] - baseline_scores[3]:.4f}\n\n")
            
            # Write paired captions
            f.write("=" * 60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 60 + "\n\n")
            for img_id in sorted(generated_captions.keys()):
                f.write(f"Image ID: {img_id}\n")
                f.write("Reference captions:\n")
                for ref in ref_captions[img_id]:
                    f.write(f"  - {ref}\n")
                f.write("Baseline caption:\n")
                f.write(f"  {generated_captions[img_id][0]}\n")
                f.write("Scratchpad reasoning:\n")
                f.write(f"  {reasoning_traces[img_id]}\n")
                f.write("Scratchpad caption:\n")
                f.write(f"  {generated_captions[img_id][1]}\n")
                f.write("\n")
        
        print(f"Results saved to {output_file}")
        
        # Create visualization if images are available
        if images and per_image_bleu:
            viz_path = output_dir / f"visualization_{method}_{timestamp}.png"
            create_visualization_table(
                images, ref_captions, generated_captions, reasoning_traces,
                per_image_bleu, viz_path, method, max_images=min(10, num_samples)
            )
        
    else:
        # Single method results
        if isinstance(bleu_scores, dict):
            scores = bleu_scores["baseline"][0] if method == "baseline" else bleu_scores["scratchpad"][0]
        else:
            scores = bleu_scores[0]
        
        output_file = output_dir / f"bleu_scores_{method}_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write(f"Number of generated captions: {num_samples}\n")
            f.write(f"Method: {method}\n\n")
            f.write("BLEU Scores (BLEU-1 to BLEU-4):\n")
            f.write(f"BLEU-1: {scores[0]:.4f}\n")
            f.write(f"BLEU-2: {scores[1]:.4f}\n")
            f.write(f"BLEU-3: {scores[2]:.4f}\n")
            f.write(f"BLEU-4: {scores[3]:.4f}\n\n")
            
            # Write paired captions
            f.write("=" * 60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 60 + "\n\n")
            for img_id in sorted(generated_captions.keys()):
                f.write(f"Image ID: {img_id}\n")
                f.write("Reference captions:\n")
                for ref in ref_captions[img_id]:
                    f.write(f"  - {ref}\n")
                
                if method == "scratchpad" and reasoning_traces:
                    f.write("Reasoning trace:\n")
                    f.write(f"  {reasoning_traces[img_id]}\n")
                
                f.write("Generated caption:\n")
                caption = generated_captions[img_id][0] if isinstance(generated_captions[img_id], list) else generated_captions[img_id]
                f.write(f"  {caption}\n")
                f.write("\n")
        
        print(f"Results saved to {output_file}")
        
        # Create visualization if images are available
        if images and per_image_bleu:
            viz_path = output_dir / f"visualization_{method}_{timestamp}.png"
            create_visualization_table(
                images, ref_captions, generated_captions, reasoning_traces,
                per_image_bleu, viz_path, method, max_images=min(10, num_samples)
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Cobra VLM with standard and scratchpad inference")
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "scratchpad", "both"],
        default="baseline",
        help="Inference method to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="cobra+3b",
        help="Model ID to load"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=".hf_token",
        help="Path to HuggingFace token file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling during generation"
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode for dataset"
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear GPU cache before loading model"
    )
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=8.0,
        help="Minimum free GPU memory required in GB (default: 8.0)"
    )
    parser.add_argument(
        "--reasoning_max_tokens",
        type=int,
        default=256,
        help="Maximum tokens for reasoning pass (default: 256)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty for reasoning generation (default: 1.2)"
    )
    
    args = parser.parse_args()
    
    # Clear GPU memory if requested
    if args.clear_cache and torch.cuda.is_available():
        print("Clearing GPU cache...")
        clear_gpu_memory()
    
    # Check GPU memory before loading
    if torch.cuda.is_available():
        if not check_gpu_memory(min_free_gb=args.min_free_gb):
            print("Exiting due to insufficient GPU memory.")
            return
    
    # Load HuggingFace token
    hf_token_path = Path(args.hf_token)
    if not hf_token_path.exists():
        raise FileNotFoundError(f"HuggingFace token file not found: {hf_token_path}")
    hf_token = hf_token_path.read_text().strip()
    
    # Setup device and dtype
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model with error handling for OOM
    print(f"Loading model: {args.model_id}")
    try:
        vlm = load(args.model_id, hf_token=hf_token)
        # Clear cache again before moving to device
        if torch.cuda.is_available():
            clear_gpu_memory()
        vlm.to(device, dtype=dtype)
        print("Model loaded successfully!")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory Error!")
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check GPU usage: nvidia-smi")
        print("2. Kill other processes using GPU:")
        print("   - Find PIDs: nvidia-smi")
        print("   - Kill: kill <PID>")
        print("3. Try clearing cache: python benchmark_scratchpad.py --clear_cache ...")
        print("4. Reduce batch size or use CPU mode")
        raise
    
    # Load dataset
    print("Loading COCO dataset...")
    try:
        dataset = load_dataset("lmms-lab/COCO-Caption", split="val", streaming=not args.no_streaming)
        print(f"Dataset loaded (streaming={not args.no_streaming})")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Generation parameters
    generation_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # Run benchmark (store images for visualization)
    ref_captions, generated_captions, reasoning_traces, images = run_benchmark(
        vlm,
        dataset,
        args.num_samples,
        method=args.method,
        use_streaming=not args.no_streaming,
        store_images=True,
        **generation_kwargs
    )
    
    # Compute BLEU scores
    print("Computing BLEU scores...")
    bleu_scores = compute_bleu_scores(ref_captions, generated_captions, method=args.method)
    
    # Compute per-image BLEU scores for visualization
    print("Computing per-image BLEU scores...")
    per_image_bleu = compute_per_image_bleu(ref_captions, generated_captions, method=args.method)
    
    # Print results
    if args.method == "both":
        baseline_scores = bleu_scores["baseline"][0]
        scratchpad_scores = bleu_scores["scratchpad"][0]
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nBaseline BLEU scores:")
        print(f"  BLEU-1: {baseline_scores[0]:.4f}")
        print(f"  BLEU-2: {baseline_scores[1]:.4f}")
        print(f"  BLEU-3: {baseline_scores[2]:.4f}")
        print(f"  BLEU-4: {baseline_scores[3]:.4f}")
        print(f"\nScratchpad BLEU scores:")
        print(f"  BLEU-1: {scratchpad_scores[0]:.4f}")
        print(f"  BLEU-2: {scratchpad_scores[1]:.4f}")
        print(f"  BLEU-3: {scratchpad_scores[2]:.4f}")
        print(f"  BLEU-4: {scratchpad_scores[3]:.4f}")
        print(f"\nImprovement (Scratchpad - Baseline):")
        print(f"  BLEU-1: {scratchpad_scores[0] - baseline_scores[0]:.4f}")
        print(f"  BLEU-2: {scratchpad_scores[1] - baseline_scores[1]:.4f}")
        print(f"  BLEU-3: {scratchpad_scores[2] - baseline_scores[2]:.4f}")
        print(f"  BLEU-4: {scratchpad_scores[3] - baseline_scores[3]:.4f}")
    else:
        if isinstance(bleu_scores, dict):
            scores = bleu_scores["baseline"][0] if args.method == "baseline" else bleu_scores["scratchpad"][0]
        else:
            scores = bleu_scores[0]
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nBLEU scores ({args.method}):")
        print(f"  BLEU-1: {scores[0]:.4f}")
        print(f"  BLEU-2: {scores[1]:.4f}")
        print(f"  BLEU-3: {scores[2]:.4f}")
        print(f"  BLEU-4: {scores[3]:.4f}")
    
    # Save results
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"run_{run_timestamp}"
    output_dir = Path(args.output_dir) / run_dir_name
    print(f"Saving all run artifacts to: {output_dir}")

    save_results(
        ref_captions,
        generated_captions,
        reasoning_traces,
        bleu_scores,
        output_dir,
        args.method,
        args.num_samples,
        images=images,
        per_image_bleu=per_image_bleu
    )


if __name__ == "__main__":
    main()

