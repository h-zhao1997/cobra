"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from PIL import Image
from pathlib import Path
from textwrap import wrap

def create_visualization_from_results(
    results_data: Dict[str, Any],
    images_map: Dict[int, Image.Image],
    output_path: Path,
    max_images: int = 10
):
    """
    Create visualization table from structured results.
    
    Args:
        results_data: The full results dictionary
        images_map: Map of image_id to PIL Image (loaded separately as JSON doesn't store images)
        output_path: Path to save image
        max_images: Limit
    """
    results_list = results_data["results"]
    if not results_list:
        print("No results to visualize")
        return

    # Filter results that have images available
    valid_results = [r for r in results_list if r["image_id"] in images_map]
    valid_results = sorted(valid_results, key=lambda x: x["image_id"])[:max_images]
    
    n_images = len(valid_results)
    if n_images == 0:
        print("No matching images found for visualization")
        return

    # Detect dataset format (MMStar has "question" key, COCO has "reference_captions")
    is_mmstar = "question" in valid_results[0] if valid_results else False

    fig = plt.figure(figsize=(24, 6 * n_images))
    
    for idx, res in enumerate(valid_results):
        img_id = res["image_id"]
        image = images_map[img_id]

        # Image subplot
        ax_img = plt.subplot(n_images, 2, idx * 2 + 1)
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f'Image {img_id}', fontsize=16, fontweight='bold', pad=15)
        
        # Text subplot
        ax_text = plt.subplot(n_images, 2, idx * 2 + 2)
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        
        text_parts = []
        
        if is_mmstar:
            # MMStar format
            question = res.get("question", "")
            ref_answer = res.get("reference_answer", "")
            gen_answer = res.get("generated_answer", "")
            trace = res.get("reasoning_trace")

            # Question
            text_parts.append(("Question:", "bold", 13))
            question_lines = wrap(question, width=80)
            for line in question_lines:
                text_parts.append((f"  {line}", "normal", 11))
            text_parts.append(("", "normal", 0))

            # Reference Answer
            text_parts.append(("Reference Answer:", "bold", 13))
            text_parts.append((f"  {ref_answer}", "normal", 11))
            text_parts.append(("", "normal", 0))

            # Metrics
            if "metrics" in res:
                text_parts.append(("Metrics:", "bold", 13))
                metrics_items = []
                if "MMStar-Correct" in res["metrics"]:
                    correct = res["metrics"]["MMStar-Correct"]
                    status = "✓ Correct" if correct > 0.5 else "✗ Incorrect"
                    metrics_items.append(f"Accuracy: {status}")
                if "MMStar-Predicted" in res["metrics"]:
                    pred = res["metrics"]["MMStar-Predicted"]
                    metrics_items.append(f"Predicted: {pred}")
                if metrics_items:
                    text_parts.append((f"  {' | '.join(metrics_items)}", "normal", 11))
                text_parts.append(("", "normal", 0))

            # Trace
            if trace:
                text_parts.append(("Reasoning Trace:", "bold", 13))
                trace_short = trace[:400] + "..." if len(trace) > 400 else trace
                text_parts.append((f"  {trace_short}", "italic", 10))
                text_parts.append(("", "normal", 0))

            # Generated Answer
            text_parts.append(("Generated Answer:", "bold", 13))
            gen_lines = wrap(gen_answer, width=80)
            for line in gen_lines:
                text_parts.append((f"  {line}", "normal", 11))
        else:
            # COCO format
            refs = res["reference_captions"]
            gen_cap = res["generated_caption"]
            trace = res.get("reasoning_trace")

            # References
            text_parts.append(("Reference Captions:", "bold", 13))
            for i, ref in enumerate(refs[:3]):
                text_parts.append((f"  {i+1}. {ref}", "normal", 11))
            text_parts.append(("", "normal", 0))

            # Metrics
            if "metrics" in res:
                text_parts.append(("Metrics:", "bold", 13))
                metrics_str = "  " + "  ".join([f"{k}: {v:.3f}" for k, v in res["metrics"].items() if isinstance(v, float)])
                text_parts.append((metrics_str, "normal", 11))
                text_parts.append(("", "normal", 0))

            # Trace
            if trace:
                text_parts.append(("Reasoning Trace:", "bold", 13))
                # Truncate
                trace_short = trace[:400] + "..." if len(trace) > 400 else trace
                text_parts.append((f"  {trace_short}", "italic", 10))
                text_parts.append(("", "normal", 0))

            # Caption
            text_parts.append(("Generated Caption:", "bold", 13))
            text_parts.append((f"  {gen_cap}", "normal", 11))
        # References
        text_parts.append(("Reference Captions:", "bold", 13))
        for i, ref in enumerate(refs[:3]):
            text_parts.append((f"  {i+1}. {ref}", "normal", 11))
        text_parts.append(("", "normal", 0))

        # Metrics
        if "metrics" in res:
            text_parts.append(("Metrics:", "bold", 13))
            metrics_str = "  " + "  ".join([f"{k}: {v:.3f}" for k, v in res["metrics"].items() if isinstance(v, float)])
            text_parts.append((metrics_str, "normal", 11))
            text_parts.append(("", "normal", 0))

        # Trace
        if trace:
            text_parts.append(("Reasoning Trace:", "bold", 13))
            # Show full trace with proper wrapping
            trace_lines = wrap(trace, width=100)
            for line in trace_lines:
                text_parts.append((f"  {line}", "italic", 10))
            text_parts.append(("", "normal", 0))

        # Caption
        text_parts.append(("Generated Caption:", "bold", 13))
        # Show full caption with proper wrapping
        caption_lines = wrap(gen_cap, width=100)
        for line in caption_lines:
            text_parts.append((f"  {line}", "normal", 11))

        # Rendering text
        y_pos = 0.98
        line_height_base = 0.045
        
        for text, style, fontsize in text_parts:
            if not text:
                y_pos -= 0.03
                continue
                
            weight = 'bold' if style == "bold" else 'normal'
            style_attr = 'italic' if style == "italic" else 'normal'
            line_height = line_height_base * (fontsize / 11)
            
            # Wrap text properly for better display
            wrapped_lines = wrap(text, width=100)
            num_lines = len(wrapped_lines)

            # Render each line separately for better control
            for line in wrapped_lines:
                ax_text.text(0.02, y_pos, line, transform=ax_text.transAxes,
                            fontsize=fontsize, weight=weight, style=style_attr,
                            verticalalignment='top', horizontalalignment='left',
                            family='sans-serif',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7) if style == "bold" and line == wrapped_lines[0] else None)
                y_pos -= line_height + 0.01
            
            # Add small spacing after each text block
            y_pos -= 0.01

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    plt.close()

def create_comparison_visualization(
    all_methods_data: Dict[str, Dict[str, Any]],
    images_map: Dict[int, Image.Image],
    output_path: Path,
    run_stats: Dict[str, Any] = None,
    config: Dict[str, Any] = None,
    max_images: int = 10
):
    """
    Create side-by-side comparison visualization for all methods.

    Args:
        all_methods_data: Dictionary mapping method names to their result data
        images_map: Map of image_id to PIL Image
        output_path: Path to save the visualization
        run_stats: Optional statistics dictionary
        config: Optional configuration dictionary
        max_images: Maximum number of images to display
    """
    from matplotlib.gridspec import GridSpec

    # Method order and display names
    method_order = ["baseline", "scratchpad", "llava_cot"]
    method_display_names = {
        "baseline": "Baseline",
        "scratchpad": "Scratchpad",
        "llava_cot": "LLaVA-CoT"
    }

    # Filter to only methods that exist
    available_methods = [m for m in method_order if m in all_methods_data]
    
    if len(available_methods) < 2:
        print(f"Need at least 2 methods for comparison, found {len(available_methods)}")
        return

    # Map results by image_id for each method
    method_maps = {}
    for method in available_methods:
        method_maps[method] = {r["image_id"]: r for r in all_methods_data[method]["results"]}

    # Find common images across all methods that exist in images_map
    common_ids = set(method_maps[available_methods[0]].keys())
    for method in available_methods[1:]:
        common_ids &= set(method_maps[method].keys())
    common_ids = common_ids & set(images_map.keys())
    common_ids = sorted(list(common_ids))[:max_images]
    
    if not common_ids:
        print("No common images found for comparison")
        return

    # Detect dataset format
    is_mmstar = "question" in base_map[common_ids[0]] if common_ids else False

    n_images = len(common_ids)
    n_methods = len(available_methods)

    # Calculate figure size: 1 column for image + refs, then 1 column per method
    row_height = 8
    total_height = row_height * n_images
    # Adjust width based on number of methods (image + methods)
    total_width = 8 + (n_methods * 8)

    fig = plt.figure(figsize=(total_width, total_height))
    
    # Use GridSpec: rows for images, columns for image + methods
    n_cols = 1 + n_methods  # 1 for image, rest for methods
    gs = GridSpec(n_images, n_cols, figure=fig, width_ratios=[1.2] + [1.0] * n_methods)
    
    # Use baseline as reference for comparison
    baseline_map = method_maps.get("baseline")
    
    for idx, img_id in enumerate(common_ids):
        row = idx
        image = images_map[img_id]
        
        # Col 0: Image & Refs
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f'Image {img_id}', fontsize=16, fontweight='bold', pad=15)
        
        # Add refs below image (use first available method's refs)
        first_res = method_maps[available_methods[0]][img_id]
        ref_text = "References:\n" + "\n".join([f"- {r}" for r in first_res["reference_captions"][:3]])
        # Add refs/question below image
        if is_mmstar:
            question = base_res.get("question", "")
            ref_answer = base_res.get("reference_answer", "")
            ref_text = f"Question:\n{question[:200]}...\n\nReference: {ref_answer}"
        else:
            ref_text = "References:\n" + "\n".join([f"- {r}" for r in base_res["reference_captions"][:3]])
        ax_img.text(0.05, -0.05, ref_text, transform=ax_img.transAxes, fontsize=11, verticalalignment='top', wrap=True)
        
        # Helper to render text block
        def render_text_block(ax, title, res, compare_with=None):
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            y = 0.95
            
            if is_mmstar:
                # MMStar format
                gen_answer = res.get("generated_answer", "")

                # Generated Answer
                ax.text(0, y, "Generated Answer:", fontweight='bold', fontsize=12)
                y -= 0.04

                gen_lines = wrap(gen_answer, width=50)
                for line in gen_lines:
                    ax.text(0.02, y, line, fontsize=11)
                    y -= 0.03
                y -= 0.02

                # Metrics
                if "metrics" in res:
                    ax.text(0, y, "Metrics:", fontweight='bold', fontsize=12)
                    y -= 0.04

                    if "MMStar-Correct" in res["metrics"]:
                        correct = res["metrics"]["MMStar-Correct"]
                        status = "✓ Correct" if correct > 0.5 else "✗ Incorrect"
                        color = "green" if correct > 0.5 else "red"
                        ax.text(0.02, y, f"Accuracy: {status}", fontsize=11, color=color, fontweight='bold')
                        y -= 0.03

                    if "MMStar-Predicted" in res["metrics"]:
                        pred = res["metrics"]["MMStar-Predicted"]
                        ref = res.get("reference_answer", "N/A")
                        ax.text(0.02, y, f"Predicted: {pred} | Reference: {ref}", fontsize=11)
                        y -= 0.03

                    y -= 0.02
            else:
                # COCO format
                # Caption
                ax.text(0, y, "Caption:", fontweight='bold', fontsize=12)
                y -= 0.04

                cap_lines = wrap(res["generated_caption"], width=50)
                for line in cap_lines:
                    ax.text(0.02, y, line, fontsize=11)
                    y -= 0.03
                y -= 0.02

                # Metrics
                if "metrics" in res:
                    ax.text(0, y, "Metrics:", fontweight='bold', fontsize=12)
                    y -= 0.04

                    # Sort metrics for consistent display
                    sorted_metrics = sorted(res["metrics"].items())

                    for k, v in sorted_metrics:
                        if not isinstance(v, (int, float)): continue

                        text = f"{k}: {v:.3f}"
                        color = "black"

                        # Comparison diff
                        if compare_with and "metrics" in compare_with and k in compare_with["metrics"]:
                            base_v = compare_with["metrics"][k]
                            diff = v - base_v
                            # Only show diff if meaningful
                            if abs(diff) > 0.001:
                                sign = "+" if diff > 0 else ""
                                diff_text = f" ({sign}{diff:.3f})"
                                # Green for positive diff, Red for negative
                                color = "green" if diff > 0 else "red"
                                text += diff_text

                        ax.text(0.02, y, text, fontsize=11, color=color, fontweight='bold' if color != 'black' else 'normal')
                        y -= 0.03
            # Caption
            ax.text(0, y, "Caption:", fontweight='bold', fontsize=12)
            y -= 0.04

            # Show full caption with proper wrapping
            cap_lines = wrap(res["generated_caption"], width=70)
            for line in cap_lines:
                ax.text(0.02, y, line, fontsize=10)
                y -= 0.028
            y -= 0.02

            # Metrics
            if "metrics" in res:
                ax.text(0, y, "Metrics:", fontweight='bold', fontsize=12)
                y -= 0.04

                # Sort metrics for consistent display
                sorted_metrics = sorted(res["metrics"].items())

                for k, v in sorted_metrics:
                    if not isinstance(v, (int, float)): continue

                    text = f"{k}: {v:.3f}"
                    color = "black"

                    # Comparison diff with baseline
                    if compare_with and "metrics" in compare_with and k in compare_with["metrics"]:
                        base_v = compare_with["metrics"][k]
                        diff = v - base_v
                        # Only show diff if meaningful
                        if abs(diff) > 0.001:
                            sign = "+" if diff > 0 else ""
                            diff_text = f" ({sign}{diff:.3f})"
                            # Green for positive diff, Red for negative
                            color = "green" if diff > 0 else "red"
                            text += diff_text

                    ax.text(0.02, y, text, fontsize=10, color=color, fontweight='bold' if color != 'black' else 'normal')
                    y -= 0.028

            y -= 0.02
            
            # Reasoning Trace
            trace = res.get("reasoning_trace")
            if trace:
                ax.text(0, y, "Reasoning Trace:", fontweight='bold', fontsize=12)
                y -= 0.04
                # Show full trace with proper wrapping (limit display length for readability)
                trace_display = trace[:500] + "..." if len(trace) > 500 else trace
                trace_lines = wrap(trace_display, width=70)
                for line in trace_lines[:10]:  # Limit to 10 lines for display
                    ax.text(0.02, y, line, fontsize=8, style='italic')
                    y -= 0.022
                if len(trace_lines) > 10:
                    ax.text(0.02, y, f"... ({len(trace_lines) - 10} more lines)", fontsize=8, style='italic', color='gray')

        # Render each method
        for col_idx, method in enumerate(available_methods):
            method_res = method_maps[method][img_id]
            ax_method = fig.add_subplot(gs[row, col_idx + 1])
            display_name = method_display_names.get(method, method.title())
            render_text_block(ax_method, display_name, method_res, compare_with=baseline_map[img_id] if baseline_map else None)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Comparison visualization saved to {output_path}")
    plt.close()
