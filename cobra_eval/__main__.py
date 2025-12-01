"""
Main entry point for Cobra Evaluation System.
"""
import torch
from pathlib import Path
from datetime import datetime
from cobra import load

from .config import parse_args
from .registry import Registry
from .data.loader import COCODatasetLoader, MMStarDatasetLoader
from .utils.gpu import clear_gpu_memory, check_gpu_memory
from .utils.io import save_json_results, load_json_results, find_latest_result_file, save_checkpoint, find_latest_checkpoint
from .utils.viz import create_visualization_from_results, create_comparison_visualization


from .generators import baseline, scratchpad, scratchpad_compare, external, llava_cot, external_models
from .metrics import bleu, bert_score, mmstar_accuracy

def main():
    args = parse_args()
    
    # 1. GPU Setup
    if args.clear_cache and torch.cuda.is_available():
        print("Clearing GPU cache...")
        clear_gpu_memory()
        
    if torch.cuda.is_available():
        if not check_gpu_memory(min_free_gb=args.min_free_gb):
            print("Exiting due to insufficient GPU memory.")
            return
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Determine methods to run
    if args.method == "both":
        methods_to_run = ["baseline", "scratchpad"]
    elif args.method == "all":
        methods_to_run = ["baseline", "scratchpad", "llava_cot"]
    elif args.method == "external":
        methods_to_run = ["gpt5", "gemini", "claude", "llama"]
    else:
        methods_to_run = [args.method]

    # 2. Load Model (only for non-external methods)
    external_methods = ["gpt5", "gemini", "claude", "llama"]
    vlm = None

    if not any(m in external_methods for m in methods_to_run):
        print(f"Loading model: {args.model_id}")
        try:
            hf_token_path = Path(args.hf_token)
            if hf_token_path.exists():
                hf_token = hf_token_path.read_text().strip()
            else:
                hf_token = None

            vlm = load(args.model_id, hf_token=hf_token)
            vlm.to(device, dtype=dtype)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Using external models - skipping local model loading")

    # Create a unique directory for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"run_{run_timestamp}"
    base_run_dir = Path(args.output_dir) / run_dir_name
    base_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving all run artifacts to: {base_run_dir}")

    all_run_data = {}
    shared_images_map = {}  # Shared across methods for comparison

    # Loop over methods
    for current_method in methods_to_run:
        print(f"\n{'='*40}")
        print(f"Running Method: {current_method}")
        print(f"{'='*40}\n")
        
        # Use method-specific subdirectories for cleaner organization
        result_dir = base_run_dir / current_method
        result_dir.mkdir(parents=True, exist_ok=True)

        # 3. Initialize Generator
        generator_cls = Registry.get_generator(current_method)
        
        # Handle different init signatures
        if current_method == "scratchpad_compare":
            # Compare mode: compare multiple pass counts (1, 2, 3, 4 up to scratchpad_passes)
            max_passes = max(4, args.scratchpad_passes)  # Default to comparing 1-4 passes
            generator = generator_cls(vlm, scratchpad_passes=max_passes)
        elif current_method in ["scratchpad", "llava_cot"]:
            generator = generator_cls(vlm, scratchpad_passes=args.scratchpad_passes)
        elif current_method == "gpt5":
            generator = generator_cls(
                api_key=args.openai_api_key,
                model=args.gpt5_model,
                api_key_file=args.openai_api_key_file
            )
        elif current_method == "gemini":
            generator = generator_cls(
                api_key=args.gemini_api_key,
                model=args.gemini_model,
                api_key_file=args.gemini_api_key_file
            )
        elif current_method == "claude":
            generator = generator_cls(
                api_key=args.anthropic_api_key,
                model=args.claude_model,
                api_key_file=args.anthropic_api_key_file
            )
        elif current_method == "llama":
            hf_token_path = Path(args.hf_token)
            hf_token = hf_token_path.read_text().strip() if hf_token_path.exists() else None
            generator = generator_cls(
                model_id=args.llama_model_id,
                hf_token=hf_token,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            generator = generator_cls(vlm)
            
        print(f"Initialized {current_method} generator")

        # 4. Load Dataset
        print(f"Loading dataset: {args.dataset} ({args.num_samples} samples)...")
        if args.dataset == "mmstar":
            data_loader = MMStarDatasetLoader(
                split="val",
                streaming=not args.no_streaming,
                limit=args.num_samples
            )
            is_mmstar = True
        else:  # coco
            data_loader = COCODatasetLoader(
                split="val",
                streaming=not args.no_streaming,
                limit=args.num_samples
            )
            is_mmstar = False

        # 4b. Load Cached Results (if any)
        loaded_results_map = {}

        path_to_load = args.load_results

        # Check for checkpoint resume
        if args.resume_from_checkpoint and not path_to_load:
            checkpoint_path = find_latest_checkpoint(result_dir, f"checkpoint_{current_method}")
            if checkpoint_path:
                print(f"Found checkpoint: {checkpoint_path}")
                path_to_load = str(checkpoint_path)

        # Check for default baseline cache
        if current_method == "baseline" and not args.no_baseline_cache and not path_to_load:
            default_cache = Path(__file__).parent / "data" / "baseline_caption_COCO_output.json"
            if default_cache.exists():
                print(f"Using default baseline cache: {default_cache}")
                path_to_load = str(default_cache)

        if path_to_load:
            load_path = None
            
            # Resolve path
            if path_to_load.lower() == "latest":
                prefix = f"results_{current_method}"
                load_path = find_latest_result_file(result_dir, prefix)
                if not load_path:
                    print(f"No previous results found for method '{current_method}' in {result_dir}")
            else:
                p = Path(path_to_load)
                if p.is_dir():
                    prefix = f"results_{current_method}"
                    load_path = find_latest_result_file(p, prefix)
                else:
                    load_path = p
                    
            if load_path and load_path.exists():
                print(f"Loading results from {load_path}...")
                try:
                    loaded_data = load_json_results(load_path)

                    # Handle different schemas (ours vs baseline cache)
                    results_list = loaded_data.get("results", [])
                    if not results_list:
                        results_list = loaded_data.get("images", [])

                    for r in results_list:
                        # Extract ID (supports 'image_id' or 'id')
                        raw_id = r.get("image_id", r.get("id"))
                        if raw_id is not None:
                            # Try to convert to int for matching with loader
                            try:
                                key_id = int(raw_id)
                            except (ValueError, TypeError):
                                key_id = raw_id

                            loaded_results_map[key_id] = r

                            # Normalize the record for our use
                            if "generated_caption" not in r and "caption" in r:
                                r["generated_caption"] = r["caption"]

                    # Check if this is a checkpoint
                    is_checkpoint = loaded_data.get("meta", {}).get("is_checkpoint", False)
                    if is_checkpoint:
                        checkpoint_info = loaded_data.get("checkpoint_info", {})
                        processed = checkpoint_info.get("processed_count", len(loaded_results_map))
                        remaining = checkpoint_info.get("remaining_count", args.num_samples - processed)
                        print(f"Loaded checkpoint: {processed} samples processed, {remaining} remaining")
                    else:
                        print(f"Loaded {len(loaded_results_map)} cached results.")
                except Exception as e:
                    print(f"Error loading results file: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing without cache...")
            elif path_to_load and path_to_load != args.load_results:
                # Silent fail if default cache is missing/invalid logic?
                # Actually we checked exists() for default cache above.
                pass
            elif path_to_load:
                 print(f"Warning: Results file not found: {path_to_load}")

        # 5. Run Evaluation
        results = []
        references_map = {}  # For global metric computation
        predictions_map = {}
        images_map = {}      # Keep images for visualization (method-specific)
        
        start_time = datetime.now()
        print(f"Starting generation at {start_time.strftime('%H:%M:%S')}...")

        if is_mmstar:
            # MMStar dataset format: (image_id, image, question_prompt, reference_answer)
            for i, (image_id, image, question_prompt, reference_answer) in enumerate(data_loader):
                # Store image for visualization (limit to save memory)
                if len(images_map) < 20:
                    images_map[image_id] = image.copy()
                    # Also store in shared map for comparison
                    if image_id not in shared_images_map:
                        shared_images_map[image_id] = image.copy()

                # Check cache first
                if args.load_results and image_id in loaded_results_map:
                    print(f"Using cached result for image {image_id}")
                    cached = loaded_results_map[image_id]

                    # Store for metrics
                    references_map[image_id] = [cached["reference_answer"]]
                    predictions_map[image_id] = [cached["generated_answer"]]

                    results.append(cached)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i + 1}/{args.num_samples} samples (Cached)...")
                    continue

                # Generate using question prompt
                gen_result = generator.generate(
                    image,
                    question_prompt,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    reasoning_max_tokens=args.reasoning_max_tokens,
                    repetition_penalty=args.repetition_penalty
                )

                # Store for metrics
                references_map[image_id] = [reference_answer]
                predictions_map[image_id] = [gen_result.caption]

                # Record result for MMStar
                results.append({
                    "image_id": image_id,
                    "question": question_prompt,
                    "reference_answer": reference_answer,
                    "generated_answer": gen_result.caption,
                    "reasoning_trace": gen_result.reasoning_trace,
                    "metadata": gen_result.metadata,
                    "metrics": {}  # Populated later
                })

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i + 1}/{args.num_samples} samples...")
                if (i + 1) % 10 == 0:
                    clear_gpu_memory()
        else:
            # COCO dataset format: (image_id, image, refs)
            for i, (image_id, image, refs) in enumerate(data_loader):
                # Store image for visualization (limit to save memory)
                if len(images_map) < 20:
                    images_map[image_id] = image.copy()
                    # Also store in shared map for comparison
                    if image_id not in shared_images_map:
                        shared_images_map[image_id] = image.copy()

                # Clean up refs (ensure list of strings)
                if isinstance(refs, str):
                    refs = [refs]

            # Check cache first
            if image_id in loaded_results_map:
                print(f"Using cached result for image {image_id}")
                cached = loaded_results_map[image_id]

                # Store for metrics
                references_map[image_id] = refs
                predictions_map[image_id] = [cached["generated_caption"]]

                # Ensure the cached result has the necessary structure for downstream processing
                if "metrics" not in cached:
                    cached["metrics"] = {}
                if "image_id" not in cached:
                    cached["image_id"] = image_id

                # If we want to prefer the fresh references from the loader over the cached ones
                # (to ensure consistency if the dataset version changed slightly), we can update it:
                cached["reference_captions"] = refs

                results.append(cached)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i + 1}/{args.num_samples} samples (Cached)...")
                continue

            # Generate
            gen_result = generator.generate(
                image,
                "Please carefully observe the image and come up with a caption for the image.",
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                reasoning_max_tokens=args.reasoning_max_tokens,
                repetition_penalty=args.repetition_penalty
            )

            # Store for metrics
            references_map[image_id] = refs

            # For scratchpad_compare, compute metrics for each pass count
            if current_method == "scratchpad_compare" and "comparison" in gen_result.metadata:
                # Store predictions for each pass count
                comparison_data = gen_result.metadata.get("comparison", {})
                for passes, comp_data in comparison_data.items():
                    pass_key = f"pass_{passes}"
                    if pass_key not in predictions_map:
                        predictions_map[pass_key] = {}
                    predictions_map[pass_key][image_id] = [comp_data["caption"]]

                # Use best pass caption for main prediction map
                predictions_map[image_id] = [gen_result.caption]
            else:
                predictions_map[image_id] = [gen_result.caption]

            # Record result
            results.append({
                "image_id": image_id,
                "reference_captions": refs,
                "generated_caption": gen_result.caption,
                "reasoning_trace": gen_result.reasoning_trace,
                "metadata": gen_result.metadata,
                "metrics": {} # Populated later
            })

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i + 1}/{args.num_samples} samples...")

            # Save checkpoint periodically
            effective_interval = args.checkpoint_interval
            if 0 < effective_interval <= 1.0:
                effective_interval = int(effective_interval * args.num_samples)
            effective_interval = max(1, int(effective_interval))

            if args.checkpoint_interval > 0 and (i + 1) % effective_interval == 0:
                # Compute partial metrics for checkpoint
                partial_references = {img_id: references_map[img_id] for img_id in references_map.keys()}
                partial_predictions = {img_id: predictions_map[img_id] for img_id in predictions_map.keys() if img_id in partial_references}

                # Create checkpoint data
                checkpoint_data = {
                    "meta": {
                        "timestamp": datetime.now().isoformat(),
                        "config": vars(args),
                        "method": current_method,
                        "samples_processed": i + 1,
                        "total_samples": args.num_samples,
                        "is_checkpoint": True
                    },
                    "results": results,
                    "checkpoint_info": {
                        "processed_count": i + 1,
                        "remaining_count": args.num_samples - (i + 1)
                    }
                }

                checkpoint_path = save_checkpoint(
                    checkpoint_data,
                    result_dir,
                    f"checkpoint_{current_method}",
                    sample_count=i + 1,
                    overwrite=True
                )
                print(f"  → Checkpoint saved: {checkpoint_path.name} ({i + 1}/{args.num_samples} samples)")

            if (i + 1) % 10 == 0:
                clear_gpu_memory()

        # 6. Compute Metrics
        aggregate_metrics = {}
        
        if is_mmstar:
            # MMStar accuracy metric
            print("Computing MMStar accuracy metrics...")
            try:
                accuracy_metric = Registry.get_metric("mmstar_accuracy")()
                scores = accuracy_metric.compute(references_map, predictions_map)
                aggregate_metrics.update(scores)

                # Store per-sample scores
                if hasattr(accuracy_metric, "per_sample_scores"):
                    for res in results:
                        img_id = res["image_id"]
                        if img_id in accuracy_metric.per_sample_scores:
                            res["metrics"].update(accuracy_metric.per_sample_scores[img_id])

                print(f"MMStar Accuracy: {scores.get('MMStar-Accuracy', 0.0):.3f} ({scores.get('MMStar-Correct', 0)}/{scores.get('MMStar-Total', 0)})")
            except Exception as e:
                print(f"Error computing MMStar accuracy metric: {e}")
        else:
            # COCO metrics
            print("Computing global metrics...")

            # Instantiate all registered metrics
            metric_instances = []
            try:
                metric_instances.append(Registry.get_metric("bleu")())
                metric_instances.append(Registry.get_metric("bert_score")())
            except Exception as e:
                print(f"Warning: Could not initialize some metrics: {e}")

        for metric in metric_instances:
            try:
                scores = metric.compute(references_map, predictions_map)
                aggregate_metrics.update(scores)
                
                # For scratchpad_compare, compute metrics for each pass count
                if current_method == "scratchpad_compare":
                    # Compute metrics for each pass count separately
                    comparison_metrics = {}
                    for pass_key in predictions_map.keys():
                        if pass_key.startswith("pass_"):
                            pass_predictions = {img_id: predictions_map[pass_key][img_id]
                                              for img_id in references_map.keys()
                                              if img_id in predictions_map[pass_key]}
                            if pass_predictions:
                                pass_scores = metric.compute(references_map, pass_predictions)
                                comparison_metrics[pass_key] = pass_scores
                    aggregate_metrics["comparison_by_passes"] = comparison_metrics

                # Hack: re-compute per sample for BLEU since it's fast
                if isinstance(metric, Registry.get_metric("bleu")):
                    for res in results:
                        img_id = res["image_id"]
                        single_ref = {img_id: references_map[img_id]}
                        single_pred = {img_id: predictions_map[img_id]}
                        sample_score = metric.compute(single_ref, single_pred)
                        res["metrics"].update(sample_score)

                        # For scratchpad_compare, also compute per-sample metrics for each pass
                        if current_method == "scratchpad_compare" and "comparison" in res.get("metadata", {}):
                            for passes, comp_data in res["metadata"]["comparison"].items():
                                pass_key = f"pass_{passes}"
                                if pass_key in predictions_map and img_id in predictions_map[pass_key]:
                                    pass_pred = {img_id: predictions_map[pass_key][img_id]}
                                    pass_sample_score = metric.compute(single_ref, pass_pred)
                                    if "comparison_metrics" not in res["metrics"]:
                                        res["metrics"]["comparison_metrics"] = {}
                                    if pass_key not in res["metrics"]["comparison_metrics"]:
                                        res["metrics"]["comparison_metrics"][pass_key] = {}
                                    res["metrics"]["comparison_metrics"][pass_key].update(pass_sample_score)
                
                # Retrieve per-sample scores for BERTScore
                if isinstance(metric, Registry.get_metric("bert_score")) and hasattr(metric, "per_sample_scores"):
                    for res in results:
                        img_id = res["image_id"]
                        if img_id in metric.per_sample_scores:
                            res["metrics"].update(metric.per_sample_scores[img_id])
                        
            except Exception as e:
                print(f"Error computing metric {metric.__class__.__name__}: {e}")

        # 7. Save Results
        # Create a copy of args with the correct method name for the record
        run_config = vars(args).copy()
        run_config["method"] = current_method
        
        output_data = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "config": run_config
            },
            "aggregate_metrics": aggregate_metrics,
            "results": results
        }
        
        output_path = save_json_results(output_data, result_dir, f"results_{current_method}")
        
        # 8. Visualize
        viz_path = output_path.parent / f"viz_{output_path.stem}.png"
        print("Generating visualization...")
        create_visualization_from_results(output_data, images_map, viz_path)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nEvaluation for {current_method} completed in {duration}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Store for comparison
        all_run_data[current_method] = output_data

    # Comparison Visualization - Compare all methods
    if len(all_run_data) >= 2:
        print("\nComputing comparison statistics...")
        
        # Map results by image_id for each method
        method_results = {}
        for method, data in all_run_data.items():
            method_results[method] = {r["image_id"]: r for r in data["results"]}
        
        # Find common images across all methods
        common_ids = set(method_results[list(method_results.keys())[0]].keys())
        for method in method_results.keys():
            common_ids &= set(method_results[method].keys())
        total_common = len(common_ids)
        
        # Check if this is MMStar format
        is_mmstar_comp = "question" in base_res[list(common_ids)[0]] if common_ids else False

        if is_mmstar_comp:
            # For MMStar, compute accuracy comparison
            base_accuracy = all_run_data["baseline"]["aggregate_metrics"].get("MMStar-Accuracy", 0.0)
            scratch_accuracy = all_run_data["scratchpad"]["aggregate_metrics"].get("MMStar-Accuracy", 0.0)
            accuracy_diff = scratch_accuracy - base_accuracy

            base_correct = all_run_data["baseline"]["aggregate_metrics"].get("MMStar-Correct", 0)
            scratch_correct = all_run_data["scratchpad"]["aggregate_metrics"].get("MMStar-Correct", 0)
            base_total = all_run_data["baseline"]["aggregate_metrics"].get("MMStar-Total", 0)
            scratch_total = all_run_data["scratchpad"]["aggregate_metrics"].get("MMStar-Total", 0)

            # Compute per-sample differences
            accuracy_wins = 0
            for img_id in common_ids:
                base_correct_sample = base_res[img_id].get("metrics", {}).get("MMStar-Correct", 0.0)
                scratch_correct_sample = scratch_res[img_id].get("metrics", {}).get("MMStar-Correct", 0.0)
                if scratch_correct_sample > base_correct_sample:
                    accuracy_wins += 1

            win_rate = (accuracy_wins / total_common * 100) if total_common > 0 else 0.0

            run_stats = {
                "comparison": "scratchpad (A) vs baseline (B)",
                "diff_meaning": "A - B",
                "total_samples": total_common,
                "baseline_accuracy": base_accuracy,
                "scratchpad_accuracy": scratch_accuracy,
                "accuracy_diff": accuracy_diff,
                "baseline_correct": base_correct,
                "scratchpad_correct": scratch_correct,
                "baseline_total": base_total,
                "scratchpad_total": scratch_total,
                "accuracy_win_rate": win_rate
            }
        else:
            # Compute differences and win rates for COCO
            metrics_sum_diff = {}
            metrics_wins = {}

            for img_id in common_ids:
                base_metrics = base_res[img_id].get("metrics", {})
                scratch_metrics = scratch_res[img_id].get("metrics", {})

                for k, v_base in base_metrics.items():
                    if not isinstance(v_base, (int, float)): continue
                    if k not in scratch_metrics: continue
                    v_scratch = scratch_metrics[k]

                    diff = v_scratch - v_base
                    metrics_sum_diff[k] = metrics_sum_diff.get(k, 0.0) + diff

                    if diff > 0:
                        metrics_wins[k] = metrics_wins.get(k, 0) + 1

            # Averages
            avg_diffs = {k: v / total_common for k, v in metrics_sum_diff.items()} if total_common > 0 else {}
            win_rates = {k: (v / total_common) * 100 for k, v in metrics_wins.items()} if total_common > 0 else {}

            run_stats = {
                "comparison": "scratchpad (A) vs baseline (B)",
                "diff_meaning": "A - B",
                "total_samples": total_common,
                "aggregate_diffs": avg_diffs,
                "win_rates": win_rates
            }
        # Compute aggregate metrics for each method
        method_aggregates = {}
        for method, results_map in method_results.items():
            method_aggregates[method] = {}
            for metric_name in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BERTScore-Precision", "BERTScore-Recall", "BERTScore-F1"]:
                values = []
                for img_id in common_ids:
                    metrics = results_map[img_id].get("metrics", {})
                    if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
                        values.append(metrics[metric_name])
                if values:
                    method_aggregates[method][metric_name] = sum(values) / len(values)

        # Compute pairwise comparisons vs baseline
        baseline_method = "baseline" if "baseline" in method_results else list(method_results.keys())[0]
        comparisons = {}

        for method in method_results.keys():
            if method == baseline_method:
                continue

            metrics_sum_diff = {}
            metrics_wins = {}

            for img_id in common_ids:
                base_metrics = method_results[baseline_method][img_id].get("metrics", {})
                method_metrics = method_results[method][img_id].get("metrics", {})

                for k, v_base in base_metrics.items():
                    if not isinstance(v_base, (int, float)): continue
                    if k not in method_metrics: continue
                    v_method = method_metrics[k]

                    diff = v_method - v_base
                    metrics_sum_diff[k] = metrics_sum_diff.get(k, 0.0) + diff

                    if diff > 0:
                        metrics_wins[k] = metrics_wins.get(k, 0) + 1

            # Averages
            avg_diffs = {k: v / total_common for k, v in metrics_sum_diff.items()} if total_common > 0 else {}
            win_rates = {k: (v / total_common) * 100 for k, v in metrics_wins.items()} if total_common > 0 else {}

            comparisons[method] = {
                "vs_baseline": avg_diffs,
                "win_rates": win_rates
            }

        run_stats = {
            "baseline_method": baseline_method,
            "total_samples": total_common,
            "method_aggregates": method_aggregates,
            "comparisons": comparisons
        }

        # Save summary JSON
        save_json_results(run_stats, base_run_dir, "comparison_summary")
        
        print("\nGenerating comparison visualization...")
        comp_path = base_run_dir / f"comparison_all_methods_{run_timestamp}.png"
        
        # Use first method's config (or scratchpad if available)
        viz_config = all_run_data.get("scratchpad", list(all_run_data.values())[0])["meta"]["config"]
        viz_config["timestamp"] = run_timestamp
        
        create_comparison_visualization(
            all_run_data,
            images_map, 
            comp_path,
            run_stats=run_stats,
            config=viz_config
        )

if __name__ == "__main__":
    main()
