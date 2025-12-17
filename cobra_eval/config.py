"""
Configuration and CLI argument parsing.
"""
import argparse
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cobra VLM Benchmark System")
    
    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "scratchpad", "scratchpad_compare", "llava_cot", "gpt5", "gemini", "claude", "llama", "both", "all", "external"],
        default="baseline",
        help="Inference method to use (both=baseline+scratchpad, all=baseline+scratchpad+llava_cot, scratchpad_compare=compare multiple pass counts, external=gpt5+gemini+claude+llama)"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "mmstar"],
        default="coco",
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode for dataset"
    )
    
    # Scratchpad settings
    parser.add_argument(
        "--scratchpad_passes",
        type=int,
        default=1,
        help="Number of reasoning passes for scratchpad and llava_cot methods. For scratchpad_compare, this is the maximum number of passes to compare (default: compares 1-4 passes)"
    )
    
    # Model settings
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
    
    # External model API keys (optional, can also use environment variables)
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--openai_api_key_file",
        type=str,
        default=None,
        help="Path to file containing OpenAI API key"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--gemini_api_key_file",
        type=str,
        default=None,
        help="Path to file containing Gemini API key"
    )
    parser.add_argument(
        "--anthropic_api_key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--anthropic_api_key_file",
        type=str,
        default=None,
        help="Path to file containing Anthropic API key"
    )
    
    # External model-specific settings
    parser.add_argument(
        "--gpt5_model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o, can use gpt-4o-mini or gpt-5 when available)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-1.5-pro",
        help="Gemini model to use (default: gemini-1.5-pro, can use gemini-1.5-flash)"
    )
    parser.add_argument(
        "--claude_model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use (default: claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--llama_model_id",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Llama model ID from HuggingFace (default: meta-llama/Llama-3.2-11B-Vision-Instruct)"
    )
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--reasoning_max_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    # System
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear GPU cache before loading model"
    )
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=8.0,
        help="Minimum free GPU memory required in GB"
    )
    
    # Input/Resume
    parser.add_argument(
        "--load_results",
        type=str,
        default=None,
        help="Path to existing results JSON file to load and resume/skip generation from"
    )

    parser.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help="Do not use the default cached baseline output (cobra_eval/data/baseline_caption_COCO_output.json)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint_interval",
        type=float,
        default=10,
        help="Save checkpoint every N samples (or fraction if < 1.0) (0 to disable checkpointing, default: 10)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Automatically resume from latest checkpoint if available"
    )
    
    # Parallelization
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for inference (default: 1, sequential processing)"
    )
    
    return parser.parse_args()

