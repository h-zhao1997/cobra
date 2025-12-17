"""
Dataset loader for LLaVA-CoT-100k with support for dataset proportion sampling.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase


def load_llava_cot_dataset(
    dataset_name: str = "Xkev/LLaVA-CoT-100k",
    dataset_root: Optional[Path] = None,
    dataset_proportion: Optional[float] = None,
    dataset_max_samples: Optional[int] = None,
    dataset_seed: int = 42,
    split: str = "train",
) -> Dataset:
    """
    Load LLaVA-CoT-100k dataset from HuggingFace or local path.
    
    Args:
        dataset_name: HuggingFace dataset name or local path
        dataset_root: Local dataset root directory (if using local data)
        dataset_proportion: Proportion of dataset to use (0.0 to 1.0)
        dataset_max_samples: Maximum number of samples to use (overrides proportion if set)
        dataset_seed: Random seed for sampling
        split: Dataset split to load
        
    Returns:
        HuggingFace Dataset object
    """
    if dataset_root is not None:
        # Load from local path
        dataset_path = Path(dataset_root)
        if (dataset_path / "train.jsonl").exists():
            # Load JSONL file
            data = []
            with open(dataset_path / "train.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset = Dataset.from_list(data)
        else:
            raise ValueError(f"Could not find train.jsonl in {dataset_path}")
    else:
        # Load from HuggingFace
        dataset = load_dataset(dataset_name, split=split)
    
    total_size = len(dataset)
    
    # Determine number of samples to use
    if dataset_max_samples is not None:
        # Use absolute number
        num_samples = min(dataset_max_samples, total_size)
        print(f"Using {num_samples} samples (requested: {dataset_max_samples}, available: {total_size})")
    elif dataset_proportion is not None and dataset_proportion < 1.0:
        # Use proportion
        num_samples = int(total_size * dataset_proportion)
        print(f"Selected {num_samples} samples ({dataset_proportion*100:.1f}%) from {total_size} total samples")
    else:
        # Use all samples
        num_samples = total_size
        print(f"Using all {total_size} samples")
    
    # Apply sampling if needed
    if num_samples < total_size:
        # Set random seed for reproducibility
        generator = random.Random(dataset_seed)
        indices = list(range(total_size))
        generator.shuffle(indices)
        selected_indices = indices[:num_samples]
        dataset = dataset.select(selected_indices)
    
    return dataset


def format_llava_conversation(conversations: List[Dict]) -> str:
    """
    Format LLaVA conversation into a single text string.
    
    Args:
        conversations: List of conversation turns with "from" and "value" keys
        
    Returns:
        Formatted conversation string
    """
    formatted = []
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        
        if role == "human" or role == "user":
            formatted.append(f"USER: {value}")
        elif role == "gpt" or role == "assistant":
            formatted.append(f"ASSISTANT: {value}")
    
    return "\n".join(formatted)


def format_for_sft(
    example: Dict,
    tokenizer: PreTrainedTokenizerBase,
    image_token: str = "<image>",
    max_length: int = 2048,
) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Format a single example for SFTTrainer.
    
    This creates a text string with the conversation, including image tokens.
    The format is compatible with SFTTrainer's text-based training.
    
    Args:
        example: Dataset example with "conversations" and optionally "image" keys
        tokenizer: Tokenizer to use
        image_token: Token to use for images
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with "text" key for SFTTrainer
    """
    conversations = example.get("conversations", [])
    
    # Build the text with image token
    text_parts = []
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        
        # Add image token if this is the first user turn
        if role == "human" or role == "user":
            if image_token in value:
                # Image token already present
                text_parts.append(f"USER: {value}")
            else:
                # Add image token at the beginning of first user message
                if not text_parts:  # First turn
                    text_parts.append(f"USER: {image_token}\n{value}")
                else:
                    text_parts.append(f"USER: {value}")
        elif role == "gpt" or role == "assistant":
            text_parts.append(f"ASSISTANT: {value}")
    
    text = "\n".join(text_parts)
    
    return {"text": text}


class LLaVACoTDataCollator:
    """
    Data collator for LLaVA-CoT dataset that handles images.
    
    Note: This is a simplified version. For full image support,
    you may need to customize SFTTrainer or use a different approach.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor=None,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of examples, each with "text" key
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Extract texts
        texts = [ex["text"] for ex in examples]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokenized["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

