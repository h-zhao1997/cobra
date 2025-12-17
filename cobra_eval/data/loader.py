"""
Dataset loading and management.
"""
from datasets import load_dataset
from itertools import islice
from typing import Optional, Iterator, Dict, Any, Tuple
from PIL import Image

class COCODatasetLoader:
    def __init__(self, 
                 split: str = "val", 
                 streaming: bool = True,
                 limit: Optional[int] = None):
        """
        Initialize COCO dataset loader.
        
        Args:
            split: Dataset split ("val", "test", etc.)
            streaming: Whether to use streaming mode
            limit: Maximum number of samples to load
        """
        self.dataset = load_dataset("lmms-lab/COCO-Caption", split=split, streaming=streaming)
        self.limit = limit
        self.streaming = streaming
        
        # If not streaming and limit is set, we can select upfront
        if not streaming and limit:
            self.dataset = self.dataset.select(range(limit))

    def __iter__(self) -> Iterator[Tuple[int, Image.Image, Any]]:
        """
        Iterate over dataset samples.
        
        Returns:
            Iterator yielding (image_id, image, reference_captions)
        """
        iterator = iter(self.dataset)
        if self.limit and self.streaming:
            iterator = islice(iterator, self.limit)
            
        for idx, example in enumerate(iterator):
            image = example["image"]
            
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # For COCO-Caption from lmms-lab, 'answer' contains the reference captions?
            # Checking previous code: caption = example["answer"]
            # And ref_captions[image_id] = caption
            # It seems 'answer' is a list of strings or a single string?
            # In benchmark.py: caption = example["answer"] -> ref_captions[image_id] = caption
            # It seems typically it's a list for COCO evaluation.
            # Let's assume it returns what we need.
            
            yield idx, image, example["answer"]


class MMStarDatasetLoader:
    def __init__(self, 
                 split: str = "test", 
                 streaming: bool = True,
                 limit: Optional[int] = None):
        """
        Initialize MMStar dataset loader.
        
        Args:
            split: Dataset split ("test" is the main split for MMStar)
            streaming: Whether to use streaming mode
            limit: Maximum number of samples to load
        """
        self.dataset = load_dataset("Lin-Chen/MMStar", split=split, streaming=streaming)
        self.limit = limit
        self.streaming = streaming
        
        # If not streaming and limit is set, we can select upfront
        if not streaming and limit:
            self.dataset = self.dataset.select(range(limit))

    def __iter__(self) -> Iterator[Tuple[int, Image.Image, str, str]]:
        """
        Iterate over dataset samples.
        
        Returns:
            Iterator yielding (image_id, image, question_prompt, reference_answer)
            where question_prompt contains the question with all options embedded,
            and reference_answer is a single letter (A, B, C, or D)
        """
        iterator = iter(self.dataset)
        if self.limit and self.streaming:
            iterator = islice(iterator, self.limit)
            
        for idx, example in enumerate(iterator):
            image = example["image"]
            
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # MMStar dataset structure:
            # - 'question' contains the question text with options embedded
            # - 'answer' contains the reference answer (single letter)
            question_prompt = example.get("question", "")
            reference_answer = example.get("answer", "")
            
            yield idx, image, question_prompt, reference_answer

