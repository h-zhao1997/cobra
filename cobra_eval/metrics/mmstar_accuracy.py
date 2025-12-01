"""
MMStar Accuracy Metric implementation.
"""
import re
from typing import Dict, List
from ..interfaces import BaseMetric
from ..registry import Registry

@Registry.register_metric("mmstar_accuracy")
class MMStarAccuracyMetric(BaseMetric):
    def __init__(self):
        """Initialize MMStar accuracy metric."""
        self.per_sample_scores = {}

    @staticmethod
    def extract_answer_letter(generated_text: str) -> str:
        """
        Extract the predicted answer letter (A, B, C, or D) from generated text.
        
        Args:
            generated_text: The generated answer text (e.g., "B: The suitcase is beneath the cat.")
            
        Returns:
            The extracted letter (A, B, C, or D) or None if not found
        """
        if not generated_text:
            return None
        
        # Strip whitespace
        text = generated_text.strip()
        
        # Pattern 1: Look for "A:", "B:", "C:", "D:" at the start
        pattern1 = re.match(r'^([A-D]):', text, re.IGNORECASE)
        if pattern1:
            return pattern1.group(1).upper()
        
        # Pattern 2: Look for "A.", "B.", "C.", "D." at the start
        pattern2 = re.match(r'^([A-D])\.', text, re.IGNORECASE)
        if pattern2:
            return pattern2.group(1).upper()
        
        # Pattern 3: Look for "A ", "B ", "C ", "D " at the start
        pattern3 = re.match(r'^([A-D])\s', text, re.IGNORECASE)
        if pattern3:
            return pattern3.group(1).upper()
        
        # Pattern 4: First character is A-D
        if text and text[0].upper() in ['A', 'B', 'C', 'D']:
            return text[0].upper()
        
        # Pattern 5: Look for any occurrence of "A:", "B:", "C:", "D:" in the text
        pattern5 = re.search(r'\b([A-D]):', text, re.IGNORECASE)
        if pattern5:
            return pattern5.group(1).upper()
        
        return None

    def compute(self, references: Dict[int, List[str]], predictions: Dict[int, List[str]]) -> Dict[str, float]:
        """
        Compute accuracy metric for MMStar dataset.
        
        Args:
            references: Dict mapping image_id to list containing reference answer (single letter)
            predictions: Dict mapping image_id to list containing generated answer (full text)
            
        Returns:
            Dictionary with accuracy metrics
        """
        sorted_ids = sorted(predictions.keys())
        
        correct_count = 0
        total_count = 0
        self.per_sample_scores = {}
        
        for img_id in sorted_ids:
            if img_id not in references:
                continue
                
            # Get reference answer (should be a single letter)
            ref_answer = references[img_id][0].strip().upper() if references[img_id] else None
            
            # Get generated answer (full text)
            gen_answer = predictions[img_id][0] if predictions[img_id] else ""
            
            # Extract predicted letter
            pred_letter = self.extract_answer_letter(gen_answer)
            
            # Compare
            is_correct = False
            if ref_answer and pred_letter:
                is_correct = (ref_answer.upper() == pred_letter.upper())
            
            # Store per-sample score
            self.per_sample_scores[img_id] = {
                "MMStar-Correct": 1.0 if is_correct else 0.0,
                "MMStar-Predicted": pred_letter if pred_letter else "N/A",
                "MMStar-Reference": ref_answer if ref_answer else "N/A"
            }
            
            if is_correct:
                correct_count += 1
            total_count += 1
        
        # Compute aggregate metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            "MMStar-Accuracy": accuracy,
            "MMStar-Correct": correct_count,
            "MMStar-Total": total_count
        }

