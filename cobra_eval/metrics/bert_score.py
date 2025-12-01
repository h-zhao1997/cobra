"""
BERTScore Metric implementation.
"""
from typing import Dict, List
from bert_score import score
from ..interfaces import BaseMetric, MetricResult
from ..registry import Registry
import torch

@Registry.register_metric("bert_score")
class BERTScoreMetric(BaseMetric):
    def __init__(self, model_type: str = "bert-base-uncased", lang: str = "en"):
        self.model_type = model_type
        self.lang = lang

    def compute(self, references: Dict[int, List[str]], predictions: Dict[int, List[str]]) -> Dict[str, float]:
        sorted_ids = sorted(predictions.keys())
        
        # Extract predictions and references in matching order
        # predictions[id] is a list, usually single caption
        preds = [predictions[idx][0] for idx in sorted_ids]
        
        # references[id] is a list of valid references. 
        # BERTScore expects list of refs for each pred.
        refs = [references[idx] for idx in sorted_ids]
        
        # Compute score
        # If CUDA is available, bert_score usually auto-detects, but explicit device is good.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        P, R, F1 = score(preds, refs, model_type=self.model_type, lang=self.lang, device=device, verbose=False)
        
        # Store per-sample scores
        self.per_sample_scores = {}
        for idx, img_id in enumerate(sorted_ids):
            self.per_sample_scores[img_id] = {
                "BERTScore-Precision": P[idx].item(),
                "BERTScore-Recall": R[idx].item(),
                "BERTScore-F1": F1[idx].item()
            }
        
        return {
            "BERTScore-Precision": P.mean().item(),
            "BERTScore-Recall": R.mean().item(),
            "BERTScore-F1": F1.mean().item()
        }

