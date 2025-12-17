"""
BLEU Metric implementation.
"""
from typing import Dict, List
from pycocoevalcap.bleu.bleu import Bleu
from ..interfaces import BaseMetric, MetricResult
from ..registry import Registry
import io
from contextlib import redirect_stdout

@Registry.register_metric("bleu")
class BLEUMetric(BaseMetric):
    def __init__(self, n_gram: int = 4):
        self.n_gram = n_gram
        self.scorer = Bleu(n_gram)

    def compute(self, references: Dict[int, List[str]], predictions: Dict[int, List[str]]) -> Dict[str, float]:
        # Capture stdout to suppress verbose output from pycocoevalcap
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            score, _ = self.scorer.compute_score(references, predictions)
        
        # score is a list of scores for BLEU-1, BLEU-2, etc.
        results = {}
        if isinstance(score, list):
            for i, s in enumerate(score):
                results[f"BLEU-{i+1}"] = s
        else:
            # Handle case where single score might be returned (unlikely with default Bleu)
            results["BLEU"] = score
            
        return results

