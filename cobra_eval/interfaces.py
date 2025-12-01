"""
Abstract Base Classes for Generators and Metrics.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from PIL import Image
from pydantic import BaseModel

class GenerationResult(BaseModel):
    caption: str
    reasoning_trace: Optional[str] = None
    metadata: Dict[str, Any] = {}

class MetricResult(BaseModel):
    score: float
    details: Dict[str, Any] = {}

class BaseGenerator(ABC):
    """
    Abstract base class for all caption generators.
    """
    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate a caption for the given image.
        """
        pass

class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    """
    @abstractmethod
    def compute(self, references: Dict[int, List[str]], predictions: Dict[int, List[str]]) -> Dict[str, float]:
        """
        Compute metric score.
        
        Args:
            references: Dict mapping image_id to list of reference captions
            predictions: Dict mapping image_id to list of generated captions (usually single element list)
            
        Returns:
            Dictionary mapping metric names to scores
        """
        pass

