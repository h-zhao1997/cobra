"""
Scratchpad generator with multipass comparison mode.
Runs multiple pass configurations (1, 2, 3, 4) and returns results for all.
"""
from PIL import Image
from typing import Any, List, Dict
from ..interfaces import BaseGenerator, GenerationResult
from ..registry import Registry
from .scratchpad import ScratchpadGenerator

@Registry.register_generator("scratchpad_compare")
class ScratchpadCompareGenerator(BaseGenerator):
    def __init__(self, vlm: Any, scratchpad_passes: int = 4, compare_passes: List[int] = None):
        """
        Initialize with VLM and comparison configuration.
        
        Args:
            vlm: The Cobra VLM model
            scratchpad_passes: Maximum number of passes (for backward compatibility)
            compare_passes: List of pass counts to compare (e.g., [1, 2, 3, 4])
                           If None, uses [1, 2, 3, 4] up to scratchpad_passes
        """
        self.vlm = vlm
        self.scratchpad_passes = scratchpad_passes
        
        # Determine which pass counts to compare
        if compare_passes is None:
            self.compare_passes = list(range(1, min(scratchpad_passes + 1, 5)))  # Default: 1, 2, 3, 4
        else:
            self.compare_passes = sorted([p for p in compare_passes if p > 0])
        
        # Create individual scratchpad generators for each pass count
        self.generators: Dict[int, ScratchpadGenerator] = {}
        for passes in self.compare_passes:
            self.generators[passes] = ScratchpadGenerator(vlm, scratchpad_passes=passes)

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate captions with multiple pass configurations and return comparison.
        
        Returns a GenerationResult where:
        - caption: The caption from the best-performing pass (highest BLEU estimate)
        - reasoning_trace: The reasoning from the best pass
        - metadata: Contains all pass results for comparison
        """
        all_results: Dict[int, GenerationResult] = {}
        
        # Generate with each pass configuration
        for passes in self.compare_passes:
            generator = self.generators[passes]
            result = generator.generate(image, prompt, **kwargs)
            all_results[passes] = result
        
        # Determine "best" result (for main caption/reasoning)
        # Use the result with most passes (most refined) as default
        # In practice, you'd want to use actual metrics, but we don't have refs here
        best_passes = max(self.compare_passes)
        best_result = all_results[best_passes]
        
        # Build comparison metadata
        comparison_data = {}
        for passes, result in all_results.items():
            comparison_data[passes] = {
                "caption": result.caption,
                "reasoning_trace": result.reasoning_trace,
                "caption_length": len(result.caption.split()),
                "reasoning_length": len(result.reasoning_trace.split()) if result.reasoning_trace else 0,
                "metadata": result.metadata
            }
        
        return GenerationResult(
            caption=best_result.caption,
            reasoning_trace=best_result.reasoning_trace,
            metadata={
                "method": "scratchpad_compare",
                "compare_passes": self.compare_passes,
                "comparison": comparison_data,
                "best_pass": best_passes,
                "all_results": {
                    passes: {
                        "caption": r.caption,
                        "reasoning": r.reasoning_trace
                    }
                    for passes, r in all_results.items()
                }
            }
        )

