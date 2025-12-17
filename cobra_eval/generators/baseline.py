"""
Standard single-pass inference generator.
"""
from PIL import Image
from typing import Any
from ..interfaces import BaseGenerator, GenerationResult
from ..registry import Registry

@Registry.register_generator("baseline")
class BaselineGenerator(BaseGenerator):
    def __init__(self, vlm: Any):
        """
        Initialize with a Cobra VLM model instance.
        """
        self.vlm = vlm

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        # Filter out scratchpad-specific parameters if present
        generation_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['reasoning_max_tokens', 'repetition_penalty']}
        
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        
        generated_text = self.vlm.generate(
            image,
            prompt_text,
            use_cache=True,
            **generation_kwargs
        )
        
        return GenerationResult(
            caption=generated_text,
            metadata={"method": "baseline"}
        )

