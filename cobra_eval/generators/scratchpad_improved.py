"""
Improved two-pass (or multi-pass) scratchpad reasoning generator with accuracy-focused refinements.
"""
from PIL import Image
from typing import Any, List, Optional
import re
from ..interfaces import BaseGenerator, GenerationResult
from ..registry import Registry
from .utils import clean_reasoning_trace

@Registry.register_generator("scratchpad_improved")
class ImprovedScratchpadGenerator(BaseGenerator):
    def __init__(self, vlm: Any, scratchpad_passes: int = 1):
        """
        Initialize with VLM and number of reasoning passes.
        
        Improvements:
        - Better refinement prompts focused on accuracy
        - Quality filtering of reasoning traces
        - Selective trace usage (only best/latest)
        - Length constraints to prevent verbosity
        """
        self.vlm = vlm
        self.scratchpad_passes = scratchpad_passes
        
        # Improved reasoning prompt - more focused
        self.reasoning_prompt = (
            "Analyze this image systematically. Identify all major objects, their colors, "
            "sizes, and positions. Describe spatial relationships between objects. "
            "Be factual and concise. Do NOT write a caption - only provide analytical observations."
        )
        
        # Quality thresholds
        self.max_trace_length = 200  # words
        self.min_trace_length = 10   # words

    def _is_low_quality_trace(self, trace: str) -> bool:
        """
        Detect low-quality reasoning traces that should be filtered out.
        
        Returns True if trace contains:
        - Emojis
        - Hashtags
        - Excessive meta-commentary
        - Off-topic rambling
        """
        if not trace:
            return True
            
        # Check for emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        if emoji_pattern.search(trace):
            return True
        
        # Check for hashtags
        if '#' in trace and any(word.startswith('#') for word in trace.split()):
            return True
        
        # Check for excessive meta-commentary
        meta_phrases = [
            "what do you think",
            "let me know",
            "thanks!",
            "keep going",
            "good job",
            "good luck",
            "appreciate your",
            "thank you",
            "please",
            "thoughts please"
        ]
        trace_lower = trace.lower()
        meta_count = sum(1 for phrase in meta_phrases if phrase in trace_lower)
        if meta_count >= 2:  # Too much meta-commentary
            return True
        
        # Check for excessive length (likely rambling)
        word_count = len(trace.split())
        if word_count > self.max_trace_length:
            return True
        
        return False

    def _refine_trace(self, trace: str) -> str:
        """
        Clean and refine a reasoning trace.
        
        - Remove low-quality content
        - Limit length
        - Remove repetitive patterns
        """
        if not trace:
            return ""
        
        # Remove emojis and hashtags
        trace = re.sub(r'[#@]\w+', '', trace)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        trace = emoji_pattern.sub('', trace)
        
        # Limit length
        words = trace.split()
        if len(words) > self.max_trace_length:
            trace = ' '.join(words[:self.max_trace_length])
        
        # Remove excessive punctuation
        trace = re.sub(r'[!]{2,}', '!', trace)
        trace = re.sub(r'[?]{2,}', '?', trace)
        trace = re.sub(r'[.]{3,}', '...', trace)
        
        return trace.strip()

    def _get_refinement_prompt(self, previous_trace: str, pass_num: int) -> str:
        """
        Generate an improved refinement prompt focused on accuracy, not expansion.
        
        Improvements:
        - Focus on correcting errors
        - Encourage conciseness
        - Maintain factual accuracy
        - Avoid meta-commentary
        """
        if pass_num == 1:
            # First refinement: focus on accuracy
            return (
                f"Review your previous analysis:\n{previous_trace}\n\n"
                f"Identify and correct any errors or inaccuracies. Remove any unnecessary details. "
                f"Be concise and factual. Focus only on what you can clearly see in the image."
            )
        else:
            # Subsequent refinements: focus on precision
            return (
                f"Review your previous analysis:\n{previous_trace}\n\n"
                f"Refine this analysis by: (1) Correcting any factual errors, "
                f"(2) Removing redundant or unnecessary information, "
                f"(3) Ensuring all statements are accurate and verifiable from the image. "
                f"Be concise. Do not add new details unless they are clearly visible."
            )

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate improved reasoning with quality filtering and accuracy-focused refinement.
        """
        reasoning_max_tokens = kwargs.get('reasoning_max_tokens', 256)
        repetition_penalty = kwargs.get('repetition_penalty', 1.2)
        
        # Clean kwargs for VLM calls
        gen_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['reasoning_max_tokens', 'repetition_penalty']}
        
        reasoning_kwargs = gen_kwargs.copy()
        reasoning_kwargs['max_new_tokens'] = reasoning_max_tokens
        reasoning_kwargs['repetition_penalty'] = repetition_penalty

        traces: List[str] = []
        quality_traces: List[str] = []  # Only high-quality traces
        
        # Multi-pass reasoning with quality filtering
        for i in range(self.scratchpad_passes):
            prompt_builder = self.vlm.get_prompt_builder()
            
            if i == 0:
                msg = self.reasoning_prompt
            else:
                # Use the last high-quality trace for refinement
                previous_trace = quality_traces[-1] if quality_traces else traces[-1]
                msg = self._get_refinement_prompt(previous_trace, i)
                
            prompt_builder.add_turn(role="human", message=msg)
            prompt_text = prompt_builder.get_prompt()
            
            trace = self.vlm.generate(
                image,
                prompt_text,
                use_cache=True,
                **reasoning_kwargs
            )
            
            # Clean and refine trace
            cleaned_trace = clean_reasoning_trace(trace)
            refined_trace = self._refine_trace(cleaned_trace)
            
            traces.append(refined_trace)
            
            # Quality filtering: only keep high-quality traces
            if not self._is_low_quality_trace(refined_trace):
                if len(refined_trace.split()) >= self.min_trace_length:
                    quality_traces.append(refined_trace)
            
            # Stop early if we got a good trace and it's not improving
            if len(quality_traces) >= 2:
                # Check if latest trace is significantly different
                if refined_trace and refined_trace in quality_traces[:-1]:
                    # Repetitive, stop
                    break

        # Use only high-quality traces, or fall back to all traces if none are high-quality
        if quality_traces:
            # Use the latest high-quality trace (most refined)
            final_reasoning = quality_traces[-1]
        elif traces:
            # Fallback: use latest trace even if not perfect
            final_reasoning = traces[-1]
        else:
            final_reasoning = ""

        # Second pass: Generate final caption
        prompt_builder_caption = self.vlm.get_prompt_builder()
        
        if not final_reasoning or len(final_reasoning.split()) < 5:
            combined_prompt = prompt
        else:
            # Improved caption prompt - emphasizes conciseness
            combined_prompt = (
                f"{prompt}\n\n"
                f"Based on this analysis:\n{final_reasoning}\n\n"
                f"Write a concise, factual caption (10-15 words) that accurately describes the image. "
                f"Match the style of typical image captions: brief and factual."
            )
            
        prompt_builder_caption.add_turn(role="human", message=combined_prompt)
        caption_prompt_text = prompt_builder_caption.get_prompt()
        
        final_caption = self.vlm.generate(
            image,
            caption_prompt_text,
            use_cache=True,
            **gen_kwargs
        )
        
        # Clean up final caption (remove any artifacts)
        final_caption = final_caption.strip()
        
        return GenerationResult(
            caption=final_caption,
            reasoning_trace=final_reasoning,
            metadata={
                "method": "scratchpad_improved",
                "passes": self.scratchpad_passes,
                "actual_passes": len(traces),
                "quality_traces_count": len(quality_traces),
                "individual_traces": traces,
                "quality_traces": quality_traces
            }
        )

