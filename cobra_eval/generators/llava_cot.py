"""
LLaVA-CoT structured reasoning generator with multiple scratchpad passes.

Implements the LLaVA-CoT method with structured reasoning stages:
- SUMMARY: Problem understanding and approach
- CAPTION: Visual interpretation from the image
- REASONING: Step-by-step logical reasoning
- CONCLUSION: Final answer/conclusion

Supports multiple passes for iterative refinement.
"""
from PIL import Image
from typing import Any, List, Optional
import re
from ..interfaces import BaseGenerator, GenerationResult
from ..registry import Registry
from .utils import clean_reasoning_trace

def extract_clean_caption(text: str, max_words: int = 50) -> str:
    """
    Extract a clean caption from potentially verbose/rambling text.
    Tries to find the first meaningful sentence or phrase.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Try to find the first complete sentence
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        first_sentence = sentences[0].strip()
        words = first_sentence.split()
        
        # If first sentence is reasonable length, use it
        if 5 <= len(words) <= max_words:
            return first_sentence
        
        # If too long, truncate intelligently
        if len(words) > max_words:
            # Try to find a natural break point
            truncated = ' '.join(words[:max_words])
            # Remove trailing incomplete words/phrases
            truncated = re.sub(r'\s+\w+\s*$', '', truncated)
            return truncated + "..."
        
        # If too short, try to combine with next sentence
        if len(words) < 5 and len(sentences) > 1:
            combined = first_sentence + ". " + sentences[1]
            words = combined.split()
            if len(words) <= max_words:
                return combined
            else:
                return ' '.join(words[:max_words]) + "..."
    
    # Fallback: use first N words
    words = text.split()
    if len(words) <= max_words:
        return text
    else:
        return ' '.join(words[:max_words]) + "..."

@Registry.register_generator("llava_cot")
class LLaVACoTGenerator(BaseGenerator):
    def __init__(self, vlm: Any, scratchpad_passes: int = 1):
        """
        Initialize with VLM and number of reasoning passes.
        
        Args:
            vlm: The Cobra VLM model
            scratchpad_passes: Number of reasoning passes (default: 1)
        """
        self.vlm = vlm
        self.scratchpad_passes = scratchpad_passes
        
        # LLaVA-CoT structured prompt template
        # Simplified and more explicit format
        self.structured_prompt = (
            "You must respond using this EXACT format with XML-style tags:\n\n"
            "<SUMMARY>\n"
            "Brief summary of what you see in the image.\n"
            "</SUMMARY>\n\n"
            "<CAPTION>\n"
            "A concise caption describing the image.\n"
            "</CAPTION>\n\n"
            "<REASONING>\n"
            "Step-by-step analysis of the image.\n"
            "</REASONING>\n\n"
            "<CONCLUSION>\n"
            "Final concise answer or caption.\n"
            "</CONCLUSION>\n\n"
            "IMPORTANT: You must include ALL four tags: <SUMMARY>, <CAPTION>, <REASONING>, and <CONCLUSION>."
        )

    def _extract_stage(self, text: str, stage_name: str) -> Optional[str]:
        """
        Extract content from a specific structured stage.
        
        Args:
            text: Full text containing structured stages
            stage_name: Name of the stage (SUMMARY, CAPTION, REASONING, CONCLUSION)
            
        Returns:
            Extracted stage content or None if not found
        """
        pattern = f"<{stage_name}>(.*?)</{stage_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_all_stages(self, text: str) -> dict:
        """
        Extract all structured stages from the response.
        
        Returns:
            Dictionary with keys: summary, caption, reasoning, conclusion
        """
        return {
            "summary": self._extract_stage(text, "SUMMARY"),
            "caption": self._extract_stage(text, "CAPTION"),
            "reasoning": self._extract_stage(text, "REASONING"),
            "conclusion": self._extract_stage(text, "CONCLUSION")
        }

    def _has_structured_format(self, text: str) -> bool:
        """Check if text contains structured LLaVA-CoT format."""
        # Check for at least one structured tag
        has_tags = bool(re.search(r"<(SUMMARY|CAPTION|REASONING|CONCLUSION)>", text, re.IGNORECASE))
        return has_tags

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate structured reasoning with multiple passes.
        
        Args:
            image: PIL Image to analyze
            prompt: User prompt/question
            **kwargs: Generation parameters
            
        Returns:
            GenerationResult with structured reasoning
        """
        reasoning_max_tokens = kwargs.get('reasoning_max_tokens', 512)
        repetition_penalty = kwargs.get('repetition_penalty', 1.2)
        
        # Clean kwargs for VLM calls
        gen_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['reasoning_max_tokens', 'repetition_penalty']}
        
        reasoning_kwargs = gen_kwargs.copy()
        reasoning_kwargs['max_new_tokens'] = reasoning_max_tokens
        reasoning_kwargs['repetition_penalty'] = repetition_penalty

        all_passes: List[dict] = []
        previous_stages: Optional[dict] = None
        
        # Multi-pass structured reasoning
        for pass_num in range(self.scratchpad_passes):
            prompt_builder = self.vlm.get_prompt_builder()
            
            if pass_num == 0:
                # First pass: Use structured prompt
                combined_prompt = f"{prompt}\n\n{self.structured_prompt}"
            else:
                # Subsequent passes: Refine based on previous output
                if previous_stages:
                    refinement_prompt = (
                        f"{prompt}\n\n"
                        f"Review and refine your previous analysis:\n\n"
                        f"Previous SUMMARY:\n{previous_stages.get('summary', 'N/A')}\n\n"
                        f"Previous CAPTION:\n{previous_stages.get('caption', 'N/A')}\n\n"
                        f"Previous REASONING:\n{previous_stages.get('reasoning', 'N/A')}\n\n"
                        f"Previous CONCLUSION:\n{previous_stages.get('conclusion', 'N/A')}\n\n"
                        f"Now provide an improved analysis using the same structured format:\n\n"
                        f"{self.structured_prompt}"
                    )
                    combined_prompt = refinement_prompt
                else:
                    combined_prompt = f"{prompt}\n\n{self.structured_prompt}"
            
            prompt_builder.add_turn(role="human", message=combined_prompt)
            prompt_text = prompt_builder.get_prompt()
            
            # Generate structured response
            response = self.vlm.generate(
                image,
                prompt_text,
                use_cache=True,
                **reasoning_kwargs
            )
            
            # Extract stages
            stages = self._extract_all_stages(response)
            
            # If structured format not found, try to parse as-is
            if not self._has_structured_format(response):
                # Try to extract a reasonable caption from the response
                cleaned_response = response.strip()
                
                # Clean up the reasoning - remove repetitive content
                cleaned_reasoning = clean_reasoning_trace(cleaned_response)
                
                # Extract a clean caption
                potential_caption = extract_clean_caption(cleaned_response, max_words=50)
                
                stages = {
                    "summary": None,
                    "caption": potential_caption,
                    "reasoning": cleaned_reasoning if cleaned_reasoning else cleaned_response[:1000],  # Limit reasoning length
                    "conclusion": potential_caption  # Use same caption as conclusion
                }
            
            all_passes.append({
                "pass": pass_num + 1,
                "raw_response": response,
                "stages": stages
            })
            
            previous_stages = stages
            
            # Stop early if we got a valid conclusion
            if stages.get("conclusion") and len(stages["conclusion"].split()) >= 3:
                break

        # Use the last (most refined) pass
        final_pass = all_passes[-1]
        final_stages = final_pass["stages"]
        
        # Extract final caption (prioritize CONCLUSION, then CAPTION, then extract from reasoning)
        # Clean and extract intelligently to avoid rambling text
        if final_stages.get("conclusion"):
            final_caption = extract_clean_caption(final_stages["conclusion"], max_words=50)
        elif final_stages.get("caption"):
            final_caption = extract_clean_caption(final_stages["caption"], max_words=50)
        elif final_stages.get("reasoning"):
            # Extract a reasonable caption from reasoning
            final_caption = extract_clean_caption(final_stages["reasoning"], max_words=50)
        else:
            # Last resort: extract from raw response
            final_caption = extract_clean_caption(final_pass["raw_response"], max_words=50)
        
        # Build full reasoning trace with all stages
        reasoning_parts = []
        if final_stages.get("summary"):
            reasoning_parts.append(f"SUMMARY: {final_stages['summary']}")
        if final_stages.get("caption"):
            reasoning_parts.append(f"CAPTION: {final_stages['caption']}")
        if final_stages.get("reasoning"):
            reasoning_parts.append(f"REASONING: {final_stages['reasoning']}")
        
        full_reasoning_trace = "\n\n".join(reasoning_parts) if reasoning_parts else final_pass["raw_response"]
        
        return GenerationResult(
            caption=final_caption.strip(),
            reasoning_trace=full_reasoning_trace,
            metadata={
                "method": "llava_cot",
                "passes": self.scratchpad_passes,
                "actual_passes": len(all_passes),
                "stages": final_stages,
                "all_passes": [
                    {
                        "pass": p["pass"],
                        "raw_response": p["raw_response"],  # Save full raw response
                        "stages": p["stages"]
                    } for p in all_passes
                ]
            }
        )

