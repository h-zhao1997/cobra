"""
External model generators for GPT-5, Gemini, Claude, and Llama.
"""
import base64
import io
import os
from pathlib import Path
from typing import Any, Optional, Dict
from PIL import Image

from ..interfaces import BaseGenerator, GenerationResult
from ..registry import Registry


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def get_api_key(env_var: str, file_path: Optional[str] = None) -> Optional[str]:
    """Get API key from environment variable or file."""
    # Try environment variable first
    key = os.getenv(env_var)
    if key:
        return key.strip()
    
    # Try file path if provided
    if file_path:
        key_file = Path(file_path)
        if key_file.exists():
            return key_file.read_text().strip()
    
    return None


@Registry.register_generator("gpt5")
class GPT5Generator(BaseGenerator):
    """OpenAI GPT-5 (or GPT-4o as fallback) generator."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", api_key_file: Optional[str] = None):
        """
        Initialize GPT-5/4o generator.
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model name (default: "gpt-4o", can use "gpt-4o-mini" or "gpt-5" when available)
            api_key_file: Path to file containing API key
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key or get_api_key("OPENAI_API_KEY", api_key_file))
        self.model = model
    
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """Generate caption using OpenAI vision model."""
        import openai
        
        # Convert image to base64
        img_base64 = image_to_base64(image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.4),
            )
            
            caption = response.choices[0].message.content
            
            return GenerationResult(
                caption=caption,
                metadata={
                    "method": "gpt5",
                    "model": self.model,
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else {}
                }
            )
        except Exception as e:
            return GenerationResult(
                caption=f"Error: {str(e)}",
                metadata={"method": "gpt5", "error": str(e)}
            )


@Registry.register_generator("gemini")
class GeminiGenerator(BaseGenerator):
    """Google Gemini generator."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro", api_key_file: Optional[str] = None):
        """
        Initialize Gemini generator.
        
        Args:
            api_key: Google API key (or use GEMINI_API_KEY env var)
            model: Model name (default: "gemini-1.5-pro", can use "gemini-1.5-flash")
            api_key_file: Path to file containing API key
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        
        api_key = api_key or get_api_key("GEMINI_API_KEY", api_key_file)
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or provide api_key parameter.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """Generate caption using Gemini."""
        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config={
                    "max_output_tokens": kwargs.get("max_new_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.4),
                }
            )
            
            caption = response.text if hasattr(response, 'text') else str(response)
            
            return GenerationResult(
                caption=caption,
                metadata={
                    "method": "gemini",
                    "model": self.model.model_name if hasattr(self.model, 'model_name') else "gemini",
                }
            )
        except Exception as e:
            return GenerationResult(
                caption=f"Error: {str(e)}",
                metadata={"method": "gemini", "error": str(e)}
            )


@Registry.register_generator("claude")
class ClaudeGenerator(BaseGenerator):
    """Anthropic Claude generator."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", api_key_file: Optional[str] = None):
        """
        Initialize Claude generator.
        
        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            model: Model name (default: "claude-3-5-sonnet-20241022", can use "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307")
            api_key_file: Path to file containing API key
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        api_key = api_key or get_api_key("ANTHROPIC_API_KEY", api_key_file)
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or provide api_key parameter.")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """Generate caption using Claude."""
        from anthropic import Anthropic
        
        # Convert image to base64
        img_base64 = image_to_base64(image)
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.4),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            caption = message.content[0].text if message.content else str(message)
            
            return GenerationResult(
                caption=caption,
                metadata={
                    "method": "claude",
                    "model": self.model,
                    "usage": message.usage.model_dump() if hasattr(message, 'usage') else {}
                }
            )
        except Exception as e:
            return GenerationResult(
                caption=f"Error: {str(e)}",
                metadata={"method": "claude", "error": str(e)}
            )


@Registry.register_generator("llama")
class LlamaGenerator(BaseGenerator):
    """Meta Llama generator (via HuggingFace or local model)."""
    
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", hf_token: Optional[str] = None, device: str = "cuda"):
        """
        Initialize Llama generator.
        
        Args:
            model_id: HuggingFace model ID or local path
            hf_token: HuggingFace token (or use HF_TOKEN env var or .hf_token file)
            device: Device to run on ("cuda" or "cpu")
        """
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
        except ImportError:
            raise ImportError("transformers and torch packages required. Install with: pip install transformers torch")
        
        # Get HF token
        if not hf_token:
            hf_token = get_api_key("HF_TOKEN", ".hf_token")
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model_id = model_id
    
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        """Generate caption using Llama."""
        import torch
        
        try:
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.4),
                do_sample=kwargs.get("temperature", 0.4) > 0,
            )
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return GenerationResult(
                caption=generated_text,
                metadata={
                    "method": "llama",
                    "model": self.model_id,
                }
            )
        except Exception as e:
            return GenerationResult(
                caption=f"Error: {str(e)}",
                metadata={"method": "llama", "error": str(e)}
            )

