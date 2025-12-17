"""
Utilities for generation (cleaning, repetition detection).
"""

def detect_repetition(text: str, min_repeat_length: int = 20) -> int:
    """
    Detect repetitive loops in text and return the position to truncate.
    """
    words = text.split()
    if len(words) < min_repeat_length * 2:
        return -1
    
    # Check for repeating patterns at the end
    for i in range(len(words) - min_repeat_length, max(0, len(words) - 200), -1):
        pattern = words[i:i + min_repeat_length]
        if len(pattern) < min_repeat_length:
            continue
        
        # Check if this pattern repeats
        pattern_str = ' '.join(pattern)
        remaining = ' '.join(words[i + min_repeat_length:])
        
        if pattern_str in remaining:
            # Found repetition, truncate before it
            return i
    
    return -1

def clean_reasoning_trace(reasoning: str) -> str:
    """
    Clean up reasoning trace by removing repetitive loops and generic phrases.
    """
    # Remove generic unhelpful phrases
    generic_phrases = [
        "Step 1: Identify objects.",
        "Step-by-step analysis:",
        "Step 1:",
    ]
    
    cleaned = reasoning.strip()
    
    # Remove generic phrases at the start
    for phrase in generic_phrases:
        if cleaned.startswith(phrase):
            cleaned = cleaned[len(phrase):].strip()
    
    # Detect and truncate repetitive loops
    truncate_pos = detect_repetition(cleaned)
    if truncate_pos > 0:
        words = cleaned.split()
        cleaned = ' '.join(words[:truncate_pos])
    
    # If reasoning is too short or generic, return empty
    if len(cleaned.split()) < 5:
        return ""
    
    return cleaned.strip()

