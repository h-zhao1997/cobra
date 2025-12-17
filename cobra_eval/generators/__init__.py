# Import all generators to ensure registration
from . import baseline
from . import scratchpad
from . import scratchpad_improved
from . import scratchpad_compare
from . import external
from . import llava_cot
from . import external_models

__all__ = ["baseline", "scratchpad", "scratchpad_improved", "scratchpad_compare", "external", "llava_cot", "external_models"]
