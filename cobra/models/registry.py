"""
registry.py

Exhaustive list of pretrained VLMs (with full descriptions / links to corresponding names and sections of paper).
"""


# === Pretrained Model Registry ===
# fmt: off
MODEL_REGISTRY = {
    "cobra+3b": {
        "model_id": "cobra+3b",
        "names": ["Cobra-DINOSigLIP 3B"],
        "description": {
            "name": "Cobra 3B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Mamba 2.8B Zephyr",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
}

# Build Global Registry (Model ID, Name) -> Metadata
GLOBAL_REGISTRY = {name: v for k, v in MODEL_REGISTRY.items() for name in [k] + v["names"]}

# fmt: on
