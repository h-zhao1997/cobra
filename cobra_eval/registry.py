"""
Registry for managing plugins (Generators and Metrics).
"""
from typing import Dict, Type, Any

class Registry:
    _generators: Dict[str, Type[Any]] = {}
    _metrics: Dict[str, Type[Any]] = {}

    @classmethod
    def register_generator(cls, name: str):
        def decorator(generator_cls):
            cls._generators[name] = generator_cls
            return generator_cls
        return decorator

    @classmethod
    def register_metric(cls, name: str):
        def decorator(metric_cls):
            cls._metrics[name] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def get_generator(cls, name: str) -> Type[Any]:
        if name not in cls._generators:
            raise ValueError(f"Generator '{name}' not found in registry. Available: {list(cls._generators.keys())}")
        return cls._generators[name]

    @classmethod
    def get_metric(cls, name: str) -> Type[Any]:
        if name not in cls._metrics:
            raise ValueError(f"Metric '{name}' not found in registry. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name]

