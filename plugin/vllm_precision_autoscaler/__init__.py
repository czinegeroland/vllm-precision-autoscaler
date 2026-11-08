"""vllm-precision-autoscaler: switch linear layers between 16-bit and FP8
per engine step, based on how busy the step is."""

__version__ = "0.3.0"


def register() -> None:
    """vLLM general-plugin entry point. Safe to call more than once."""
    from . import config, ops  # noqa: F401  (import registers config and op)
