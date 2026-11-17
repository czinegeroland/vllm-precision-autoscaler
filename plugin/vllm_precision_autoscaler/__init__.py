"""vllm-precision-autoscaler: change the precision of a model's linear layers
(16-bit, FP8, INT4) while vLLM keeps serving."""

__version__ = "0.6.0"


def register() -> None:
    """vLLM general-plugin entry point. Safe to call more than once."""
    from . import config, ops  # noqa: F401  (import registers config and op)
