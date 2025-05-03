"""
Custom GPT models

Snippets taken from
https://github.com/karpathy/nanoGPT/blob/master/model.py"""

from .model import (  # noqa: F401
    MITransformerConfig, MITransformerLM, MITransformer,
    TransformerDescription, description_builder)  # type: ignore
