"""Colab / mixed-install compatibility for TorchAO-related PyTorch attributes.

Some dependency stacks expect ``torch.torchao_version`` (and occasionally
``torch.version.torchao_version``) to exist. Stock PyTorch wheels on Colab may
not define them, which surfaces as ``AttributeError`` during ``import torch`` /
``import trl`` / training startup.

This module provides a no-op–style default so imports can proceed. Prefer
matching **torch** and **torchao** versions from the official matrices when you
use TorchAO quantization; this shim is only for the syllabus SFT path.
"""

from __future__ import annotations

import os


def apply_torch_ao_compat_patches() -> None:
    """Set env defaults and patch missing attributes on ``torch`` if needed."""
    os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")

    try:
        import torch
    except ImportError:
        return

    sentinel = str(getattr(torch, "__version__", "0.0.0"))
    if not hasattr(torch, "torchao_version"):
        setattr(torch, "torchao_version", sentinel)

    version_module = getattr(torch, "version", None)
    if version_module is not None and not hasattr(version_module, "torchao_version"):
        setattr(version_module, "torchao_version", sentinel)
