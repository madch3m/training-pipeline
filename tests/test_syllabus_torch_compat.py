import pytest

torch = pytest.importorskip("torch")

import syllabus_torch_compat as tc


def test_apply_torch_ao_compat_patches_idempotent():
    tc.apply_torch_ao_compat_patches()
    assert hasattr(torch, "torchao_version")
    assert isinstance(torch.torchao_version, str)
    ver = getattr(torch, "version", None)
    if ver is not None:
        assert hasattr(ver, "torchao_version")
    tc.apply_torch_ao_compat_patches()
