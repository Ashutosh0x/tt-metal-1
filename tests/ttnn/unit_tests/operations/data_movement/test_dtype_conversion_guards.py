"""Verify guards from issue #58372 reject unsupported dtype pairs in datacopy ops."""

import pytest
import torch
import ttnn


DT = {
    "float32": ttnn.float32,
    "bfloat16": ttnn.bfloat16,
    "int32": ttnn.int32,
    "uint32": ttnn.uint32,
    "uint16": ttnn.uint16,
    "uint8": ttnn.uint8,
}

TORCH_DT = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "uint32": torch.int64,
    "uint16": torch.int32,
    "uint8": torch.uint8,
}

BAD_PAIRS = [
    ("float32", "int32"),
    ("float32", "uint32"),
    ("float32", "uint16"),
    ("bfloat16", "int32"),
    ("bfloat16", "uint32"),
    ("bfloat16", "uint16"),
    ("int32", "float32"),
    ("int32", "bfloat16"),
    ("uint32", "float32"),
    ("uint16", "float32"),
    ("uint16", "bfloat16"),
    ("int32", "uint16"),
    ("uint32", "uint16"),
    ("uint16", "int32"),
    ("uint16", "uint32"),
]

SAFE_PAIRS = [
    ("float32", "bfloat16"),
    ("bfloat16", "float32"),
    ("int32", "uint32"),
    ("uint32", "int32"),
    ("uint8", "int32"),
    ("uint8", "uint32"),
]


def _mk(device, shape, dt):
    x = torch.randint(0, 2, shape).to(TORCH_DT[dt])
    return ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_clone_rejects(device, expect_error, src, dst):
    t = _mk(device, (1, 1, 32, 32), src)
    with expect_error(RuntimeError, "typecast"):
        ttnn.clone(t, dtype=DT[dst])


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_copy_rejects(device, expect_error, src, dst):
    t = _mk(device, (1, 1, 32, 32), src)
    out = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32).to(TORCH_DT[dst]), dtype=DT[dst], layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "typecast"):
        ttnn.copy(t, out)


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_tilize_rejects(device, expect_error, src, dst):
    x = torch.zeros(1, 1, 32, 32).to(TORCH_DT[src])
    t = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "typecast"):
        ttnn.tilize(t, dtype=DT[dst])


@pytest.mark.parametrize("src,dst", SAFE_PAIRS)
def test_clone_safe_pairs_still_work(device, src, dst):
    t = _mk(device, (1, 1, 32, 32), src)
    ttnn.clone(t, dtype=DT[dst])


@pytest.mark.parametrize("dst", ["int32", "uint32", "uint16"])
def test_matmul_rejects_integer_output(device, expect_error, dst):
    a = torch.randint(0, 2, (1, 1, 32, 64)).float()
    w = torch.randint(0, 2, (1, 1, 64, 32)).float()
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "typecast"):
        ttnn.matmul(ta, tw, dtype=DT[dst])


def test_matmul_float_still_works(device):
    a = torch.randint(0, 2, (1, 1, 32, 64)).float()
    w = torch.randint(0, 2, (1, 1, 64, 32)).float()
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(ttnn.matmul(ta, tw, dtype=ttnn.float32)).double()
    ref = (a @ w).double()
    assert torch.equal(y, ref)
