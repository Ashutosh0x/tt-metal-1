# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The L1 a pipeline leaves to its matmuls, the `l1_limit` of `models.tt_dit.utils.matmul_config_search`.

Each case builds a pipeline with its preset for the mesh and runs it. Before every single-chip
matmul of the first transformer forward it reads the lowest allocated L1 address; the forward then
stops the run. The allocator fills L1 top-down, so a matmul's static circular buffers must end at
or below that address. Pick cases with -k (the model and mesh, e.g. -k "klein and 2x4").

    pytest models/tt_dit/tests/test_l1_probe.py -k "klein and 2x4"

Results go to `generated/outputs/l1_probe/<case>.json` and to a `L1_PROBE` log line.
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.utils.test import (
    line_params_req_exact_devices,
    ring_params_8k_req_exact_devices,
    ring_params_req_exact_devices,
)

from .models.flux2.device_params import line_params_flux2_klein
from .models.qwenimage.device_params import EDIT_MESH_DEVICE_PARAMS

_MATMULS = ("minimal_matmul", "minimal_matmul_split", "dit_minimal_matmul_addcmul_fused")
_IMAGE = Path(__file__).resolve().parents[2] / "sample_data" / "demo.jpeg"
_QWEN_EDIT_PARAMS = {p.values[0]: p.values[1] for p in EDIT_MESH_DEVICE_PARAMS}
_IDEOGRAM = {"l1_small_size": 32768, "trace_region_size": 60000000}


class _Stop(Exception):  # noqa: N818
    pass


def _run_klein(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:
    from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        trace_warmup=False,
    )
    pipeline(prompts=["a watercolor"], images=[Image.open(_IMAGE)], num_inference_steps=4, seed=0)


def _run_qwen(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:  # noqa: ARG001
    from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name="Qwen/Qwen-Image-Edit-2511")
    pipeline(prompts=["a watercolor"], num_inference_steps=4, seed=0, image=Image.open(_IMAGE).convert("RGB").resize((1024, 1024)))


def _run_wan(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:  # noqa: ARG001
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V

    pipeline = WanPipelineI2V.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        height=480,
        width=832,
        run_warmup=False,
    )
    image = Image.open(_IMAGE).convert("RGB").resize((832, 480))
    pipeline(prompts=["a watercolor"], image_prompt=[ImagePrompt(image=image, frame_pos=0)], num_inference_steps=2)


def _run_ideogram(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:  # noqa: ARG001
    from models.tt_dit.pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline

    pipeline = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=1024, width=1024, run_warmup=False)
    pipeline(prompts=["a watercolor"], seed=0, traced=False)


_MODELS = {
    "klein": (_run_klein, "models.tt_dit.models.transformers.transformer_flux2", "Flux2Transformer", "forward"),
    "qwen": (_run_qwen, "models.tt_dit.models.transformers.transformer_qwenimage", "QwenImageTransformer", "forward"),
    "wan": (
        _run_wan,
        "models.tt_dit.models.transformers.wan2_2.transformer_wan",
        "WanTransformer3DModel",
        "inner_step cache_tail",  # cache_tail ends a forward under DBCache
    ),
    "ideogram": (
        _run_ideogram,
        "models.tt_dit.models.transformers.transformer_ideogram4",
        "Ideogram4Transformer",
        "forward",
    ),
}

_CASES = [
    ("klein", (1, 8), line_params_flux2_klein),
    ("klein", (2, 2), line_params_flux2_klein),
    ("klein", (2, 4), line_params_flux2_klein),
    ("klein", (4, 8), line_params_flux2_klein),
    *(("qwen", shape, _QWEN_EDIT_PARAMS[shape]) for shape in ((2, 2), (2, 4), (4, 8))),
    ("wan", (1, 4), line_params_req_exact_devices),
    ("wan", (2, 2), line_params_req_exact_devices),
    ("wan", (2, 4), line_params_req_exact_devices),
    ("wan", (4, 8), ring_params_req_exact_devices),
    ("wan", (4, 32), ring_params_8k_req_exact_devices),
    ("ideogram", (2, 4), {**line_params_req_exact_devices, **_IDEOGRAM}),
    ("ideogram", (4, 8), {**ring_params_req_exact_devices, **_IDEOGRAM}),
]


def _lowest_allocated(device: ttnn.MeshDevice) -> int | None:
    """The lowest allocated L1 address, relative to the allocator's base."""
    view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    addresses = [int(b["address"]) for b in view.block_table if b.get("allocated") == "yes"]
    return min(addresses) if addresses else None


def _allocator_base(device: ttnn.MeshDevice) -> int:
    """The absolute address of the allocator's relative 0: allocate one page per bank and compare."""
    view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([view.num_banks, 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    absolute = t.buffer_address()
    relative = _lowest_allocated(device)
    t.deallocate()
    assert relative is not None
    return absolute - relative


@pytest.mark.parametrize(
    ("model", "mesh_device", "device_params"),
    [pytest.param(m, shape, params, id=f"{m}-{shape[0]}x{shape[1]}") for m, shape, params in _CASES],
    indirect=["mesh_device", "device_params"],
)
def test_l1_probe(
    *,
    model: str,
    mesh_device: ttnn.MeshDevice,
    model_location_generator: Callable[[str], str],
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    import importlib

    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    run, module, cls, method = _MODELS[model]
    lowest: dict[tuple[int, int], int] = {}
    calls: dict[tuple[int, int], int] = defaultdict(int)

    def wrap_matmul(fn: Any) -> Any:  # noqa: ANN401
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            x = args[0] if args else kwargs.get("input_tensor")
            w = args[1] if len(args) > 1 else kwargs.get("weight_tensor")
            x = x[0] if isinstance(x, (list, tuple)) else x  # concatenated inputs
            assert x is not None and w is not None
            key = (int(w.padded_shape[-2]), int(w.padded_shape[-1]))
            calls[key] += 1
            address = _lowest_allocated(x.device())
            if address is not None:
                lowest[key] = min(lowest.get(key, address), address)
            return fn(*args, **kwargs)

        return wrapper

    def wrap_forward(fn: Any) -> Any:  # noqa: ANN401
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            fn(*args, **kwargs)
            raise _Stop

        return wrapper

    for name in _MATMULS:
        monkeypatch.setattr(ttnn.experimental, name, wrap_matmul(getattr(ttnn.experimental, name)))
    klass = getattr(importlib.import_module(module), cls)
    for name in method.split():
        monkeypatch.setattr(klass, name, wrap_forward(getattr(klass, name)))

    with pytest.raises(_Stop):
        run(mesh_device, model_location_generator)

    assert lowest, "no matmul ran in the first forward"
    base = _allocator_base(mesh_device)
    result = {
        "case": request.node.callspec.id,
        "arch": "blackhole" if ttnn.device.is_blackhole() else "wormhole_b0",
        "mesh": list(mesh_device.shape),
        "grid": str(mesh_device.compute_with_storage_grid_size()),
        "l1_limit": base + min(lowest.values()),
        "per_matmul": [
            {"K": k, "N": n, "lowest_allocated": base + lowest[(k, n)], "calls": calls[(k, n)]}
            for k, n in sorted(lowest)
        ],
    }
    out = Path("generated/outputs/l1_probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['case']}.json").write_text(json.dumps(result, indent=2))
    logger.info(f"L1_PROBE {json.dumps(result)}")
