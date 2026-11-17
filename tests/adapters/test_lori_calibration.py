"""Model-free global LoRI mask and artifact checks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.adapters.train.lori import (
    lori_layer_init,
    make_sparse_init,
    read_lori_state,
)
from tests._util.negative import must_fail


def calibration():
    state = {}
    for name, scores in (
        ("lora_dit_blocks_0_q", [1, 2, 3, 4]),
        ("lora_dit_blocks_1_q", [10, 11, 12, 13]),
    ):
        state[f"{name}.A"] = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        state[f"{name}.alpha"] = torch.tensor(4.0)
        state[f"{name}.b_dense"] = torch.tensor(scores, dtype=torch.float32).reshape(
            2, 2
        )
    return state


def test_global_threshold_zero_layer_exact_projection_and_metadata():
    dense = calibration()
    sparse, meta = make_sparse_init(dense, 0.5)
    info = json.loads(meta["kohakumoss_lori"])
    assert info["phase"] == "sparse_init"
    assert info["version"] == 1
    assert info["threshold"] == 7.0
    assert info["nnz"] == 4
    assert info["total"] == 8
    assert info["actual_density"] == 0.5
    for name, nnz in (("lora_dit_blocks_0_q", 0), ("lora_dit_blocks_1_q", 4)):
        assert torch.equal(
            sparse[f"{name}.A"].view(torch.uint8), dense[f"{name}.A"].view(torch.uint8)
        )
        assert sparse[f"{name}.b_values"].numel() == nnz
        assert torch.count_nonzero(sparse[f"{name}.b_values"]) == 0
        layer = info["layers"][name]
        assert layer["target_shape"] == [2, 3]
        assert layer["rank"] == 2 and layer["scale"] == 2.0
        assert layer["nnz"] == nnz
        init = lori_layer_init(sparse, name)
        assert torch.equal(init["a"], dense[f"{name}.A"])
        assert init["lora_dim"] == 2
    assert not any("b_dense" in key for key in sparse)


@pytest.mark.parametrize(
    "scores,sparsity,expected",
    [
        ([0, 0, 0, 0], 0.99, [0, 1, 2, 3]),
        ([1, 2, 2, 3], 0.5, [1, 2, 3]),
        ([1, 2, 3, 4], 0, [0, 1, 2, 3]),
        ([-1, -2, -3, -4], 0.5, [2, 3]),
    ],
)
def test_ties_zero_and_absolute_scores(scores, sparsity, expected):
    state = {
        "layer.A": torch.ones(2, 3),
        "layer.alpha": torch.tensor(2.0),
        "layer.b_dense": torch.tensor(scores, dtype=torch.float32).reshape(2, 2),
    }
    sparse, _ = make_sparse_init(state, sparsity)
    assert sparse["layer.active_indices"].tolist() == expected


@pytest.mark.parametrize("sparsity", [-0.1, 1.0, 1.1, float("nan"), float("inf")])
def test_invalid_sparsity(sparsity):
    with pytest.raises(ValueError):
        make_sparse_init(calibration(), sparsity)


@pytest.mark.parametrize(
    "key,value",
    [
        ("A", torch.ones(0, 3)),
        ("A", torch.ones(3, 3)),
        ("A", torch.ones(2, 0)),
        ("A", torch.full((2, 3), float("nan"))),
        ("alpha", torch.tensor(float("inf"))),
        ("alpha", torch.ones(2)),
        ("b_dense", torch.ones(2, 3)),
        ("b_dense", torch.full((2, 2), float("inf"))),
    ],
)
def test_invalid_dense_state(key, value):
    state = calibration()
    state[f"lora_dit_blocks_0_q.{key}"] = value
    with pytest.raises(ValueError):
        make_sparse_init(state, 0.5)


def test_reject_missing_and_unrelated_state():
    for state in (
        {},
        {"layer.A": torch.ones(2, 3)},
        {**calibration(), "moment": torch.ones(1)},
    ):
        with pytest.raises(ValueError):
            make_sparse_init(state, 0.5)


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([-1]),
        torch.tensor([4]),
        torch.tensor([1, 1]),
        torch.tensor([2, 1]),
        torch.tensor([1.0]),
    ],
)
def test_read_rejects_invalid_indices(tmp_path, indices):
    sparse, meta = make_sparse_init(calibration(), 0.5)
    sparse["lora_dit_blocks_0_q.active_indices"] = indices
    save_file(sparse, tmp_path / "bad.safetensors", metadata=meta)
    with pytest.raises(ValueError):
        read_lori_state(tmp_path / "bad.safetensors")


@pytest.mark.parametrize("attempt", [0, 1])
def test_roundtrip_metadata_digest_and_cli(tmp_path, attempt):
    dense = calibration()
    source = tmp_path / "calibration.safetensors"
    save_file(
        dense,
        source,
        metadata={
            "kohakumoss_adapter_state": "1",
            "kohakumoss_spec": "{}",
            "kohakumoss_fold": "null",
        },
    )
    loaded, _ = read_lori_state(source)
    sparse, meta = make_sparse_init(loaded, 0.5)
    script = Path(__file__).parents[2] / "scripts" / "adapters" / "lori_mask.py"
    output = tmp_path / f"sparse{attempt}.safetensors"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            str(output),
            "--sparsity",
            "0.5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual, metadata = read_lori_state(output)
    assert metadata["kohakumoss_spec"] == "{}"
    assert metadata["kohakumoss_lori"] == meta["kohakumoss_lori"]
    assert actual.keys() == sparse.keys()
    for key in sparse:
        assert torch.equal(actual[key], sparse[key])
    assert output.read_bytes() != b""

    sparse["lora_dit_blocks_1_q.A"][0, 0] = 99
    save_file(sparse, tmp_path / "tampered.safetensors", metadata=meta)
    with pytest.raises(ValueError, match="digest"):
        read_lori_state(tmp_path / "tampered.safetensors")


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 999),
        ("phase", "calibration"),
        ("total", 99),
        ("nnz", 99),
        ("actual_density", 0.1),
        ("threshold", -1),
    ],
)
def test_reject_invalid_global_metadata(tmp_path, field, value):
    state, meta = make_sparse_init(calibration(), 0.5)
    info = json.loads(meta["kohakumoss_lori"])
    info[field] = value
    meta["kohakumoss_lori"] = json.dumps(info)
    save_file(state, tmp_path / "bad.safetensors", metadata=meta)
    with pytest.raises(ValueError):
        read_lori_state(tmp_path / "bad.safetensors")


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_shape", [0, 3]),
        ("target_shape", [2, 4]),
        ("target_shape", [1, 3]),
        ("rank", 3),
        ("scale", 9),
        ("nnz", 3),
    ],
)
def test_reject_invalid_layer_metadata(tmp_path, field, value):
    state, meta = make_sparse_init(calibration(), 0.5)
    info = json.loads(meta["kohakumoss_lori"])
    info["layers"]["lora_dit_blocks_1_q"][field] = value
    meta["kohakumoss_lori"] = json.dumps(info)
    save_file(state, tmp_path / "bad.safetensors", metadata=meta)
    with pytest.raises(ValueError):
        read_lori_state(tmp_path / "bad.safetensors")


def test_sparse_init_requires_zero_b_but_resume_accepts_values(tmp_path):
    state, meta = make_sparse_init(calibration(), 0.5)
    state["lora_dit_blocks_1_q.b_values"].fill_(1)
    path = tmp_path / "state.safetensors"
    save_file(state, path, metadata=meta)
    with pytest.raises(ValueError, match="zero"):
        read_lori_state(path)
    info = json.loads(meta["kohakumoss_lori"])
    info["phase"] = "sparse"
    meta["kohakumoss_lori"] = json.dumps(info)
    save_file(state, path, metadata=meta)
    restored, _ = read_lori_state(path)
    assert torch.equal(
        restored["lora_dit_blocks_1_q.b_values"], state["lora_dit_blocks_1_q.b_values"]
    )


def test_calibration_digest_tracks_dense_values_and_order():
    dense = calibration()
    _, first = make_sparse_init(dense, 0.5)
    _, reordered = make_sparse_init(dict(reversed(list(dense.items()))), 0.5)
    assert first == reordered
    dense["lora_dit_blocks_0_q.b_dense"][0, 0] = 0.5
    _, changed = make_sparse_init(dense, 0.5)
    assert (
        json.loads(first["kohakumoss_lori"])["calibration_digest"]
        != json.loads(changed["kohakumoss_lori"])["calibration_digest"]
    )


def test_global_mask_check_rejects_per_layer_quantile():
    state = calibration()
    sparse, _ = make_sparse_init(state, 0.5)
    scores = state["lora_dit_blocks_0_q.b_dense"].flatten().abs()
    per_layer = torch.nonzero(scores >= torch.quantile(scores, 0.5), as_tuple=True)[0]
    must_fail(
        lambda: torch.testing.assert_close(
            sparse["lora_dit_blocks_0_q.active_indices"], per_layer
        ),
        "per-layer threshold substituted for global quantile",
    )
