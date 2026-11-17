"""Portable native LoRI state, export and in-place continuation."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kohakumoss.adapters.build.apply import apply
from kohakumoss.adapters.infer.inference import load
from kohakumoss.adapters.train.lori import META_KEY, make_sparse_init, read_lori_state
from tests.adapters.test_lori_build import config, dense_state, operators

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def unfolded_metadata(monkeypatch):
    monkeypatch.setattr(
        "kohakumoss.adapters.train.adapter_set.fold_record", lambda _: None
    )


def setup_state(tmp_path):
    state, meta = make_sparse_init(dense_state(), 0.6)
    artifact = tmp_path / "mask.safetensors"
    save_file(state, artifact, metadata=meta)
    cfg = config(algo_kwargs={"phase": "sparse", "mask_path": str(artifact)})
    aset = apply(operators(), cfg)
    with torch.no_grad():
        for p in aset.trainable_params():
            p.fill_(0.25)
    return aset, artifact


def test_native_roundtrip_without_artifact(tmp_path):
    aset, artifact = setup_state(tmp_path)
    path = tmp_path / "native.safetensors"
    aset.save_state(path)
    artifact.unlink()
    state, meta = read_lori_state(path)
    assert json.loads(meta[META_KEY])["phase"] == "sparse"
    restored = apply(
        operators(), aset.config, initial_state=state, initial_metadata=meta
    )
    ids = [id(p) for p in restored.trainable_params()]
    optim = torch.optim.AdamW(restored.trainable_params())
    restored.load_state_dict(state)
    assert ids == [id(p) for p in restored.trainable_params()]
    assert ids == [id(p) for g in optim.param_groups for p in g["params"]]
    for key, value in state.items():
        assert torch.equal(value, restored.state_dict()[key])
    for a, b in zip(aset.targets, restored.targets):
        x = torch.randn(2, a.linear.in_features)
        torch.testing.assert_close(
            a.lyco.bypass_forward_diff(x), b.lyco.bypass_forward_diff(x), rtol=0, atol=0
        )


@pytest.mark.parametrize("field", ["version", "rank", "target"])
def test_metadata_refusal_before_placement(tmp_path, monkeypatch, field):
    aset, _ = setup_state(tmp_path)
    meta = aset.state_metadata()
    info = json.loads(meta[META_KEY])
    if field == "version":
        info["version"] += 1
    elif field == "rank":
        info["layers"]["lora_unet_0"]["rank"] += 1
    else:
        # All indices remain legal when the declared output is larger.
        info["layers"]["lora_unet_0"]["target_shape"][0] += 1
        info["total"] += 2
        info["actual_density"] = info["nnz"] / info["total"]
    meta[META_KEY] = json.dumps(info)

    def reject_placement(*args):
        pytest.fail("invalid metadata reached placement")

    monkeypatch.setattr("kohakumoss.adapters.build.apply._place", reject_placement)
    with pytest.raises(ValueError):
        apply(
            operators(),
            aset.config,
            initial_state=aset.state_dict(),
            initial_metadata=meta,
        )


@pytest.mark.parametrize("m", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("fmt", ["state", "lycoris", "comfy"])
def test_native_and_comfy_match_delta(tmp_path, m, fmt):
    aset, artifact = setup_state(tmp_path)
    base = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    weights = {k: v.clone() for k, v in base.state_dict().items()}
    path = tmp_path / "adapter.safetensors"
    if fmt == "state":
        aset.save_state(path)
    else:
        aset.export(path, fmt, dtype=torch.bfloat16)
    artifact.unlink()
    x = torch.randn(2, 4)
    expected = x
    for i, target in enumerate(aset.targets):
        dw, _ = target.lyco.get_diff_weight(m)
        expected = torch.nn.functional.linear(
            expected, weights[f"{i}.weight"] + dw, weights[f"{i}.bias"]
        )
    stack = load(base, str(path), multiplier=m)
    torch.testing.assert_close(
        base(x),
        expected,
        rtol=0.02 if fmt == "comfy" else 1e-5,
        atol=0.02 if fmt == "comfy" else 1e-5,
    )
    stack.remove()


def test_adamw_continuation(tmp_path):
    aset, _ = setup_state(tmp_path)
    opt = torch.optim.AdamW(aset.trainable_params(), lr=0.01)

    def step(adapter, optimizer):
        optimizer.zero_grad()
        sum(p.square().sum() for p in adapter.trainable_params()).backward()
        optimizer.step()

    step(aset, opt)
    path = tmp_path / "state.safetensors"
    aset.save_state(path)
    state, meta = read_lori_state(path)
    other = apply(operators(), aset.config, initial_state=state, initial_metadata=meta)
    other_opt = torch.optim.AdamW(other.trainable_params(), lr=0.01)
    other.load_state_dict(state)
    other_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    step(aset, opt)
    step(other, other_opt)
    for a, b in zip(aset.trainable_params(), other.trainable_params()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_keeps_projection_bytes_under_master_and_save_dtype(tmp_path, dtype):
    aset, artifact = setup_state(tmp_path)
    state, meta = read_lori_state(artifact)
    a = state["lora_unet_0.A"].clone()
    a[0, 0] = 0.1234567
    # Regenerate metadata through the calibration helper for the exact projection.
    dense = dense_state()
    dense["lora_unet_0.A"] = a
    state, meta = make_sparse_init(dense, 0.6)
    restored = apply(
        operators(),
        aset.config,
        dtype=dtype,
        initial_state=state,
        initial_metadata=meta,
    )
    path = tmp_path / "exact.safetensors"
    restored.export(path, "lycoris", dtype=dtype)
    saved, _ = read_lori_state(path)
    assert torch.equal(saved["lora_unet_0.A"], a)
    assert saved["lora_unet_0.A"].dtype == a.dtype


@pytest.mark.parametrize("m", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("fmt", ["lycoris", "comfy", "delta"])
def test_row_scale_export_and_inference(tmp_path, m, fmt):
    aset, _ = setup_state(tmp_path)
    source = {"0": torch.tensor([0.5, 1.0, 2.0]), "1": torch.tensor([2.0, 0.5])}
    destination = {"0": torch.tensor([2.0, 0.5, 1.0]), "1": torch.tensor([0.5, 2.0])}
    # Tensor conversion works entirely on real linear operators.
    tensors = aset.tensors("lycoris" if fmt == "delta" else fmt, row_scales=source)
    base = operators()
    x = torch.randn(2, 4)
    expected = x
    for target in aset.targets:
        layer = base[int(target.path)]
        dw, _ = target.lyco.get_diff_weight(m)
        dw = dw * (destination[target.path] / source[target.path]).reshape(-1, 1)
        expected = torch.nn.functional.linear(expected, layer.weight + dw, layer.bias)
    stack = load(base, tensors, multiplier=m, row_scales=destination)
    torch.testing.assert_close(base(x), expected)
    stack.remove()


@pytest.mark.parametrize("m", [0.0, 0.5, 1.0])
def test_native_export_unfolds_compact_b_without_casting_a(tmp_path, m):
    aset, _ = setup_state(tmp_path)
    scales = {"0": torch.tensor([0.5, 1.0, 2.0]), "1": torch.tensor([2.0, 0.5])}
    path = tmp_path / "unfolded.safetensors"
    aset.export(path, "lycoris", dtype=torch.bfloat16, row_scales=scales)
    state, meta = read_lori_state(path)
    assert meta["kohakumoss_fold"] == "null"
    assert "ss_network_module" not in meta
    for t in aset.targets:
        assert torch.equal(state[f"{t.lora_name}.A"], t.lyco.A)
    base = operators()
    x = torch.randn(2, 4)
    expected = x
    for t in aset.targets:
        layer = base[int(t.path)]
        dw, _ = t.lyco.get_diff_weight(m)
        expected = torch.nn.functional.linear(expected, layer.weight + dw, layer.bias)
    stack = load(base, str(path), multiplier=m, row_scales=scales)
    torch.testing.assert_close(base(x), expected)
    stack.remove()


def test_load_refuses_broadcast_shape_without_partial_copy(tmp_path):
    aset, _ = setup_state(tmp_path)
    state = aset.state_dict()
    original = {key: value.clone() for key, value in state.items()}
    key = next(key for key in state if key.endswith("b_values"))
    state[key] = torch.ones(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        aset.load_state_dict(state)
    for key, value in aset.state_dict().items():
        assert torch.equal(value, original[key])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fmt", ["lycoris", "comfy"])
def test_low_precision_master_folded_delta_export(tmp_path, dtype, fmt):
    aset, artifact = setup_state(tmp_path)
    state, metadata = read_lori_state(artifact)
    for key, value in state.items():
        if key.endswith("b_values"):
            value.fill_(0.25)
    metadata_info = json.loads(metadata[META_KEY])
    metadata_info["phase"] = "sparse"
    metadata[META_KEY] = json.dumps(metadata_info)
    aset = apply(
        operators(),
        aset.config,
        dtype=dtype,
        initial_state=state,
        initial_metadata=metadata,
    )
    aset.load_state_dict(state)
    ids = [id(p) for p in aset.trainable_params()]
    frozen = {
        key: value.clone()
        for key, value in aset.state_dict().items()
        if not key.endswith("b_values")
    }
    scales = {"0": torch.tensor([0.5, 1.0, 2.0]), "1": torch.tensor([2.0, 0.5])}
    for target in aset.targets:
        module = target.lyco
        assert module.A.dtype == torch.float32
        assert module.b_values.dtype == dtype
        reference = (module.dense_b() @ module.A.to(dtype)) * module.scale
        for multiplier in (0.0, 0.5, 1.0):
            torch.testing.assert_close(
                target.delta(multiplier)[0], reference.float() * multiplier
            )
    tensors = aset.tensors("comfy", row_scales=scales)
    for target in aset.targets:
        module = target.lyco
        reference = ((module.dense_b() @ module.A.to(dtype)) * module.scale).float()
        torch.testing.assert_close(
            tensors[f"{target.lora_name}.diff"],
            reference / scales[target.path].reshape(-1, 1),
        )
    path = tmp_path / "folded.safetensors"
    aset.export(path, fmt, dtype=dtype, row_scales=scales)
    # Exported local native files remain readable with exact projection bytes.
    if fmt == "lycoris":
        saved, _ = read_lori_state(path)
        for key, value in frozen.items():
            assert torch.equal(saved[key], value)
            assert saved[key].dtype == value.dtype
    assert ids == [id(p) for p in aset.trainable_params()]
    for key, value in frozen.items():
        assert torch.equal(aset.state_dict()[key], value)
        assert aset.state_dict()[key].dtype == value.dtype
