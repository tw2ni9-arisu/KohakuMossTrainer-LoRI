"""LoRI export and checkpoint facts: both formats name the phase, mechanism, trainable
tensor, frozen projection digest and mask counts; the comfy container is marked
inference-only and warns when a sparse mask is expanded; the native file keeps the mask;
a checkpoint's state.json carries the same facts under a lori block."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kohakumoss.adapters.build.apply import apply
from kohakumoss.adapters.train.lori import META_KEY, make_sparse_init
from kohakumoss.training.checkpoint.io import save_checkpoint
from kohakumoss.training.drivers.common import Parts
from kohakumoss.training.state import TrainState
from tests.adapters.test_lori_build import config, dense_state, operators

pytestmark = pytest.mark.cpu
FACT_KEYS = (
    "kohakumoss_lori_phase",
    "kohakumoss_lori_mechanism",
    "kohakumoss_lori_trainable",
    "kohakumoss_lori_a",
    "kohakumoss_lori_sparsity",
    "kohakumoss_lori_a_sha256",
)


@pytest.fixture(autouse=True)
def unfolded_metadata(monkeypatch):
    monkeypatch.setattr(
        "kohakumoss.adapters.train.adapter_set.fold_record", lambda _: None
    )


def calibration_adapters():
    """LoRI on two linears in the dense calibration phase, B non-zero."""
    aset = apply(operators(), config())
    with torch.no_grad():
        for p in aset.trainable_params():
            p.fill_(0.25)
    return aset


def sparse_adapters(tmp_path):
    """LoRI on two linears in the sparse phase, from a written mask artifact."""
    state, meta = make_sparse_init(dense_state(), 0.6)
    artifact = tmp_path / "mask.safetensors"
    save_file(state, artifact, metadata=meta)
    cfg = config(
        algo_kwargs={"phase": "sparse", "sparsity": 0.6, "mask_path": str(artifact)}
    )
    return apply(operators(), cfg)


def reference_digest(aset) -> str:
    """The documented digest: every layer's A as fp32 bytes, in lora_name order."""
    digest = hashlib.sha256()
    for t in sorted(aset.targets, key=lambda t: t.lora_name):
        a = t.lyco.A.detach().to(torch.float32).contiguous()
        digest.update(a.numpy().tobytes())
    return digest.hexdigest()


def kept(aset) -> int:
    return sum(t.lyco.active_indices.numel() for t in aset.targets)


def read(path):
    """(tensors, metadata) of a written safetensors file."""
    with safe_open(path, framework="pt") as handle:
        return (
            {key: handle.get_tensor(key) for key in handle.keys()},
            handle.metadata(),
        )


@pytest.mark.parametrize("fmt", ["lycoris", "comfy"])
def test_calibration_metadata_names_a_dense_run(fmt):
    aset = calibration_adapters()
    meta = aset.metadata(fmt, "lora_unet")
    assert meta["kohakumoss_lori_phase"] == "calibration"
    assert meta["kohakumoss_lori_mechanism"] == "dense_calibration"
    assert meta["kohakumoss_lori_trainable"] == "b_dense"
    assert meta["kohakumoss_lori_a"] == "frozen_buffer"
    assert meta["kohakumoss_lori_sparsity"] == "0.0"
    assert meta["kohakumoss_container"] == "lora"
    assert "kohakumoss_lori_kept_total" not in meta
    assert "kohakumoss_lori_rank" not in meta
    assert ("kohakumoss_lori_comfy" in meta) == (fmt == "comfy")
    assert meta["kohakumoss_lori_a_sha256"] == reference_digest(aset)
    assert json.loads(meta["ss_network_args"]) == {"algo": "lora"}


@pytest.mark.parametrize("fmt", ["lycoris", "comfy"])
def test_sparse_metadata_names_the_mask(fmt, tmp_path):
    aset = sparse_adapters(tmp_path)
    meta = aset.metadata(fmt, "lora_unet")
    for key in FACT_KEYS:
        assert key in meta
    assert meta["kohakumoss_lori_phase"] == "sparse"
    assert meta["kohakumoss_lori_mechanism"] == "sparse_mask"
    assert meta["kohakumoss_lori_trainable"] == "b_values"
    assert meta["kohakumoss_lori_sparsity"] == "0.6"
    assert meta["kohakumoss_lori_kept_total"] == str(kept(aset))
    assert meta["kohakumoss_lori_rank"] == "2"
    assert meta["kohakumoss_lori_a_sha256"] == reference_digest(aset)
    args = json.loads(meta["ss_network_args"])
    assert args == {"algo": "lora"}
    assert not [key for key in args if "lori" in key]
    assert meta["kohakumoss_container"] == "lora"


def test_native_state_metadata_carries_the_facts(tmp_path):
    aset = sparse_adapters(tmp_path)
    sparse = aset.state_metadata()
    assert json.loads(sparse[META_KEY])["phase"] == "sparse"
    assert sparse["kohakumoss_lori_mechanism"] == "sparse_mask"
    assert sparse["kohakumoss_lori_trainable"] == "b_values"
    assert sparse["kohakumoss_lori_sparsity"] == "0.6"
    assert sparse["kohakumoss_lori_kept_total"] == str(kept(aset))
    assert sparse["kohakumoss_lori_a_sha256"] == reference_digest(aset)
    dense = calibration_adapters()
    meta = dense.state_metadata()
    assert META_KEY not in meta
    assert meta["kohakumoss_lori_phase"] == "calibration"
    assert meta["kohakumoss_lori_mechanism"] == "dense_calibration"
    assert meta["kohakumoss_lori_trainable"] == "b_dense"
    assert meta["kohakumoss_lori_sparsity"] == "0.0"
    assert meta["kohakumoss_lori_a_sha256"] == reference_digest(dense)


@pytest.mark.parametrize("layer", [0, 1])
def test_projection_digest_is_stable_and_tracks_every_layer(tmp_path, layer):
    aset = calibration_adapters()
    expected = reference_digest(aset)
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    aset.export(str(first), "comfy", dtype=torch.float32)
    aset.export(str(second), "comfy", dtype=torch.float32)
    _, one = read(first)
    _, two = read(second)
    assert one["kohakumoss_lori_a_sha256"] == expected
    assert two["kohakumoss_lori_a_sha256"] == expected
    with torch.no_grad():
        aset.targets[layer].lyco.A.add_(0.5)
    assert aset.metadata("comfy", "lora_unet")["kohakumoss_lori_a_sha256"] != expected


def test_sparse_native_export_keeps_the_mask(tmp_path):
    aset = sparse_adapters(tmp_path)
    out = tmp_path / "native.safetensors"
    aset.export(str(out), "lycoris")
    tensors, meta = read(out)
    assert {key.rsplit(".", 1)[1] for key in tensors} == {
        "A",
        "alpha",
        "b_values",
        "active_indices",
    }
    assert meta["kohakumoss_format"] == "kohakumoss_lori"
    assert meta["kohakumoss_lori_mechanism"] == "sparse_mask"
    assert meta["kohakumoss_lori_a_sha256"] == reference_digest(aset)
    for t in aset.targets:
        saved = f"{t.lora_name}."
        assert torch.equal(tensors[saved + "A"], t.lyco.A)
        assert torch.equal(tensors[saved + "b_values"], t.lyco.b_values)
        assert torch.equal(tensors[saved + "active_indices"], t.lyco.active_indices)


def test_sparse_comfy_export_is_a_marked_inference_container(tmp_path):
    aset = sparse_adapters(tmp_path)
    out = tmp_path / "comfy.safetensors"
    aset.export(str(out), "comfy", dtype=torch.float32)
    tensors, meta = read(out)
    for t in aset.targets:
        stem = f"lora_unet_{t.path}"
        assert torch.equal(tensors[f"{stem}.lora_down.weight"], t.lyco.A)
        assert torch.equal(tensors[f"{stem}.lora_up.weight"], t.lyco.dense_b())
        assert f"{stem}.alpha" in tensors
    assert meta["kohakumoss_lori_comfy"] == "inference_only_dense_b"
    assert meta["kohakumoss_lori_phase"] == "sparse"
    assert meta["kohakumoss_lori_trainable"] == "b_values"
    assert meta["kohakumoss_container"] == "lora"


def test_comfy_sparse_export_warns_inference_only(tmp_path, caplog):
    aset = sparse_adapters(tmp_path)
    out = tmp_path / "comfy.safetensors"
    with caplog.at_level("WARNING", logger="kohakumoss.adapters.train.adapter_set"):
        aset.export(str(out), "comfy", dtype=torch.float32)
    assert out.exists()
    assert "sparse mask expanded to dense LoRA" in caplog.text
    assert "inference only" in caplog.text
    assert "resuming training would train A" in caplog.text


def test_comfy_calibration_export_does_not_warn(tmp_path, caplog):
    aset = calibration_adapters()
    out = tmp_path / "comfy.safetensors"
    with caplog.at_level("WARNING", logger="kohakumoss.adapters.train.adapter_set"):
        aset.export(str(out), "comfy", dtype=torch.float32)
    assert out.exists()
    assert "inference only" not in caplog.text


class Stub:
    """A checkpoint part that returns its stored mapping and no RNG state."""

    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return self.value

    def rng_state(self):
        return {}


def parts_of(aset) -> Parts:
    return Parts(aset, Stub({}), Stub({}), Stub({}), Stub({}))


def test_state_json_records_the_lori_facts(tmp_path):
    aset = sparse_adapters(tmp_path)
    built = parts_of(aset)(TrainState())
    out = save_checkpoint(
        tmp_path / "step_00000001",
        built["trainable"],
        {},
        {},
        {**built["state"], "step": 1},
    )
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert state["step"] == 1
    assert state["lori"] == {
        "phase": "sparse",
        "mechanism": "sparse_mask",
        "trainable": "b_values",
        "sparsity": 0.6,
        "kept_total": kept(aset),
        "a_sha256": reference_digest(aset),
    }


def test_state_json_names_a_calibration_run():
    aset = calibration_adapters()
    block = parts_of(aset)(TrainState())["state"]["lori"]
    assert block["phase"] == "calibration"
    assert block["mechanism"] == "dense_calibration"
    assert block["trainable"] == "b_dense"
    assert block["kept_total"] == sum(t.lyco.b_dense.numel() for t in aset.targets)
    assert block["a_sha256"] == reference_digest(aset)


def test_checkpoint_block_reads_a_given_state(tmp_path):
    aset = sparse_adapters(tmp_path)
    assert aset.checkpoint_metadata(aset.state_dict()) == aset.checkpoint_metadata()


def test_plain_lora_state_json_has_no_lori_block():
    aset = apply(operators(), replace(config(), algo="lora"))
    built = parts_of(aset)(TrainState())
    assert "lori" not in built["state"]
    assert aset.checkpoint_metadata() == {}
    meta = aset.metadata("comfy", "lora_unet")
    assert not [key for key in meta if "lori" in key]
    assert "kohakumoss_container" not in meta
