"""LoRI configuration, native target construction and compact optimizer groups."""

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kohakumoss.adapters.build import network
from kohakumoss.adapters.build.apply import apply
from kohakumoss.adapters.build.config import (
    AdapterConfig,
    accepted_kwargs,
    config_from_knobs,
)
from kohakumoss.adapters.build.network import build_targets
from kohakumoss.adapters.kinds.lori import LoRIModule
from kohakumoss.adapters.train.lori import make_sparse_init
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def config(**kwargs):
    return AdapterConfig(
        algo="lori", preset={"target_module": ["Linear"]}, dim=2, alpha=3, **kwargs
    )


def operators():
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def dense_state():
    return {
        f"lora_unet_{i}.{key}": value
        for i, (out_dim, in_dim) in enumerate(((3, 4), (2, 3)))
        for key, value in {
            "A": torch.arange(2 * in_dim, dtype=torch.float32).reshape(2, in_dim),
            "alpha": torch.tensor(7.0),
            "b_dense": torch.arange(out_dim * 2, dtype=torch.float32).reshape(
                out_dim, 2
            ),
        }.items()
    }


def test_config_accepts_only_local_options():
    assert (
        config(algo_kwargs={"phase": "calibration", "sparsity": 0.9}).validate() == []
    )
    assert accepted_kwargs("lori") == {"phase", "sparsity", "mask_path"}
    assert AdapterConfig().validate() == []
    assert AdapterConfig().module_kwargs() == {"bypass_mode": True}


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"dora": True}, "DORA"),
        ({"dropout": 0.1}, "dropout"),
        ({"rank_dropout": 0.1}, "dropout"),
        ({"module_dropout": 0.1}, "dropout"),
        ({"train_norms": True}, "TRAIN_NORMS"),
        ({"lr_ratio_up": 2.0}, "LORAPLUS"),
        ({"algo_kwargs": {"a": torch.zeros(2, 4)}}, "argument"),
        ({"algo_kwargs": {"phase": "unknown"}}, "phase"),
        ({"algo_kwargs": {"sparsity": float("nan")}}, "sparsity"),
    ],
)
def test_config_refuses_unsupported_options(kwargs, needle):
    assert any(needle in error for error in config(**kwargs).validate())


def test_calibration_discovery_and_lora_regression():
    for cfg, kind in (
        (config(), LoRIModule),
        (replace(config(), algo="lora"), nn.Module),
    ):
        targets, norms = build_targets(operators(), cfg)
        assert [target.lora_name for target in targets] == [
            "lora_unet_0",
            "lora_unet_1",
        ]
        assert not norms
        assert all(isinstance(target.lyco, kind) for target in targets)
        if cfg.algo == "lori":
            assert all(
                list(dict(target.lyco.named_parameters())) == ["b_dense"]
                for target in targets
            )


def test_initial_state_precedes_stale_mask_and_load_preserves_parameters():
    state, _ = make_sparse_init(dense_state(), 0.6)
    for key, value in state.items():
        if key.endswith("b_values"):
            value.fill_(0.25)
    cfg = config(algo_kwargs={"phase": "sparse", "mask_path": "missing.safetensors"})
    adapters = apply(operators(), cfg, initial_state=state)
    parameters = [target.lyco.b_values for target in adapters.targets]
    assert all(torch.count_nonzero(parameter) == 0 for parameter in parameters)
    adapters.load_state_dict(state)
    for target, parameter in zip(adapters.targets, parameters):
        assert target.lyco.b_values is parameter
        assert torch.equal(target.lyco.A, state[f"{target.lora_name}.A"])
        assert torch.equal(target.lyco.alpha, state[f"{target.lora_name}.alpha"])
        assert torch.equal(parameter, state[f"{target.lora_name}.b_values"])
    groups = adapters.param_groups(lr=0.02, weight_decay=0.1, rules="all")
    assert len(groups) == 1
    check_compact_groups(groups, parameters)
    assert groups[0]["lr"] == 0.02 and groups[0]["weight_decay"] == 0.1
    assert not groups[0]["use_muon"]
    assert adapters.param_groups(0.02, 0.1, rules="moss")[0]["weight_decay"] == 0
    assert adapters.param_groups(0.02)[0]["weight_decay"] == 0


def test_mask_artifact_read_once(tmp_path, monkeypatch):
    state, metadata = make_sparse_init(dense_state(), 0.6)
    path = tmp_path / "mask.safetensors"
    save_file(state, path, metadata=metadata)
    original = network.read_lori_state
    calls = []

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(network, "read_lori_state", read)
    targets, _ = build_targets(
        operators(), config(algo_kwargs={"phase": "sparse", "mask_path": str(path)})
    )
    assert len(calls) == 1
    assert all(isinstance(t.lyco, LoRIModule) for t in targets)


@pytest.mark.parametrize(
    "damage", ["name", "rank", "input", "output", "indices", "phase"]
)
def test_initial_state_mismatch_rejected(damage):
    state = dense_state()
    if damage == "name":
        state = {k.replace("unet_0", "unet_extra"): v for k, v in state.items()}
    elif damage == "rank":
        state["lora_unet_0.A"] = torch.zeros(3, 4)
        state["lora_unet_0.b_dense"] = torch.zeros(3, 3)
    elif damage == "input":
        state["lora_unet_0.A"] = torch.zeros(2, 5)
    elif damage == "output":
        state["lora_unet_0.b_dense"] = torch.zeros(4, 2)
    elif damage == "indices":
        state, _ = make_sparse_init(state, 0.6)
        state["lora_unet_0.active_indices"][0] = 99
    cfg = config(
        algo_kwargs={
            "phase": "sparse" if damage in ("indices", "phase") else "calibration"
        }
    )
    with pytest.raises(ValueError):
        build_targets(operators(), cfg, initial_state=state)


def test_artifact_output_shape_mismatch_rejected(tmp_path):
    state = dense_state()
    state["lora_unet_0.b_dense"] = torch.arange(8.0).reshape(4, 2)
    sparse, metadata = make_sparse_init(state, 0.6)
    path = tmp_path / "wrong_shape.safetensors"
    save_file(sparse, path, metadata=metadata)
    with pytest.raises(ValueError, match="target shape"):
        build_targets(
            operators(), config(algo_kwargs={"phase": "sparse", "mask_path": str(path)})
        )


def test_compact_update_matches_masked_dense_with_all_decay():
    state, _ = make_sparse_init(dense_state(), 0.6)
    adapters = apply(
        operators(), config(algo_kwargs={"phase": "sparse"}), initial_state=state
    )
    groups = adapters.param_groups(0.02, 0.1, rules="all")
    compact = torch.optim.AdamW(groups)
    for target in adapters.targets:
        module = target.lyco
        with torch.no_grad():
            module.b_values.fill_(0.25)
        dense = nn.Parameter(module.dense_b().detach())
        reference = torch.optim.AdamW([dense], lr=0.02, weight_decay=0.1)
        dense.grad = torch.zeros_like(dense)
        dense.grad.view(-1)[module.active_indices] = 2
        module.b_values.grad = torch.full_like(module.b_values, 2)
        reference.step()
        target.expected_values = dense.detach().view(-1)[module.active_indices].clone()
    compact.step()
    for target in adapters.targets:
        torch.testing.assert_close(target.lyco.b_values, target.expected_values)


def test_lori_dora_network_kwarg_is_not_silently_consumed():
    cfg = config_from_knobs({"ALGO": "lori", "ALGO_KWARGS": {"wd_on_out": False}})
    assert any("wd_on_out" in error for error in cfg.validate())


def check_compact_groups(groups, expected):
    parameters = [p for group in groups for p in group["params"]]
    assert [id(p) for p in parameters] == [id(p) for p in expected]
    assert all(p.ndim == 1 for p in parameters)


def test_compact_group_check_rejects_frozen_projection():
    values = nn.Parameter(torch.zeros(2))
    projection = torch.zeros(2, 4)
    must_fail(
        lambda: check_compact_groups([{"params": [values, projection]}], [values]),
        "frozen projection in compact groups",
    )
