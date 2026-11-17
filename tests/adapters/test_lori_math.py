"""Model-free LoRI factor, gradient and conversion contracts."""

import math
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.adapters.build.target import Target
from kohakumoss.adapters.kinds import BY_ALGO, record_of
from kohakumoss.adapters.kinds.lori import LoRIModule
from tests._util.negative import must_fail


def make_module(phase="calibration", indices=None, a=None):
    return LoRIModule(
        "adapter",
        nn.Linear(5, 4),
        lora_dim=3,
        alpha=7,
        phase=phase,
        a=a,
        active_indices=indices,
    )


def test_initialization():
    module = make_module()
    assert module.A.shape == (3, 5)
    assert module.A.abs().max() <= math.sqrt(3 / 5)
    assert not module.A.requires_grad
    assert set(dict(module.named_parameters())) == {"b_dense"}
    assert not module.b_dense.any()
    assert not module.get_diff_weight()[0].any()
    assert not module.bypass_forward_diff(torch.randn(2, 5)).any()
    assert set(module.state_dict()) == {"A", "alpha", "b_dense"}


@pytest.mark.parametrize("indices", [[], [0, 2, 7, 11], list(range(12))])
@pytest.mark.parametrize("multiplier", [0.0, 0.5, 1.0])
def test_sparse_matches_explicit_mask_and_gradients(indices, multiplier):
    indices = torch.tensor(indices, dtype=torch.int64)
    module = make_module("sparse", indices)
    with torch.no_grad():
        module.b_values.copy_(torch.randn(indices.numel()))
    assert set(dict(module.named_parameters())) == {"b_values"}
    assert set(module.state_dict()) == {"A", "alpha", "active_indices", "b_values"}
    dense = torch.zeros(12)
    dense[indices] = module.b_values.detach()
    dense = dense.reshape(4, 3).requires_grad_()
    x = torch.randn(2, 3, 5, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = module.bypass_forward_diff(x, multiplier)
    expected = F.linear(F.linear(reference_x, module.A), dense) * (7 / 3) * multiplier
    torch.testing.assert_close(module.dense_b(), dense)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(module.b_values.grad, dense.grad.flatten()[indices])
    delta, bias = module.get_diff_weight(multiplier)
    assert bias is None
    torch.testing.assert_close(delta, dense.detach() @ module.A * (7 / 3) * multiplier)


@pytest.mark.parametrize("phase", ["calibration", "sparse"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_buffers_follow_dtype_and_preserve_indices(phase, dtype):
    module = make_module(phase, torch.tensor([1, 5]) if phase == "sparse" else None)
    module.to(dtype=dtype)
    assert module.A.dtype == dtype
    assert module.alpha.dtype == dtype
    assert next(module.parameters()).dtype == dtype
    if phase == "sparse":
        assert module.active_indices.dtype == torch.int64
    if torch.cuda.is_available():
        module.cuda()
        assert all(t.device.type == "cuda" for t in module.state_dict().values())


@pytest.mark.parametrize("phase", ["calibration", "sparse"])
def test_kind_target_and_comfy_scale(phase):
    linear = nn.Linear(5, 4)
    linear._impl = nn.Linear.forward
    module = make_module(phase, torch.tensor([1, 4, 9]) if phase == "sparse" else None)
    with torch.no_grad():
        next(module.parameters()).normal_()
    target = Target("projection", "adapter", linear, module)
    kind = record_of(module)
    assert BY_ALGO["lori"] is kind
    converted = kind.comfy_tensors(target)
    converted_delta = converted["lora_up.weight"] @ converted["lora_down.weight"]
    converted_delta *= converted["alpha"] / converted["lora_down.weight"].shape[0]
    torch.testing.assert_close(converted_delta, kind.diff_weight(target))
    x = torch.randn(2, 5)
    for multiplier in (0, 0.5, 1):
        forward = kind.module_forward(target, types.SimpleNamespace(m=multiplier))
        torch.testing.assert_close(
            forward(x), linear(x) + F.linear(x, converted_delta) * multiplier
        )
        torch.testing.assert_close(
            target.delta(multiplier)[0], converted_delta * multiplier
        )


def test_supplied_a_preserved_and_sparse_b_zero():
    dense = make_module()
    sparse = make_module("sparse", torch.tensor([0, 11]), dense.A)
    assert torch.equal(sparse.A, dense.A)
    assert not sparse.b_values.any()
    assert not sparse.dense_b().any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_calibration_output_and_gradients(dtype):
    module = make_module().to(dtype=dtype)
    with torch.no_grad():
        module.b_dense.normal_()
    x = torch.randn(2, 5, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_b = module.b_dense.detach().clone().requires_grad_()
    expected = F.linear(F.linear(reference_x, module.A), reference_b) * module.scale
    actual = module.bypass_forward_diff(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(module.b_dense.grad, reference_b.grad)
    assert module.A.grad is None


def test_scale_check_rejects_double_scaling():
    module = make_module()
    with torch.no_grad():
        module.b_dense.fill_(1)
    correct = module.get_diff_weight(0.5)[0]
    broken = correct * module.scale
    must_fail(
        lambda: torch.testing.assert_close(broken, correct),
        "LoRI scale applied twice",
    )


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([1, 1]),
        torch.tensor([2, 1]),
        torch.tensor([-1]),
        torch.tensor([12]),
        torch.tensor([1.0]),
    ],
)
def test_invalid_indices_rejected(indices):
    with pytest.raises(ValueError):
        make_module("sparse", indices)


@pytest.mark.parametrize("phase", ["calibration", "sparse"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_delta_with_fp32_projection_and_low_precision_b(phase, dtype):
    indices = torch.tensor([0, 3, 7, 11]) if phase == "sparse" else None
    module = make_module(phase, indices)
    parameter = next(module.parameters())
    parameter.data = (
        torch.linspace(-0.3, 0.5, parameter.numel()).reshape(parameter.shape).to(dtype)
    )
    before = {name: value.clone() for name, value in module.named_buffers()}
    parameter_id = id(parameter)
    b = module.dense_b()
    for multiplier in (0.0, 0.5, 1.0):
        actual, _ = module.get_diff_weight(multiplier)
        expected = (b @ module.A.to(dtype)) * (module.scale * multiplier)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert parameter.grad is not None
    assert id(next(module.parameters())) == parameter_id
    for name, value in module.named_buffers():
        assert value.dtype == before[name].dtype
        assert torch.equal(value, before[name])
