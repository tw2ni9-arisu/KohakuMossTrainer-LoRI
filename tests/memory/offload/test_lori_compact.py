"""Compact LoRI host optimizer state, empty masks and invariant rejection."""

import pytest
import torch
from torch import nn

from kohakumoss.adapters.kinds.lori import LoRIModule
from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.grad import GradOffload
from tests._util.negative import must_fail
from tests.training.run._lori_checks import check_compact

pytestmark = pytest.mark.cpu


def test_compact_host_state_and_empty_mask():
    linear = nn.Linear(8, 6).requires_grad_(False)
    modules = [
        LoRIModule(
            "active", linear, 2, phase="sparse", active_indices=torch.tensor([0, 5])
        ),
        LoRIModule(
            "empty",
            linear,
            2,
            phase="sparse",
            active_indices=torch.empty(0, dtype=torch.int64),
        ),
    ]
    params = [m.b_values for m in modules]
    ids = [id(p) for p in params]
    a = [m.A.clone() for m in modules]
    grads = GradOffload(params, pinned=False)
    opt = CPUAdamW(params, pinned=False)
    opt.use_host_grads(grads)
    grads.start(False)
    x = torch.randn(3, 8)
    sum(
        m.bypass_forward_diff(x).square().sum() + m.bypass_forward_diff(x).sum()
        for m in modules
    ).backward()
    grads.drain()
    assert all(torch.isfinite(t).all() for t in grads.host.values())
    opt.step()
    assert check_compact(params, opt, grads)["moment_numel"] == 4
    assert [id(p) for p in params] == ids
    assert all(torch.equal(m.A, saved) for m, saved in zip(modules, a))
    assert params[0].count_nonzero() > 0 and params[1].numel() == 0
    opt.state[params[0]]["exp_avg"] = torch.zeros(6, 2)
    must_fail(lambda: check_compact(params, opt, grads), "dense optimizer moments")
    opt.close()
    grads.close()
