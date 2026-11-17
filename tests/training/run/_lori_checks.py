"""Compact LoRI optimizer and gradient-offload invariants."""

import torch

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW


def check_compact(params, optimizer, grads) -> dict:
    """Check live identities and compact host state, excluding scalar step counters."""
    params = list(params)
    ids = {id(p) for p in params}
    assert ids == {id(p) for g in optimizer.param_groups for p in g["params"]}
    if isinstance(optimizer, CPUAdamW):
        assert ids == {id(p) for p in optimizer.state}
    else:
        assert isinstance(optimizer, HostStateAdamW)
        assert ids == set(optimizer.host) == set(optimizer.steps)
    assert ids == {id(p) for p in grads.params} == set(grads.lanes)
    nnz = sum(p.numel() for p in params)
    moments = 0
    for p in params:
        assert p.ndim == 1
        state = (
            [optimizer.state[p][name] for name in ("master", "exp_avg", "exp_avg_sq")]
            if isinstance(optimizer, CPUAdamW)
            else optimizer.host[id(p)]
        )
        for i, value in enumerate(state):
            assert value.shape == p.shape
            assert value.device.type == "cpu" and value.dtype == torch.float32
            assert torch.isfinite(value).all()
            if i:
                moments += value.numel()
        assert grads.host[p].shape == p.shape
        assert grads.host[p].dtype == torch.float32
        if isinstance(optimizer, CPUAdamW):
            assert optimizer._lanes[id(p)].grad_host is grads.host[p]
        else:
            assert any(
                optimizer.group_of[id(p)] is group for group in optimizer.param_groups
            )
    assert moments == 2 * nnz
    if isinstance(optimizer, HostStateAdamW):
        largest = max(p.numel() for p in params)
        assert all(t.shape == (4, largest) for t in optimizer.slots)
    assert sum(t.numel() for t in grads.host.values()) == nnz
    for device, ring in grads.rings.items():
        largest = max(p.numel() for p in params if p.device == device)
        assert all(t.numel() == largest for t in ring.buffers)
    largest_bytes = max(p.nbytes for p in params)
    staging = grads.acc.staging_bytes()
    assert staging <= len(grads.acc.ring) * largest_bytes
    return {
        "nnz": nnz,
        "moment_numel": moments,
        "host_grad_numel": nnz,
        "host_staging_bytes": staging,
        "host_staging_bound": len(grads.acc.ring) * largest_bytes,
        "optimizer_state_ring_bytes": (
            sum(t.nbytes for t in optimizer.slots)
            if isinstance(optimizer, HostStateAdamW)
            else 0
        ),
    }
