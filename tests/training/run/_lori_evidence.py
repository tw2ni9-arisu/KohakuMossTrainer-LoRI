"""Exact state snapshots and JSON tensor digests for the real LoRI gate."""

import hashlib

import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    return value


def assert_exact(expected, got, path="root"):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected.cpu(), got.cpu(), rtol=0, atol=0, msg=path)
    elif isinstance(expected, dict):
        assert expected.keys() == got.keys(), path
        for key, value in expected.items():
            assert_exact(value, got[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(got), path
        for i, (a, b) in enumerate(zip(expected, got)):
            assert_exact(a, b, f"{path}.{i}")
    else:
        assert expected == got, path


def digests(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): digests(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [digests(item) for item in value]
    return value


def batch_state(batch):
    fields = {
        name: getattr(batch, name)
        for name in ("ids", "latents", "text", "text_drop", "weight", "loss_mask")
    }
    fields["seq"] = {
        name: value
        for name, value in vars(batch.seq).items()
        if isinstance(value, (int, torch.Tensor))
    }
    for name in ("attn_joint", "attn_text"):
        plan = getattr(batch.seq, name)
        fields[name] = {
            "cu": plan.cu_seqlens,
            "max": plan.max_seqlen,
            "lengths": plan.lengths,
        }
    return snapshot(fields)
