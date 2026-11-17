"""Full stored-INT8 MOSS LoRI gate with synthetic latent/text operator inputs.

This numerical/offload gate does not certify real image or caption learning.
"""

import gc
import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from kohakumoss.adapters.train.lori import make_sparse_init
from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.offload.weights.stream import StreamedQuantWeight
from kohakumoss.precision import autocast_factory, build_precision
from kohakumoss.training.build.loop import build_optim, build_step
from kohakumoss.training.build.model import build_model
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.checkpoint.io import load_lori_initialization
from kohakumoss.training.memory.optimizers import device_optimizer
from kohakumoss.training.optim.bundle.fused import FusedStepBundle
from kohakumoss.types.batch import DeviceBatch
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from tests.training.knobs.test_knobs import runnable
from tests.training.run._lori_checks import check_compact
from tests.training.run._lori_evidence import (
    assert_exact,
    batch_state,
    digests,
    snapshot,
)

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]


def _reject_dense_base(*args):
    raise AssertionError("LoRI bypass requested a dense target base")


def _knobs(root, out, phase="calibration", mask=None):
    return runnable(
        WEIGHTS=str(root),
        TRANSFORMER="moss_image_base_int8_convrot.safetensors",
        TEXT_ENCODER="qwen3.5_9b_int8_convrot.safetensors",
        VAE="flux2-vae_small.safetensors",
        ALGO="lori",
        PRESET="attn-mlp",
        DIM=8,
        ALPHA=8.0,
        ALGO_KWARGS={"phase": phase, **({"mask_path": str(mask)} if mask else {})},
        DROPOUT={},
        DORA=False,
        DORA_DETACH=False,
        LORAPLUS_RATIO=1.0,
        TRAIN_NORMS=False,
        PRECISION="bf16",
        FP16_FIX="none",
        MEMORY_PROFILE="ckpt-ring4-actoff-optstream-gradoff",
        BASE_PRECISION="int8",
        LATENT_CACHE="offline",
        TEXT_CACHE="offline",
        SAMPLE_EVERY=0,
        COMPILE="off",
        KERNELS={"lycoris": "torch"},
        OPTIMIZER="adamw",
        LR=1e-3,
        GRAD_CLIP=1.0,
        STEPS=3,
        SCHEDULE={"mode": "constant", "warmup": 0, "end": -1},
        OUTPUT_DIR=str(out),
    )


def _build(k, initial=None):
    dev = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prec = build_precision("bf16", k["KEEP_FP32"], "none", {})
    state, meta = initial if initial else (None, None)
    built = build_model(k, dev, prec, initial_state=state, initial_metadata=meta)
    for target in built.trainable.targets:
        target.base_fn = _reject_dense_base
    assert built.vae is None and built.text_encoder is None
    assert built.plan.checkpoint == "block" and built.plan.act_offload
    assert built.engines.weight_stream.ring.n_slots == 4
    bundle, accumulate = build_optim(k, built, dev, prec, None)
    assert isinstance(bundle, FusedStepBundle)
    assert built.plan.fused_step and built.plan.optim_state == "stream"
    assert bundle.hooks is built.engines.grads
    step = build_step(k, built, TokenNormalizer(), autocast_factory(prec, dev), dev)
    built.engines.reports["gate_build"] = {
        "seconds": time.perf_counter() - start,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }
    return built, bundle, accumulate, step


def _batch(built):
    cfg = built.backbone.cfg
    gen = torch.Generator().manual_seed(17)
    hw = torch.tensor([[16, 16]], dtype=torch.int32)
    seq = build_seqinfo(hw, torch.tensor([8], dtype=torch.int32)).to("cuda:0")
    return DeviceBatch(
        ids=torch.tensor([0]),
        seq=seq,
        latents=torch.randn(256, cfg.latent_channels, generator=gen).to("cuda:0"),
        text=torch.randn(8, cfg.text_dim, generator=gen).to("cuda:0"),
        text_drop=torch.zeros(1, dtype=torch.bool, device="cuda:0"),
        weight=torch.ones(1, device="cuda:0"),
        loss_mask=None,
        pixels=None,
        prompts=None,
    )


def _base_digest(built):
    h = hashlib.sha256()
    stream = built.engines.weight_stream
    managed = {
        id(module.weight)
        for module in built.backbone.modules()
        if getattr(getattr(module, "base", None), "weight_stream", None) is stream
    }
    managed.update(id(value) for value in stream.storages())
    values = [
        (name, value)
        for name, value in [
            *built.backbone.named_parameters(),
            *built.backbone.named_buffers(),
        ]
        if ".lyco." not in name and id(value) not in managed
    ]
    for i, block in enumerate(built.engines.weight_stream.host):
        values.extend((f"host.{i}.{key}", value) for key, value in block.items())
    for name, value in sorted(values):
        h.update(name.encode("ascii"))
        h.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


def _frozen(built):
    ids = {id(p) for p in built.trainable.trainable_params()}
    for p in built.backbone.parameters():
        if id(p) not in ids:
            assert not p.requires_grad and p.grad is None
    return {
        key: value.cpu().clone()
        for key, value in built.trainable.state_dict().items()
        if key.endswith((".A", ".alpha", ".active_indices"))
    }


def _sparse_checks(built, bundle, ids):
    params = list(built.trainable.trainable_params())
    assert [id(p) for p in params] == ids
    assert all(id(t.lyco.b_values) in ids for t in built.trainable.targets)
    assert all(t.stored for t in built.trainable.targets)
    assert all(
        isinstance(t.linear.base, StreamedQuantWeight) for t in built.trainable.targets
    )
    assert all(t.linear.weight.numel() == 0 for t in built.trainable.targets)
    assert any(
        value.dtype == torch.int8
        for block in built.engines.weight_stream.host
        for value in block.values()
    )
    assert len(built.trainable.targets) == built.backbone.cfg.depth * 6
    assert len({t.path for t in built.trainable.targets}) == len(ids)
    for t in built.trainable.targets:
        m = t.lyco
        assert not m.A.requires_grad and m.A.grad is None
        assert m.active_indices.dtype == torch.int64
        assert m.active_indices.device == m.b_values.device == m.A.device
        assert m.b_values.numel() == m.active_indices.numel()
        if m.b_values.numel():
            assert m.b_values.untyped_storage().data_ptr() in built.engines.offload.skip
    return check_compact(params, device_optimizer(bundle), built.engines.grads)


def _step(run, batch, index, prepared=None, expected=None, evidence=None, oracle=False):
    built, bundle, accumulate, step = run
    backend = "eager" if oracle else "triton"
    assert built.backbone.ops.attention_varlen.backend == backend
    calls = []
    handle = built.backbone.layers[0].register_forward_pre_hook(
        lambda *args: calls.append(1)
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    gradients = {}
    handles = [
        p.register_hook(lambda g, pid=id(p): _gradient(gradients, pid, g))
        for p in built.trainable.trainable_params()
    ]

    def capture(p, denom):
        state = snapshot(
            {
                "z": p.z,
                "text": p.text,
                "sigma": p.sigma,
                "target": p.target,
                "weight": p.weight,
                "mask": p.mask,
                "denom": denom,
            }
        )
        if prepared is not None:
            prepared.update(state)
        if evidence is not None:
            evidence.write_text(json.dumps(digests(state), indent=2), encoding="utf-8")
        if expected is not None:
            assert_exact(expected, state, "prepared")

    step.keep = capture
    try:
        with (
            sdpa_kernel(SDPBackend.MATH) if oracle else nullcontext(),
            accumulate(True),
        ):
            out = step(batch, step.denominator(batch), index)
            assert torch.isfinite(out.loss)
            bundle.backward(out.loss)
        assert {name for name, _ in bundle._flags} == set(bundle.name_of.values())
        assert bundle.step()
        assert bundle.last_ok and bundle.last_nonfinite_params == 0
        assert built.backbone.ops.attention_varlen.backend == backend
        built.trainable.end_step()
        torch.cuda.synchronize()
        assert set(gradients) == {id(p) for p in built.trainable.trainable_params()}
        assert all(finite.item() for finite, _ in gradients.values())
        assert any(nonzero.item() for _, nonzero in gradients.values())
        assert len(calls) >= 2, "block checkpoint did not recompute"
        assert not any(built.engines.offload.pending.values())
        assert built.engines.offload.stats["packed"] > 0
        assert built.engines.weight_stream.stats["loads"] >= built.backbone.cfg.depth
        return {
            "loss": bundle.last_loss,
            "grad_norm": bundle.last_norm,
            "seconds": time.perf_counter() - start,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "block0_calls": len(calls),
            "weight_stream": dict(built.engines.weight_stream.stats),
            "activation_offload": dict(built.engines.offload.stats),
            "fused_step": True,
            "optimizer": type(device_optimizer(bundle)).__name__,
            "attention": "eager_math" if oracle else "default_triton",
            "gradient_accumulation": 1,
            "host_gradient_sum_exercised": False,
        }
    finally:
        handle.remove()
        for grad_handle in handles:
            grad_handle.remove()


def _gradient(records, pid, grad):
    records[pid] = (torch.isfinite(grad).all(), torch.count_nonzero(grad))
    return grad


def _oracle(built):
    built.backbone.bind_ops(
        replace(
            built.backbone.ops,
            attention_varlen=registry.resolve(
                "attention_varlen", "cuda:0", torch.bfloat16, backend="eager"
            ),
        )
    )


def _boundary(run, batch):
    built, bundle, _, step = run
    return snapshot(
        {
            "trainable": built.trainable.state_dict(),
            "optim": bundle.state_dict(),
            "rng": {
                "step": step.rng_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state("cuda:0"),
            },
            "batch": batch_state(batch),
            "lr": [group["lr"] for group in device_optimizer(bundle).param_groups],
        }
    )


def _close(run):
    run[1].close()
    run[0].engines.close()


def _equal_state(expected, got):
    assert expected.keys() == got.keys()
    for key, value in expected.items():
        torch.testing.assert_close(value.cpu(), got[key].cpu(), rtol=0, atol=0)


def _progress(out, report):
    (out / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def test_lori_int8_ring_checkpoint_host_restore(tmp_path):
    root = Path(os.environ["KOHAKUMOSS_COMFY"])
    assert (
        root / "diffusion_models" / "moss_image_base_int8_convrot.safetensors"
    ).is_file()
    out = Path(os.environ.get("KOHAKUMOSS_LORI_GATE_OUTPUT", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "input_boundary": "full stored-INT8 MOSS; synthetic 256 latent and 8 text tokens"
    }
    run = _build(_knobs(root, out))
    try:
        report["calibration_build"] = run[0].engines.reports["gate_build"]
        report["compiler_environment"] = {
            key: os.environ.get(key)
            for key in ("CC", "CXX", "VCToolsVersion", "WindowsSDKVersion")
        }
        batch = _batch(run[0])
        a = _frozen(run[0])
        base = _base_digest(run[0])
        report["calibration"] = _step(run, batch, 0)
        _progress(out, report)
        _equal_state(a, _frozen(run[0]))
        assert base == _base_digest(run[0])
        dense = {k: v.cpu() for k, v in run[0].trainable.state_dict().items()}
        first = next(k for k in sorted(dense) if k.endswith(".b_dense"))
        dense[first].zero_()
        sparse, meta = make_sparse_init(dense, 0.9)
        assert any(
            k.endswith(".b_values") and v.numel() == 0 for k, v in sparse.items()
        )
        mask = out / "mask.safetensors"
        save_file(sparse, mask, metadata=meta)
        report["empty_target"] = first.rsplit(".", 1)[0]
    finally:
        _close(run)
    del run, batch, dense
    gc.collect()
    torch.cuda.empty_cache()

    k = _knobs(root, out, "sparse", mask)
    run = _build(k)
    try:
        built, bundle, _, step = run
        report["sparse_build"] = built.engines.reports["gate_build"]
        ids = [id(p) for p in built.trainable.trainable_params()]
        assert all(not p.count_nonzero() for p in built.trainable.trainable_params())
        assert all(
            not s[1].count_nonzero() for s in device_optimizer(bundle).host.values()
        )
        report["compact"] = _sparse_checks(built, bundle, ids)
        fixed = _frozen(built)
        _equal_state(a, {key: fixed[key] for key in a})
        batch = _batch(built)
        report["sparse_step1"] = _step(run, batch, 0)
        assert any(p.count_nonzero() for p in built.trainable.trainable_params())
        assert any(s[1].count_nonzero() for s in device_optimizer(bundle).host.values())
        _progress(out, report)
        _sparse_checks(built, bundle, ids)
        _equal_state(fixed, _frozen(built))
        assert base == _base_digest(built)
        manager = CheckpointManager(out, keep_last=1, devices=(torch.device("cuda:0"),))
        checkpoint = manager.save(
            1,
            {
                "trainable": built.trainable.state_dict(),
                "trainable_meta": built.trainable.state_metadata(),
                "ema": {},
                "optim": bundle.state_dict(),
                "step_rng": step.rng_state(),
                "state": {"step": 1},
            },
        )
        mask.unlink()
        _oracle(built)
        before = _boundary(run, batch)
        (out / "boundary-continuous.json").write_text(
            json.dumps(digests(before), indent=2), encoding="utf-8"
        )
        prepared = {}
        report["continuous_step2"] = _step(
            run,
            batch,
            1,
            prepared=prepared,
            evidence=out / "prepared-continuous.json",
            oracle=True,
        )
        _progress(out, report)
        expected = {k: v.cpu() for k, v in built.trainable.state_dict().items()}
        moments = snapshot(bundle.state_dict())
        _equal_state(fixed, _frozen(built))
        assert base == _base_digest(built)
    finally:
        _close(run)
    del run, built, bundle, step, batch
    gc.collect()
    torch.cuda.empty_cache()

    run = _build(k, load_lori_initialization(checkpoint))
    try:
        built, bundle, _, step = run
        report["resume_build"] = built.engines.reports["gate_build"]
        ids = [id(p) for p in built.trainable.trainable_params()]
        saved = manager.load(checkpoint)
        built.trainable.load_state_dict(saved["trainable"])
        bundle.load_state_dict(saved["optim"])
        step.load_rng_state(saved["step_rng"])
        built.engines.after_restore()
        _sparse_checks(built, bundle, ids)
        _oracle(built)
        batch = _batch(built)
        restored = _boundary(run, batch)
        (out / "boundary-resumed.json").write_text(
            json.dumps(digests(restored), indent=2), encoding="utf-8"
        )
        assert_exact(before, restored, "restored_pre_step")
        report["restore_boundary_exact"] = True
        _equal_state(fixed, _frozen(built))
        assert base == _base_digest(built)
        _progress(out, report)
        report["resumed_step2"] = _step(
            run,
            batch,
            1,
            expected=prepared,
            evidence=out / "prepared-resumed.json",
            oracle=True,
        )
        report["compact_after_resume"] = _sparse_checks(built, bundle, ids)
        _equal_state(fixed, _frozen(built))
        assert base == _base_digest(built)
        _progress(out, report)
        _equal_state(expected, built.trainable.state_dict())
        assert_exact(moments, bundle.state_dict(), "continued_optimizer")
        built.trainable.save_state(out / "final-native.safetensors")
        report.update(
            status="passed",
            checkpoint=str(checkpoint),
            exact_resume=True,
            live_parameter_ids_preserved=True,
            base_digest=base,
            frozen_projection_unchanged=True,
            target_count=len(ids),
        )
        (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        _close(run)
