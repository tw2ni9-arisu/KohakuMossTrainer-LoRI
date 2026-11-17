"""LoRI's real 12g optimizer wiring and exact model-free fused continuation."""

import pytest
import torch
from torch import nn

from kohakumoss.adapters.kinds.lori import LoRIModule
from kohakumoss.memory.apply import MemoryEngines
from kohakumoss.memory.profiles import resolve
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.memory.optimizers import build_memory_optim, device_optimizer
from kohakumoss.training.optim.bundle.fused import FusedStepBundle
from kohakumoss.types.plans import PrecisionConfig
from tests._util.negative import must_fail
from tests.training.run._lori_checks import check_compact
from tests.training.run._lori_evidence import assert_exact, snapshot

pytestmark = pytest.mark.cuda


def test_fused_profile_compact_empty_and_restore(tmp_path):
    plan = resolve("ckpt-ring4-actoff-optstream-gradoff", device="cuda:0")
    assert plan.fused_step and plan.optim_state == "stream" and plan.grad_offload
    x = torch.randn(3, 8, device="cuda:0", dtype=torch.bfloat16)

    def make():
        linear = nn.Linear(8, 6).requires_grad_(False)
        modules = nn.ModuleList(
            [
                LoRIModule(
                    "active",
                    linear,
                    2,
                    phase="sparse",
                    active_indices=torch.tensor([0, 5]),
                ),
                LoRIModule(
                    "empty",
                    linear,
                    2,
                    phase="sparse",
                    active_indices=torch.empty(0, dtype=torch.int64),
                ),
            ]
        ).to("cuda:0")
        params = [m.b_values for m in modules]
        engines = MemoryEngines(plan)
        bundle, accumulate = build_memory_optim(
            [
                {
                    "params": params,
                    "names": ["active.b_values", "empty.b_values"],
                    "lr": 0.001,
                    "weight_decay": 0.0,
                }
            ],
            plan,
            engines,
            torch.device("cuda:0"),
            "adamw",
            {"lr": 0.001, "betas": (0.9, 0.95), "eps": 1e-8},
            {"mode": "constant", "warmup": 0, "end": -1},
            3,
            PrecisionConfig(),
            1.0,
        )
        assert isinstance(bundle, FusedStepBundle)
        assert isinstance(bundle.shared, HostStateAdamW)
        assert bundle.hooks is engines.grads
        return modules, engines, bundle, accumulate

    def step(run, microbatches):
        modules, engines, bundle, accumulate = run
        for i in range(microbatches):
            with (
                accumulate(i == microbatches - 1),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                loss = (
                    sum(m.bypass_forward_diff(x).sum() for m in modules) / microbatches
                )
                bundle.backward(loss)
        assert {name for name, _ in bundle._flags} == set(bundle.name_of.values())
        assert bundle.step() and bundle.last_ok
        torch.cuda.synchronize()
        return check_compact(
            [m.b_values for m in modules], device_optimizer(bundle), engines.grads
        )

    first = make()
    try:
        counts = step(first, 2)
        assert counts["moment_numel"] == 4 and counts["host_staging_bytes"] > 0
        manager = CheckpointManager(tmp_path, keep_last=1)
        path = manager.save(
            1,
            {
                "trainable": first[0].state_dict(),
                "ema": {},
                "optim": first[2].state_dict(),
                "step_rng": {},
                "state": {"step": 1},
            },
        )
        saved = manager.load(path)
        before = snapshot(
            {"trainable": first[0].state_dict(), "optim": first[2].state_dict()}
        )
        step(first, 1)
        expected = snapshot(
            {"trainable": first[0].state_dict(), "optim": first[2].state_dict()}
        )
        second = make()
        try:
            ids = [id(m.b_values) for m in second[0]]
            second[0].load_state_dict(saved["trainable"])
            second[2].load_state_dict(saved["optim"])
            assert ids == [id(m.b_values) for m in second[0]]
            assert_exact(
                before,
                {"trainable": second[0].state_dict(), "optim": second[2].state_dict()},
            )
            step(second, 1)
            got = {"trainable": second[0].state_dict(), "optim": second[2].state_dict()}
            assert_exact(expected, got)
            broken = snapshot(got)
            broken["optim"]["clip"] += 0.1
            must_fail(
                lambda: assert_exact(expected, broken), "restored clip factor changed"
            )
        finally:
            second[2].close()
            second[1].close()
    finally:
        first[2].close()
        first[1].close()
