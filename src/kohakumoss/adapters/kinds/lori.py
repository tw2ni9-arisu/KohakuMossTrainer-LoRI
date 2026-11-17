"""LoRI with frozen random projection and dense or compact trainable B."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.adapters.bind.eager import module_bypass
from kohakumoss.adapters.kinds.common import unit_diff
from kohakumoss.adapters.kinds.record import Kind, Stacking, register


class LoRIModule(nn.Module):
    """Linear LoRI delta; sparse indices address flattened B[out, rank]."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        lora_dim: int = 8,
        alpha: float = 16.0,
        *,
        phase: str = "calibration",
        a: Tensor | None = None,
        active_indices: Tensor | None = None,
    ):
        super().__init__()
        if lora_dim <= 0:
            raise ValueError("LoRI rank must be positive")
        if phase not in ("calibration", "sparse"):
            raise ValueError("LoRI phase must be calibration or sparse")
        if not hasattr(org_module, "in_features") or not hasattr(
            org_module, "out_features"
        ):
            raise ValueError("LoRI requires a linear operator")
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.shape = (org_module.out_features, org_module.in_features)
        self.phase = phase
        a_shape = (lora_dim, self.shape[1])
        if a is None:
            bound = math.sqrt(3 / self.shape[1])
            a = torch.empty(a_shape).uniform_(-bound, bound)
        elif a.shape != a_shape or not a.is_floating_point():
            raise ValueError("LoRI A must be floating point with shape [rank, in]")
        self.register_buffer("A", a.detach().clone())
        self.register_buffer("alpha", self.A.new_tensor(float(alpha)))
        if phase == "calibration":
            if active_indices is not None:
                raise ValueError("Calibration does not accept sparse indices")
            self.b_dense = nn.Parameter(self.A.new_zeros((self.shape[0], lora_dim)))
            self._dense_b = type(self)._calibration_b
        else:
            count = self.shape[0] * lora_dim
            if (
                active_indices is None
                or active_indices.dtype != torch.int64
                or active_indices.ndim != 1
                or bool((active_indices < 0).any())
                or bool((active_indices >= count).any())
                or bool((active_indices[1:] <= active_indices[:-1]).any())
            ):
                raise ValueError("LoRI indices must be sorted unique int64 B offsets")
            indices = active_indices.detach().to(self.A.device).clone()
            self.register_buffer("active_indices", indices)
            self.b_values = nn.Parameter(self.A.new_zeros(indices.numel()))
            self._dense_b = type(self)._sparse_b

    @property
    def scale(self) -> Tensor:
        """The persistent alpha divided by rank."""
        return self.alpha / self.lora_dim

    def _calibration_b(self) -> Tensor:
        return self.b_dense

    def _sparse_b(self) -> Tensor:
        flat = self.b_values.new_zeros(self.shape[0] * self.lora_dim)
        return flat.scatter(0, self.active_indices, self.b_values).reshape(
            self.shape[0], self.lora_dim
        )

    def dense_b(self) -> Tensor:
        """B[out, rank], with differentiable zero fill outside sparse indices."""
        return self._dense_b(self)

    def bypass_forward_diff(self, x: Tensor, multiplier: float = 1.0) -> Tensor:
        """Compute multiplier * alpha/rank * B(Ax) in the input dtype."""
        return F.linear(F.linear(x, self.A.to(x.dtype)), self.dense_b().to(x.dtype)) * (
            self.scale * multiplier
        )

    def get_diff_weight(self, multiplier: float = 1.0) -> tuple[Tensor, None]:
        """Materialize the scaled delta weight and no bias."""
        return (self.dense_b() @ self.A) * (self.scale * multiplier), None


def _module_forward(t, ctx):
    return module_bypass(t.layer, t.lyco, ctx.m)


def _comfy_tensors(t) -> dict[str, Tensor]:
    module = t.lyco
    return {
        "lora_up.weight": module.dense_b().detach().float(),
        "lora_down.weight": module.A.detach().float(),
        "alpha": module.alpha.detach().float(),
    }


LORI = register(
    Kind(
        name="lori",
        module=LoRIModule,
        algos={"lori": "lora"},
        diff_weight=unit_diff,
        module_forward=_module_forward,
        comfy_tensors=_comfy_tensors,
        stacking=Stacking.ADDITIVE,
    )
)
