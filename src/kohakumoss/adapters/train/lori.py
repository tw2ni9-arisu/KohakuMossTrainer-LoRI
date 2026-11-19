"""CPU calibration masks, self-contained raw LoRI initialization artifacts and the
export facts that describe them."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from kohakumoss.adapters.build.config import AdapterConfig

META_KEY = "kohakumoss_lori"
VERSION = 1
MECHANISM_OF = {"calibration": "dense_calibration", "sparse": "sparse_mask"}
TRAINABLE_OF = {"calibration": "b_dense", "sparse": "b_values"}


def _digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{list(value.shape)}:".encode("ascii")
    return hashlib.sha256(
        header + value.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _layers(state: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    layers = {}
    for key, tensor in state.items():
        name, separator, field = key.rpartition(".")
        if (
            not separator
            or not name
            or field not in {"A", "alpha", "b_dense", "b_values", "active_indices"}
        ):
            raise ValueError(f"Unexpected LoRI state key: {key}")
        if not isinstance(tensor, Tensor):
            raise ValueError(f"LoRI state is not a tensor: {key}")
        layers.setdefault(name, {})[field] = tensor.detach().cpu()
    if not layers:
        raise ValueError("LoRI state contains no layers")
    return dict(sorted(layers.items()))


def _validate(layer: dict[str, Tensor], out_features: int | None = None) -> str:
    if "A" not in layer or "alpha" not in layer:
        raise ValueError("LoRI layer requires A and alpha")
    a, alpha = layer["A"], layer["alpha"]
    if (
        a.ndim != 2
        or min(a.shape) <= 0
        or not a.is_floating_point()
        or not bool(torch.isfinite(a).all())
    ):
        raise ValueError("LoRI A requires finite floating [rank, in] dimensions")
    if (
        alpha.ndim != 0
        or not alpha.is_floating_point()
        or not bool(torch.isfinite(alpha))
    ):
        raise ValueError("LoRI alpha requires a finite floating scalar")
    if set(layer) == {"A", "alpha", "b_dense"}:
        b = layer["b_dense"]
        if (
            b.ndim != 2
            or b.shape[0] <= 0
            or b.shape[1] != a.shape[0]
            or (out_features is not None and b.shape[0] != out_features)
        ):
            raise ValueError("LoRI B requires target output and rank dimensions")
        phase = "calibration"
    elif set(layer) == {"A", "alpha", "b_values", "active_indices"}:
        b, indices = layer["b_values"], layer["active_indices"]
        if (
            b.ndim != 1
            or indices.ndim != 1
            or indices.dtype != torch.int64
            or b.numel() != indices.numel()
        ):
            raise ValueError("LoRI sparse values require matching int64 indices")
        if (
            bool((indices < 0).any())
            or bool((indices[1:] <= indices[:-1]).any())
            or (
                out_features is not None
                and bool((indices >= out_features * a.shape[0]).any())
            )
        ):
            raise ValueError("LoRI indices require sorted unique in-range B offsets")
        phase = "sparse"
    else:
        raise ValueError("LoRI layer has missing or mixed phase state")
    if not b.is_floating_point() or not bool(torch.isfinite(b).all()):
        raise ValueError("LoRI B requires finite floating values")
    return phase


def lori_layer_init(state: dict[str, Tensor], lora_name: str) -> dict[str, Tensor]:
    """Constructor kwargs from a raw layer; the target validates sparse output size."""
    prefix = f"{lora_name}."
    selected = {key: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise KeyError(lora_name)
    layer = _layers(selected)[lora_name]
    phase = _validate(layer)
    result = {
        "a": layer["A"].clone(),
        "alpha": layer["alpha"].clone(),
        "lora_dim": layer["A"].shape[0],
    }
    if phase == "sparse":
        result["active_indices"] = layer["active_indices"].clone()
    return result


def _layer_metadata(layer: dict[str, Tensor], target_shape: list[int]) -> dict:
    rank = layer["A"].shape[0]
    return {
        "target_shape": target_shape,
        "rank": rank,
        "alpha": float(layer["alpha"]),
        "scale": float(layer["alpha"]) / rank,
        "nnz": layer["b_values"].numel(),
        "a_digest": _digest(layer["A"]),
        "alpha_digest": _digest(layer["alpha"]),
        "indices_digest": _digest(layer["active_indices"]),
    }


def a_sha256(state: dict[str, Tensor]) -> str:
    """sha256 of a state's A tensors: fp32 contiguous bytes, layer names sorted."""
    digest = hashlib.sha256()
    for layer in _layers(state).values():
        digest.update(layer["A"].to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def kept_total(state: dict[str, Tensor], phase: str) -> int:
    """Trainable B elements of a state: the kept indices (sparse) or all of B."""
    field = "active_indices" if phase == "sparse" else "b_dense"
    return sum(layer[field].numel() for layer in _layers(state).values())


def _phase(config: AdapterConfig) -> str:
    return config.algo_kwargs.get("phase", "calibration")


def _sparsity(config: AdapterConfig) -> float:
    """The configured mask sparsity (0.0 when ALGO_KWARGS sets none)."""
    return float(config.algo_kwargs.get("sparsity", 0.0))


def export_facts(config: AdapterConfig, state: dict[str, Tensor]) -> dict[str, str]:
    """The LoRI facts of a written file: phase, mechanism, trainable tensor, frozen A,
    its digest and, when sparse, the kept offsets and rank."""
    phase = _phase(config)
    facts = {
        "kohakumoss_lori_phase": phase,
        "kohakumoss_lori_mechanism": MECHANISM_OF[phase],
        "kohakumoss_lori_trainable": TRAINABLE_OF[phase],
        "kohakumoss_lori_a": "frozen_buffer",
        "kohakumoss_lori_sparsity": str(_sparsity(config)),
        "kohakumoss_lori_a_sha256": a_sha256(state),
    }
    if phase == "sparse":
        facts["kohakumoss_lori_kept_total"] = str(kept_total(state, phase))
        facts["kohakumoss_lori_rank"] = str(int(config.dim))
    return facts


def checkpoint_block(config: AdapterConfig, state: dict[str, Tensor]) -> dict:
    """The `lori` block of a checkpoint's state.json."""
    phase = _phase(config)
    return {
        "phase": phase,
        "mechanism": MECHANISM_OF[phase],
        "trainable": TRAINABLE_OF[phase],
        "sparsity": _sparsity(config),
        "kept_total": kept_total(state, phase),
        "a_sha256": a_sha256(state),
    }


def make_sparse_init(
    calibration_state: dict[str, Tensor], sparsity: float
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Select global abs-B quantile ties (0 <= sparsity < 1); reset compact B."""
    if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError("LoRI sparsity must be finite in [0, 1)")
    layers = _layers(calibration_state)
    for layer in layers.values():
        if _validate(layer) != "calibration":
            raise ValueError("Mask calibration requires dense B")
    scores = torch.cat(
        [layer["b_dense"].float().reshape(-1).abs() for layer in layers.values()]
    )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("LoRI calibration scores overflow float32")
    threshold = float(torch.quantile(scores, sparsity))
    state, descriptions = {}, {}
    for name, layer in layers.items():
        b = layer["b_dense"]
        indices = torch.nonzero(
            b.float().reshape(-1).abs() >= threshold, as_tuple=True
        )[0]
        sparse = {
            "A": layer["A"].clone(),
            "alpha": layer["alpha"].clone(),
            "active_indices": indices,
            "b_values": b.new_zeros(indices.numel()),
        }
        state.update({f"{name}.{key}": value for key, value in sparse.items()})
        descriptions[name] = _layer_metadata(sparse, [b.shape[0], layer["A"].shape[1]])
    nnz = sum(layer["nnz"] for layer in descriptions.values())
    info = {
        "calibration_digest": hashlib.sha256(
            json.dumps(
                {
                    key: _digest(value)
                    for key, value in sorted(calibration_state.items())
                },
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "version": VERSION,
        "phase": "sparse_init",
        "sparsity": sparsity,
        "threshold": threshold,
        "total": scores.numel(),
        "nnz": nnz,
        "actual_density": nnz / scores.numel(),
        "layers": descriptions,
    }
    return state, {META_KEY: json.dumps(info, sort_keys=True, separators=(",", ":"))}


def read_lori_state(path: str | Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Read CPU raw state; validate sparse version, shapes, counts and digests."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata() or {}
    validate_lori_state(state, metadata)
    return state, metadata


def validate_lori_state(state: dict[str, Tensor], metadata: dict[str, str]) -> None:
    """Validate already loaded native tensors with the artifact metadata contract."""
    layers = _layers(state)
    phases = {_validate(layer) for layer in layers.values()}
    if len(phases) != 1:
        raise ValueError("LoRI state mixes calibration and sparse layers")
    if META_KEY not in metadata:
        if phases != {"calibration"}:
            raise ValueError("Sparse LoRI state requires versioned shape metadata")
        return
    try:
        info = json.loads(metadata[META_KEY])
        if (
            info["version"] != VERSION
            or info["phase"] not in ("sparse_init", "sparse")
            or phases != {"sparse"}
        ):
            raise ValueError("Unsupported LoRI artifact version or phase")
        if set(info["layers"]) != set(layers):
            raise ValueError("LoRI metadata layer names mismatch")
        digest = info["calibration_digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("Invalid LoRI calibration digest")
        total, nnz = 0, 0
        for name, layer in layers.items():
            description = info["layers"][name]
            shape = description["target_shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(size) is not int or size <= 0 for size in shape)
                or shape[1] != layer["A"].shape[1]
            ):
                raise ValueError("Invalid LoRI target shape")
            _validate(layer, shape[0])
            if info["phase"] == "sparse_init" and bool(
                torch.count_nonzero(layer["b_values"])
            ):
                raise ValueError("LoRI sparse initialization requires zero B values")
            expected = _layer_metadata(layer, shape)
            if description != expected:
                raise ValueError("LoRI layer metadata or digest mismatch")
            total += shape[0] * layer["A"].shape[0]
            nnz += layer["b_values"].numel()
        if (
            info["total"] != total
            or info["nnz"] != nnz
            or info["actual_density"] != nnz / total
        ):
            raise ValueError("LoRI global counts mismatch")
        if (
            not math.isfinite(info["sparsity"])
            or not 0 <= info["sparsity"] < 1
            or not math.isfinite(info["threshold"])
            or info["threshold"] < 0
        ):
            raise ValueError("Invalid LoRI calibration metadata")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed LoRI artifact metadata") from error
