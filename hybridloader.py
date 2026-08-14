"""H3 FL2VA/Ref2VA hybrid checkpoint loader.

SPDX-License-Identifier: GPL-3.0-or-later

Loads a MiniMax H3 FL2VA checkpoint and optionally folds a validated set of
Ref2VA tensors over it while staying on ComfyUI's file-backed
DynamicVRAM/AIMDO loading path. With the ``none`` preset only the FL2VA
checkpoint is opened, so memory and runtime match a plain single-model load.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager

import torch
from safetensors import safe_open

import comfy.memory_management
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths


SUPPORTED_PRESETS = ("none", "block_range_adaln", "all_adaln")
DEFAULT_PRESET = "block_range_adaln"
DEFAULT_BLOCK_RANGE = (45, 49)

_BLOCK_ADALN = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.(?:weight|bias)$")
_FINAL_ADALN = {
    "final_layer.adaln_proj.linear.weight",
    "final_layer.adaln_proj.linear.bias",
}
_TIME_KEYS = {
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
}
_QUANT_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
)


@contextmanager
def _open_H3_safetensors(path: str):
    """Expose one checkpoint as a tensor source plus its keys and metadata."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"H3HybridLoader: checkpoint not found: {path}")
    if comfy.memory_management.aimdo_enabled:
        state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        yield state_dict, set(state_dict), metadata or {}
        return
    with safe_open(path, framework="pt", device="cpu") as source:
        yield source, set(source.keys()), source.metadata() or {}


def _read_tensor(source, key: str) -> torch.Tensor:
    if isinstance(source, dict):
        return source[key]
    return source.get_tensor(key)


def _quant_parent(key: str) -> str | None:
    for suffix in _QUANT_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _checkpoint_format(keys: set[str], path: str = "") -> str:
    has_table = "adaln_t_table" in keys
    has_time = bool(keys & _TIME_KEYS)
    if has_table == has_time:
        name = os.path.basename(path) if path else "checkpoint"
        raise RuntimeError(
            f"H3HybridLoader: cannot identify pruned/full layout in {name} "
            "(expected exactly one of adaln_t_table / time_embedder.*)"
        )
    return "pruned" if has_table else "full"


def _in_block_range(key: str, start: int | None = None, end: int | None = None) -> bool:
    match = _BLOCK_ADALN.match(key)
    if match is None:
        return False
    block = int(match.group(1))
    return start is None or start <= block <= end


def _overlay_root(key: str, preset: str, checkpoint_format: str) -> bool:
    if preset == "none":
        return False
    if preset == "block_range_adaln":
        start, end = DEFAULT_BLOCK_RANGE
        return _in_block_range(key, start, end)
    if preset != "all_adaln":
        raise ValueError(f"H3HybridLoader: unsupported preset {preset!r}")
    if _in_block_range(key) or key in _FINAL_ADALN:
        return True
    if checkpoint_format == "pruned":
        return key == "adaln_t_table"
    return key in _TIME_KEYS


def _take_from_overlay(key: str, preset: str, checkpoint_format: str) -> bool:
    if _overlay_root(key, preset, checkpoint_format):
        return True
    parent = _quant_parent(key)
    if parent is None:
        return False
    return _overlay_root(parent + ".weight", preset, checkpoint_format)


def _check_compatible_key_sets(base_keys: set[str], overlay_keys: set[str]) -> None:
    base_only = base_keys - overlay_keys
    overlay_only = overlay_keys - base_keys
    bad_base = sorted(key for key in base_only if _quant_parent(key) is None)
    bad_overlay = sorted(key for key in overlay_only if _quant_parent(key) is None)
    if bad_base or bad_overlay:
        raise RuntimeError(
            "H3HybridLoader: base and overlay checkpoints have different weight "
            f"key sets; base-only={bad_base[:5]} overlay-only={bad_overlay[:5]}"
        )


def _metadata_conflicts(base: dict, overlay: dict) -> list[str]:
    conflicts = []
    for key in ("format", "modelspec.architecture", "model_type"):
        if key in base and key in overlay and base[key] != overlay[key]:
            conflicts.append(key)
    return conflicts


def _read_single_checkpoint(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    with _open_H3_safetensors(path) as (source, keys, metadata):
        return {key: _read_tensor(source, key) for key in sorted(keys)}, metadata


def merge_overlay_state_dict(
    base_path: str,
    overlay_path: str | None,
    preset: str = DEFAULT_PRESET,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Merge overlay tensors into the base state dict and return it with metadata."""
    if preset not in SUPPORTED_PRESETS:
        raise ValueError(
            f"H3HybridLoader: unsupported preset {preset!r}; "
            f"choose one of {', '.join(SUPPORTED_PRESETS)}"
        )
    if preset == "none" or overlay_path is None:
        return _read_single_checkpoint(base_path)

    with _open_H3_safetensors(base_path) as (base_source, base_keys, metadata):
        with _open_H3_safetensors(overlay_path) as (overlay_source, overlay_keys, overlay_metadata):
            _check_compatible_key_sets(base_keys, overlay_keys)
            base_format = _checkpoint_format(base_keys, base_path)
            overlay_format = _checkpoint_format(overlay_keys, overlay_path)
            if base_format != overlay_format:
                raise RuntimeError(
                    "H3HybridLoader: base and overlay use different checkpoint "
                    f"layouts (base={base_format}, overlay={overlay_format})"
                )
            conflicts = _metadata_conflicts(metadata, overlay_metadata)
            if conflicts:
                raise RuntimeError(
                    "H3HybridLoader: checkpoint metadata disagrees on "
                    + ", ".join(conflicts)
                )

            overlay_args = (preset, base_format)
            keys = set(base_keys)
            for key in overlay_keys - base_keys:
                parent = _quant_parent(key)
                if parent is not None and _overlay_root(parent + ".weight", *overlay_args):
                    keys.add(key)

            state_dict: dict[str, torch.Tensor] = {}
            for key in sorted(keys):
                take_overlay = _take_from_overlay(key, *overlay_args)
                source = overlay_source if take_overlay else base_source
                source_keys = overlay_keys if take_overlay else base_keys
                if key not in source_keys:
                    raise RuntimeError(
                        "H3HybridLoader: quant metadata for a selected weight is "
                        f"missing from its checkpoint: {key}"
                    )
                value = _read_tensor(source, key)
                if take_overlay and key in base_keys:
                    base_value = _read_tensor(base_source, key)
                    if value.shape != base_value.shape or value.dtype != base_value.dtype:
                        raise RuntimeError(
                            "H3HybridLoader: selected overlay tensor mismatches base "
                            f"for {key}: base={tuple(base_value.shape)}/{base_value.dtype}, "
                            f"overlay={tuple(value.shape)}/{value.dtype}"
                        )
                state_dict[key] = value
            return state_dict, metadata


def load_hybrid_h3_model(
    base_path: str,
    overlay_path: str | None = None,
    preset: str = DEFAULT_PRESET,
    *,
    disable_dynamic: bool = False,
):
    """Load the merged checkpoint as a ComfyUI MODEL."""
    comfy.model_management.free_pins(1e32, evict_active=True, loaded=True)
    comfy.model_management.unload_all_models()

    if overlay_path is None or preset == "none":
        state_dict, metadata = _read_single_checkpoint(base_path)
    else:
        if not comfy.memory_management.aimdo_enabled:
            available_ram = comfy.model_management.psutil.virtual_memory().available
            required_ram = os.path.getsize(base_path) + 1 * 1024 ** 3
            if available_ram < required_ram:
                available_gb = available_ram / 1024 ** 3
                required_gb = required_ram / 1024 ** 3
                raise RuntimeError(
                    "H3HybridLoader: not enough system RAM for this checkpoint "
                    f"(available={available_gb:.1f} GiB, required~={required_gb:.1f} GiB). "
                    "Use the *_pruned_int8_convrot.safetensors pair or enable DynamicVRAM."
                )
        state_dict, metadata = merge_overlay_state_dict(base_path, overlay_path, preset)

    model_patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, metadata=metadata, disable_dynamic=disable_dynamic
    )
    if model_patcher is None:
        raise RuntimeError(
            "H3HybridLoader: ComfyUI could not detect the composed MiniMax H3 checkpoint"
        )
    # Lets multigpu deepclone and non-dynamic delegates rebuild the model from disk.
    model_patcher.cached_patcher_init = (
        load_hybrid_h3_model,
        (base_path, overlay_path, preset),
    )
    return model_patcher


def _list_diffusion_models() -> list[str]:
    return folder_paths.get_filename_list("diffusion_models")


class MinimaxH3HybridLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_diffusion_models()
        base_files = [name for name in files if "fl2va" in name.lower()]
        overlay_files = [name for name in files if "ref2va" in name.lower()]
        base_default = next((name for name in base_files if "pruned" in name.lower()), None)
        overlay_default = next((name for name in overlay_files if "pruned" in name.lower()), None)
        base_options = {"tooltip": "FL2VA checkpoint used as the base model."}
        overlay_options = {"tooltip": "Optional Ref2VA checkpoint. Ignored in none mode."}
        if base_default is not None:
            base_options["default"] = base_default
        if overlay_default is not None:
            overlay_options["default"] = overlay_default
        return {
            "required": {
                "base_model": (base_files, base_options),
                "overlay_model": ([""] + overlay_files, overlay_options),
                "overlay_preset": (list(SUPPORTED_PRESETS), {
                    "default": DEFAULT_PRESET,
                    "tooltip": "none, block_range_adaln (45-49), or all_adaln",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("MiniMax H3 model loaded with the selected FL2VA/Ref2VA composition.",)
    OUTPUT_NODE = True
    FUNCTION = "load_recommended"
    CATEGORY = "ANe5s Nodes"
    DESCRIPTION = "Load MiniMax H3 FL2VA with a validated Ref2VA tensor composition."

    def load_recommended(self, base_model, overlay_model=None, overlay_preset=DEFAULT_PRESET):
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
        overlay_path = None
        if overlay_preset != "none":
            if not overlay_model:
                raise ValueError("H3HybridLoader: overlay_model is required for this preset")
            overlay_path = folder_paths.get_full_path_or_raise("diffusion_models", overlay_model)
        logging.info(
            "[H3HybridLoader] base=%s overlay=%s preset=%s",
            os.path.basename(base_path),
            os.path.basename(overlay_path) if overlay_path else "ignored",
            overlay_preset,
        )
        return (load_hybrid_h3_model(base_path, overlay_path, overlay_preset),)
