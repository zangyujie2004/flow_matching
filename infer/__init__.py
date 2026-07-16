"""Flow Matching deployment inference package."""

from __future__ import annotations

import sys
from pathlib import Path

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[1]
if str(_FLOW_MATCHING_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOW_MATCHING_ROOT))

from infer.config import (
    ACTION_DIMS,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SOLVER,
    DEFAULT_VELOCITY_MODEL,
    DeployConfig,
    action_dim_for_config,
    build_policy_from_cfg,
    infer_velocity_model_from_state_dict,
    load_run_config,
    load_runtime_checkpoint,
    parse_deploy_config,
    policy_config_from_checkpoint_state,
    resolve_fm_cfg_for_inference,
)
from infer.postprocess import apply_action_process, eef_rot6d_abs_to_rpy_abs, rot6d_to_rpy
from infer.preprocess import build_obs_from_frames, parse_preprocess_config
from infer.runtime import FMInferenceRuntime, random_smoke_obs
from infer.tensor import numpy_obs_to_torch
from infer.types import InferenceChunk, PreprocessConfig
from infer.zarr_bridge import obs_from_zarr_window

__all__ = [
    "ACTION_DIMS",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_SOLVER",
    "DEFAULT_VELOCITY_MODEL",
    "DeployConfig",
    "FMInferenceRuntime",
    "InferenceChunk",
    "PreprocessConfig",
    "action_dim_for_config",
    "apply_action_process",
    "build_obs_from_frames",
    "build_policy_from_cfg",
    "eef_rot6d_abs_to_rpy_abs",
    "infer_velocity_model_from_state_dict",
    "load_run_config",
    "load_runtime_checkpoint",
    "numpy_obs_to_torch",
    "obs_from_zarr_window",
    "parse_deploy_config",
    "parse_preprocess_config",
    "policy_config_from_checkpoint_state",
    "random_smoke_obs",
    "resolve_fm_cfg_for_inference",
    "rot6d_to_rpy",
]
