from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from infer.config import (
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SOLVER,
    action_dim_for_config,
    build_policy_from_cfg,
    load_run_config,
    parse_deploy_config,
    policy_config_from_checkpoint_state,
)
from infer.postprocess import apply_action_process
from infer.preprocess import build_obs_from_frames, parse_preprocess_config
from infer.tensor import as_float32_array, default_tactile_norm, numpy_obs_to_torch
from infer.types import InferenceChunk, PreprocessConfig
from tools.normalizer import DatasetNormalizer
from utils.train_utils import cfg_get

_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


class FMInferenceRuntime:
    def __init__(
        self,
        run_dir: Path | str,
        *,
        checkpoint: Path | str | None = None,
        device: str | None = None,
        warmup: bool = True,
    ):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.cfg = load_run_config(self.run_dir)
        self.deploy = parse_deploy_config(self.cfg)

        device_name = device or cfg_get(self.cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else self.run_dir / "checkpoints" / "latest.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path

        ckpt_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy_cfg = policy_config_from_checkpoint_state(ckpt_state, self.cfg)
        self.policy_cfg = policy_cfg
        self.policy = build_policy_from_cfg(policy_cfg, match_training=True).to(self.device)
        self.normalizer = DatasetNormalizer.load_state_dict(ckpt_state["normalizer_state_dict"])
        self.policy.load_state_dict(ckpt_state["policy_state_dict"])
        self.policy.eval()
        self.use_tactile = bool(self.policy.use_tactile)
        if self.use_tactile and self.normalizer.tactile is None:
            raise RuntimeError(
                "checkpoint use_tactile=true but normalizer_state_dict has no tactile stats"
            )

        data_cfg = self.policy_cfg["data"]
        fm_cfg = self.policy_cfg["models"]["fm"]
        self.window_size = int(data_cfg["window_size"])
        self.action_horizon = int(fm_cfg["action_horizon"])
        self.n_image_views = int(fm_cfg.get("n_image_views", 3))
        self.num_inference_steps = int(fm_cfg.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
        self.solver = str(fm_cfg.get("solver", DEFAULT_SOLVER))

        if warmup:
            self._warmup()

    @property
    def action_dim(self) -> int:
        return action_dim_for_config(self.policy_cfg)

    def _warmup(self) -> None:
        image_size = int(cfg_get(self.policy_cfg, "data.image_size", 224))
        n_views = self.n_image_views
        dummy_obs = {
            "image": np.zeros((1, 1, n_views, 3, image_size, image_size), dtype=np.uint8),
            "state": np.zeros((self.window_size, self.action_dim), dtype=np.float32),
        }
        if self.use_tactile:
            dummy_obs["tactile"] = default_tactile_norm(
                self.normalizer,
                self.window_size,
            )
        obs_torch = numpy_obs_to_torch(
            dummy_obs,
            self.device,
            use_tactile=self.use_tactile,
            normalizer=self.normalizer,
            window_size=self.window_size,
        )
        with torch.inference_mode():
            self.policy.predict_action(
                obs_torch,
                num_inference_steps=min(2, self.num_inference_steps),
                solver=self.solver,
            )

    @torch.inference_mode()
    def predict_rot6d_abs(
        self,
        obs: Mapping[str, Any],
        *,
        state_raw: np.ndarray,
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> np.ndarray:
        """Realtime deployment entry: obs(image+state) -> absolute rot6d chunk (T, D)."""
        state_raw = as_float32_array(state_raw, name="state_raw")
        if state_raw.shape != (self.window_size, self.action_dim):
            raise ValueError(
                f"state_raw shape {state_raw.shape} != ({self.window_size}, {self.action_dim})"
            )

        obs_torch = numpy_obs_to_torch(
            obs,
            self.device,
            use_tactile=self.use_tactile,
            normalizer=self.normalizer,
            window_size=self.window_size,
        )
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        solver_name = self.solver if solver is None else str(solver)

        result = self.policy.predict_action(
            obs_torch,
            num_inference_steps=steps,
            solver=solver_name,
        )
        pred_norm = result["action_pred_normalized"].detach().cpu().numpy()
        if pred_norm.ndim == 3:
            pred_norm = pred_norm[0]
        pred_abs = self.normalizer.unnormalize_action_np(pred_norm, state_raw)
        return as_float32_array(pred_abs, name="pred_abs")

    @torch.inference_mode()
    def predict_rot6d_abs_batch(
        self,
        obs_list: Sequence[Mapping[str, Any]],
        state_raw_list: Sequence[np.ndarray],
        *,
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> np.ndarray:
        if len(obs_list) != len(state_raw_list):
            raise ValueError("obs_list and state_raw_list must have the same length")
        if not obs_list:
            raise ValueError("obs_list must be non-empty")

        preds = [
            self.predict_rot6d_abs(
                obs,
                state_raw=state_raw,
                num_inference_steps=num_inference_steps,
                solver=solver,
            )
            for obs, state_raw in zip(obs_list, state_raw_list)
        ]
        return np.stack(preds, axis=0).astype(np.float32, copy=False)

    def benchmark_single(self, obs: Mapping[str, Any], *, state_raw: np.ndarray, repeats: int = 3) -> dict[str, float]:
        """Return wall-clock ms for a single predict (after construction warmup)."""
        repeats = max(1, int(repeats))
        start = time.perf_counter()
        for _ in range(repeats):
            self.predict_rot6d_abs(obs, state_raw=state_raw)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
        return {"infer_ms": float(elapsed_ms), "repeats": float(repeats)}

    def infer_from_window(
        self,
        frames: Sequence[Any],
        *,
        preprocess: PreprocessConfig | None = None,
        robot: Mapping[str, Any] | None = None,
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> InferenceChunk:
        """ROS window -> preprocess -> predict -> deploy action_process -> InferenceChunk."""
        preprocess_cfg = preprocess or parse_preprocess_config(self.cfg, robot=robot)
        if len(preprocess_cfg.camera_views) != self.n_image_views:
            raise ValueError(
                "preprocess camera_views count does not match model n_image_views: "
                f"{len(preprocess_cfg.camera_views)} != {self.n_image_views}. "
                "Check run_dir resolved_config (data.camera_views / models.fm.n_image_views)."
            )
        if bool(self.use_tactile) != bool(preprocess_cfg.use_tactile):
            raise ValueError(
                "runtime.use_tactile does not match resolved_config data.use_tactile: "
                f"{self.use_tactile} != {preprocess_cfg.use_tactile}"
            )
        obs, state_raw = build_obs_from_frames(
            frames,
            preprocess_cfg,
            self.normalizer,
            window_size=self.window_size,
        )
        pred_rot6d = self.predict_rot6d_abs(
            obs,
            state_raw=state_raw,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        actions = apply_action_process(pred_rot6d, self.deploy.action_process)
        expected = (self.action_horizon, 14)
        if actions.shape != expected:
            raise ValueError(f"postprocess shape {actions.shape} != expected {expected}")
        return InferenceChunk(
            actions=actions,
            action_space=self.deploy.action_process,
            hz=self.deploy.action_hz,
            metadata={
                "action_horizon": self.action_horizon,
                "window_size": self.window_size,
            },
        )


def random_smoke_obs(
    runtime: FMInferenceRuntime,
    *,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Finite random obs for __main__ smoke (no zarr). rot6d uses identity columns."""
    rng = np.random.default_rng(int(seed))
    window_size = runtime.window_size
    action_dim = runtime.action_dim
    image_size = int(cfg_get(runtime.policy_cfg, "data.image_size", 224))
    n_views = runtime.n_image_views

    state_raw = np.zeros((window_size, action_dim), dtype=np.float32)
    if action_dim == 14:
        for t in range(window_size):
            state_raw[t] = rng.normal(0.0, 0.05, size=14).astype(np.float32)
            state_raw[t, [6, 13]] = rng.uniform(0.0, 0.05, size=2).astype(np.float32)
    else:
        for t in range(window_size):
            for arm in range(2):
                base = arm * (action_dim // 2)
                state_raw[t, base : base + 3] = rng.normal(0.0, 0.05, size=3).astype(np.float32)
                state_raw[t, base + 3 : base + 9] = _IDENTITY_ROT6D
                state_raw[t, base + 9] = float(rng.uniform(0.0, 0.05))

    state_norm = runtime.normalizer.normalize_state_np(state_raw)
    image = rng.integers(0, 256, size=(1, 1, n_views, 3, image_size, image_size), dtype=np.uint8)
    obs = {"image": image, "state": state_norm.astype(np.float32, copy=False)}
    if runtime.use_tactile:
        obs["tactile"] = default_tactile_norm(
            runtime.normalizer,
            runtime.window_size,
        )
    return obs, state_raw


if __name__ == "__main__":
    run_dir = Path("/mnt/workspace/zyj/deploy/PrometheusV2/third_party/flow_matching/outputs/chahua_eef_vision")
    runtime = FMInferenceRuntime(run_dir, warmup=True)
    obs, state_raw = random_smoke_obs(runtime, seed=42)
    out = runtime.predict_rot6d_abs(obs, state_raw=state_raw)
    print(f"action_dim={runtime.action_dim}, output shape={out.shape}")