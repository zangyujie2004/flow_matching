from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .tactile_token import ResidualBlock2D, TactileResidualTokenEncoder


class TactileResidualDecoder(nn.Module):
    """Decode one 16-D sensor token into a normalized (3, 35, 20) map."""

    def __init__(
        self,
        *,
        token_dim: int = 16,
        hidden_dims: tuple[int, int, int] = (32, 64, 128),
        spatial_shape: tuple[int, int] = (35, 20),
        n_groups: int = 8,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if tuple(spatial_shape) != (35, 20):
            raise ValueError(
                "the transposed-convolution decoder currently requires "
                f"spatial_shape=(35, 20), got {spatial_shape}"
            )
        if len(hidden_dims) != 3:
            raise ValueError(f"hidden_dims must contain three values, got {hidden_dims}")
        dim0, dim1, dim2 = (int(dim) for dim in hidden_dims)
        for dim in (dim0, dim1, dim2):
            if dim <= 0 or dim % int(n_groups) != 0:
                raise ValueError(
                    f"hidden channel dim={dim} must be positive and divisible by "
                    f"n_groups={n_groups}"
                )

        activation = str(output_activation).strip().lower()
        if activation not in {"identity", "none", "tanh"}:
            raise ValueError(
                "output_activation must be one of identity, none, tanh; "
                f"got {output_activation!r}"
            )

        self.token_dim = int(token_dim)
        self.spatial_shape = (35, 20)
        self.project = nn.Linear(self.token_dim, dim2 * 9 * 5)
        self.decoder = nn.Sequential(
            ResidualBlock2D(dim2, n_groups=n_groups),
            nn.ConvTranspose2d(
                dim2,
                dim1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GroupNorm(n_groups, dim1),
            nn.SiLU(),
            ResidualBlock2D(dim1, n_groups=n_groups),
            nn.ConvTranspose2d(
                dim1,
                dim0,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1),
            ),
            nn.GroupNorm(n_groups, dim0),
            nn.SiLU(),
            ResidualBlock2D(dim0, n_groups=n_groups),
            nn.Conv2d(dim0, 3, kernel_size=3, padding=1),
        )
        self.output_activation = nn.Tanh() if activation == "tanh" else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected tokens (N,{self.token_dim}), got {tuple(tokens.shape)}"
            )
        x = self.project(tokens).reshape(tokens.shape[0], -1, 9, 5)
        reconstruction = self.output_activation(self.decoder(x))
        expected = (tokens.shape[0], 3, *self.spatial_shape)
        if tuple(reconstruction.shape) != expected:
            raise RuntimeError(
                f"decoder shape {tuple(reconstruction.shape)} != expected {expected}"
            )
        return reconstruction


class TactileAutoencoder(nn.Module):
    """
    Shared-weight autoencoder for four tactile sensors.

    Canonical layouts:
      sensor tokens: (B, 4, 16)
      token grid:    (B, 16, 4)
      flattened:     (B, 64)
    """

    def __init__(
        self,
        *,
        num_sensors: int = 4,
        channels_per_sensor: int = 3,
        token_dim: int = 16,
        hidden_dims: tuple[int, int, int] = (32, 64, 128),
        spatial_shape: tuple[int, int] = (35, 20),
        n_groups: int = 8,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.channels_per_sensor = int(channels_per_sensor)
        self.token_dim = int(token_dim)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.latent_dim = self.num_sensors * self.token_dim

        self.encoder = TactileResidualTokenEncoder(
            num_sensors=self.num_sensors,
            channels_per_sensor=self.channels_per_sensor,
            token_dim=self.token_dim,
            hidden_dims=hidden_dims,
            spatial_shape=self.spatial_shape,
            n_groups=n_groups,
        )
        self.decoder = TactileResidualDecoder(
            token_dim=self.token_dim,
            hidden_dims=hidden_dims,
            spatial_shape=self.spatial_shape,
            n_groups=n_groups,
            output_activation=output_activation,
        )

    @property
    def input_channels(self) -> int:
        return self.num_sensors * self.channels_per_sensor

    def sensor_tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_sensors,
            self.token_dim,
        ):
            raise ValueError(
                "expected sensor tokens "
                f"(B,{self.num_sensors},{self.token_dim}), got {tuple(tokens.shape)}"
            )
        return tokens.transpose(1, 2).contiguous()

    def grid_to_sensor_tokens(self, token_grid: torch.Tensor) -> torch.Tensor:
        if token_grid.ndim != 3 or tuple(token_grid.shape[1:]) != (
            self.token_dim,
            self.num_sensors,
        ):
            raise ValueError(
                f"expected token grid (B,{self.token_dim},{self.num_sensors}), "
                f"got {tuple(token_grid.shape)}"
            )
        return token_grid.transpose(1, 2).contiguous()

    def flatten_sensor_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.sensor_tokens_to_grid(tokens).flatten(start_dim=1)

    def unflatten_sensor_tokens(self, flattened: torch.Tensor) -> torch.Tensor:
        if flattened.ndim != 2 or flattened.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected flattened tokens (B,{self.latent_dim}), "
                f"got {tuple(flattened.shape)}"
            )
        grid = flattened.reshape(
            flattened.shape[0],
            self.token_dim,
            self.num_sensors,
        )
        return self.grid_to_sensor_tokens(grid)

    def encode_sensor_tokens(self, tactile: torch.Tensor) -> torch.Tensor:
        if tactile.ndim == 4:
            tactile = tactile.unsqueeze(1)
        return self.encoder.encode_sensor_tokens(tactile)

    def encode_flattened(self, tactile: torch.Tensor) -> torch.Tensor:
        return self.flatten_sensor_tokens(self.encode_sensor_tokens(tactile))

    def decode_sensor_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_sensors,
            self.token_dim,
        ):
            raise ValueError(
                "expected sensor tokens "
                f"(B,{self.num_sensors},{self.token_dim}), got {tuple(tokens.shape)}"
            )
        batch = tokens.shape[0]
        decoded = self.decoder(tokens.reshape(batch * self.num_sensors, self.token_dim))
        return decoded.reshape(
            batch,
            self.num_sensors,
            self.channels_per_sensor,
            *self.spatial_shape,
        )

    def sensors_to_channel_last(self, tactile: torch.Tensor) -> torch.Tensor:
        expected_tail = (
            self.num_sensors,
            self.channels_per_sensor,
            *self.spatial_shape,
        )
        if tactile.ndim != 5 or tuple(tactile.shape[1:]) != expected_tail:
            raise ValueError(
                f"expected tactile (B,{','.join(map(str, expected_tail))}), "
                f"got {tuple(tactile.shape)}"
            )
        batch = tactile.shape[0]
        height, width = self.spatial_shape
        return (
            tactile.permute(0, 3, 4, 1, 2)
            .contiguous()
            .reshape(batch, height, width, self.input_channels)
        )

    def decode_flattened(self, flattened: torch.Tensor) -> torch.Tensor:
        sensor_maps = self.decode_sensor_tokens(
            self.unflatten_sensor_tokens(flattened)
        )
        return self.sensors_to_channel_last(sensor_maps)

    def forward(self, tactile: torch.Tensor) -> dict[str, torch.Tensor]:
        if tactile.ndim == 5:
            if tactile.shape[1] != 1:
                raise ValueError(
                    "TactileAutoencoder.forward expects one frame; "
                    f"got time={tactile.shape[1]}"
                )
            tactile = tactile[:, 0]
        tokens = self.encode_sensor_tokens(tactile)
        flattened = self.flatten_sensor_tokens(tokens)
        reconstruction = self.decode_flattened(flattened)
        return {
            "reconstruction": reconstruction,
            "sensor_tokens": tokens,
            "token_grid": self.sensor_tokens_to_grid(tokens),
            "latent": flattened,
        }

    def config_dict(self) -> dict[str, Any]:
        encoder = self.encoder
        hidden_dims = (
            int(encoder.shared_encoder[0].out_channels),
            int(encoder.shared_encoder[4].out_channels),
            int(encoder.shared_encoder[8].out_channels),
        )
        return {
            "num_sensors": self.num_sensors,
            "channels_per_sensor": self.channels_per_sensor,
            "token_dim": self.token_dim,
            "hidden_dims": list(hidden_dims),
            "spatial_shape": list(self.spatial_shape),
            "n_groups": int(encoder.shared_encoder[1].num_groups),
            "output_activation": (
                "tanh" if isinstance(self.decoder.output_activation, nn.Tanh) else "identity"
            ),
        }


def build_tactile_autoencoder(config: Mapping[str, Any] | None = None) -> TactileAutoencoder:
    cfg = dict(config or {})
    if "hidden_dims" in cfg:
        cfg["hidden_dims"] = tuple(int(value) for value in cfg["hidden_dims"])
    if "spatial_shape" in cfg:
        cfg["spatial_shape"] = tuple(int(value) for value in cfg["spatial_shape"])
    return TactileAutoencoder(**cfg)


def load_tactile_autoencoder_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TactileAutoencoder, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"tactile AE checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model_config = state.get("model_config")
    if not isinstance(model_config, Mapping):
        raise KeyError(
            f"tactile AE checkpoint has no model_config mapping: {checkpoint_path}"
        )
    model = build_tactile_autoencoder(model_config)
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise KeyError(
            f"tactile AE checkpoint has no model_state_dict: {checkpoint_path}"
        )
    model.load_state_dict(model_state, strict=True)
    return model, state
