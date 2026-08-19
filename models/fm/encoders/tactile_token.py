from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    """Shape-preserving Conv2D residual block."""

    def __init__(self, channels: int, *, n_groups: int = 8) -> None:
        super().__init__()
        if channels % n_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by n_groups={n_groups}"
            )
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TactileResidualTokenEncoder(nn.Module):
    """
    Encode four tactile sensors independently with one shared residual CNN.

    Input:
        tactile: (B, T, H, W, num_sensors * channels_per_sensor)

    Only the latest tactile frame is encoded. Channel order is sensor-major:
        [sensor_0(dx,dy,dz), ..., sensor_3(dx,dy,dz)]

    Output:
        flattened tokens: (B, token_dim * num_sensors)

    The explicit intermediate layout is:
        (B, num_sensors, token_dim)
        -> (B, token_dim, 1, num_sensors)
        -> (B, token_dim * num_sensors)
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
    ) -> None:
        super().__init__()
        if num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {num_sensors}")
        if channels_per_sensor <= 0:
            raise ValueError(
                "channels_per_sensor must be positive, "
                f"got {channels_per_sensor}"
            )
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if len(hidden_dims) != 3 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError(
                f"hidden_dims must contain three positive values, got {hidden_dims}"
            )

        self.num_sensors = int(num_sensors)
        self.channels_per_sensor = int(channels_per_sensor)
        self.token_dim = int(token_dim)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.output_dim = self.num_sensors * self.token_dim

        dim0, dim1, dim2 = (int(dim) for dim in hidden_dims)
        for dim in (dim0, dim1, dim2):
            if dim % n_groups != 0:
                raise ValueError(
                    f"hidden channel dim={dim} must be divisible by "
                    f"n_groups={n_groups}"
                )

        self.shared_encoder = nn.Sequential(
            nn.Conv2d(
                self.channels_per_sensor,
                dim0,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(n_groups, dim0),
            nn.SiLU(),
            ResidualBlock2D(dim0, n_groups=n_groups),
            nn.Conv2d(dim0, dim1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(n_groups, dim1),
            nn.SiLU(),
            ResidualBlock2D(dim1, n_groups=n_groups),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(n_groups, dim2),
            nn.SiLU(),
            ResidualBlock2D(dim2, n_groups=n_groups),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim2, self.token_dim),
        )

    @property
    def input_channels(self) -> int:
        return self.num_sensors * self.channels_per_sensor

    def encode_sensor_tokens(self, tactile: torch.Tensor) -> torch.Tensor:
        """Return sensor-major tokens with shape (B, num_sensors, token_dim)."""

        if tactile.ndim != 5:
            raise ValueError(
                "expected tactile (B,T,H,W,C), "
                f"got shape={tuple(tactile.shape)}"
            )
        batch, time, height, width, channels = tactile.shape
        if time < 1:
            raise ValueError("tactile time dimension must contain at least one frame")
        if (height, width) != self.spatial_shape:
            raise ValueError(
                f"expected tactile spatial shape {self.spatial_shape}, "
                f"got {(height, width)}"
            )
        if channels != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} tactile channels "
                f"({self.num_sensors} sensors x "
                f"{self.channels_per_sensor} channels), got {channels}"
            )

        latest = tactile[:, -1]
        per_sensor = (
            latest.reshape(
                batch,
                height,
                width,
                self.num_sensors,
                self.channels_per_sensor,
            )
            .permute(0, 3, 4, 1, 2)
            .contiguous()
            .reshape(
                batch * self.num_sensors,
                self.channels_per_sensor,
                height,
                width,
            )
        )
        tokens = self.shared_encoder(per_sensor)
        return tokens.reshape(batch, self.num_sensors, self.token_dim)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        sensor_tokens = self.encode_sensor_tokens(tactile)
        token_layout = sensor_tokens.transpose(1, 2).unsqueeze(2)
        return token_layout.flatten(start_dim=1)
