# Peel FM experiment configs

Pairing rule: `action_horizon=32` → `window_size=8`; `action_horizon=64` → `window_size=16`.

Latent cache keyed by `(window_size, action_horizon)` only (3-view cache, 2-view slices).

| id | ready | config | task | backbone | window | chunk | cam | pos | latent_cache |
|----|-------|--------|------|----------|--------|-------|-----|-----|--------------|
| 01 | yes | `configs/train/peel/01_peel_all_unet_h8_c32_v2_abs.yaml` | peel_all | unet | 8 | 32 | 2 | absolute | `dinov2_s14` |
| 02 | precompute | `configs/train/peel/02_peel_all_unet_h16_c64_v2_rel.yaml` | peel_all | unet | 16 | 64 | 2 | relative | `dinov2_s14_w16_ah64` |
| 03 | yes | `configs/train/peel/03_peel_all_dit_h8_c32_v2_abs.yaml` | peel_all | dit | 8 | 32 | 2 | absolute | `dinov2_s14` |
| 04 | precompute | `configs/train/peel/04_peel_all_dit_h16_c64_v3_rel.yaml` | peel_all | dit | 16 | 64 | 3 | relative | `dinov2_s14_w16_ah64` |
| 05 | yes | `configs/train/peel/05_peel_new_unet_h8_c32_v2_abs.yaml` | peel_new | unet | 8 | 32 | 2 | absolute | `dinov2_s14` |
| 06 | precompute | `configs/train/peel/06_peel_new_unet_h16_c64_v2_rel.yaml` | peel_new | unet | 16 | 64 | 2 | relative | `dinov2_s14_w16_ah64` |
| 07 | yes | `configs/train/peel/07_peel_new_dit_h8_c32_v2_abs.yaml` | peel_new | dit | 8 | 32 | 2 | absolute | `dinov2_s14` |
| 08 | precompute | `configs/train/peel/08_peel_new_dit_h16_c64_v3_rel.yaml` | peel_new | dit | 16 | 64 | 3 | relative | `dinov2_s14_w16_ah64` |

## Ready now (existing `dinov2_s14` = w8/ah32)

```bash
./scripts/train.sh --config configs/train/peel/01_peel_all_unet_h8_c32_v2_abs.yaml --gpus 0
./scripts/train.sh --config configs/train/peel/05_peel_new_unet_h8_c32_v2_abs.yaml --gpus 0
```

## Need precompute (`dinov2_s14_w16_ah64`)

```bash
./scripts/precompute.sh --config configs/train/peel/02_peel_all_unet_h16_c64_v2_rel.yaml
./scripts/precompute.sh --config configs/train/peel/06_peel_new_unet_h16_c64_v2_rel.yaml
```
