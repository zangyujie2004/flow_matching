
# build data loader
-> dataset, train_loader # 无 val loader

# build policy
-> build_flow_policy(cfg, ds)

# train_one_epoch
-> flow_matching_loss + TensorBoard (Step/*, Epoch/*)
-> 可选 train.max_train_batches 用于 smoke/debug

# eval_open_loop
rng = np.random.default_rng(seed + epoch)
随机抽 max_batches 个 batch
GT absolute: ds.get_action(a0, a1)
pred absolute: normalizer.unnormalize_action_np(pred_norm, state_raw)
state_raw: meta.idx -> state_range -> get_state
-> action_l1 / action_mse（绝对空间）
-> 折线图 outputs/{run}/open_loop/plots/

# checkpoint save
每 epoch: checkpoints/latest.pt
save_every: checkpoints/epoch_XXXX.pt
{
  epoch, global_step,
  policy_state_dict,
  normalizer_state_dict,
  optimizer_state_dict,
  config
}
不存 best.pt

# 入口（参数均在 config yaml 中配置）
python train.py --config configs/config.yaml
./scripts/train.sh

# DINO 特征预计算（训练前跑一次）
./scripts/precompute.sh --config configs/config.yaml
# 然后 config 里设 data.use_camera_latent: true

./scripts/train.sh --smoke
