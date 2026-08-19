import matplotlib
# 若是服务器/无 GUI 环境，解开下一行更稳妥：
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import os
import cv2


def plot_tactile_grids_animation_v2(pred, gt, save_path, plt_gt=True, plt_pred=True,fps=8):
    """
    pred, gt: np.ndarray, shape = (T, 35, 20, 6)
      通道含义假设：
        左手: 0->Ux_L, 1->Uy_L
        右手: 3->Ux_R, 4->Uy_R
    """
    assert isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray), "pred/gt 必须是 np.ndarray"
    assert pred.ndim == 4 and gt.ndim == 4, f"expected (T,35,20,6), got {pred.shape} / {gt.shape}"
    assert pred.shape == gt.shape and pred.shape[1:] == (35,20,6), f"expected (T,35,20,6), got {pred.shape}"
    assert plt_gt or plt_pred
    T = pred.shape[0]

    # 坐标网格
    x = np.linspace(-8.5, 8.5, 20)
    y = np.linspace(30, 0, 35)
    X, Y = np.meshgrid(x, y)

    color_gt   = 'blue'
    color_pred = 'red'
    quiver_scale = 1
    quiver_alpha = 0.5
    quiver_width = 0.003

    fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)

    for ax in axs:
        ax.scatter(X, Y, color='k', s=10, alpha=0.3)

    # 只创建一次 Quiver
    # 左手
    if plt_gt:
        left_gt_qv = axs[0].quiver(X, Y,
                                gt[0, :, :, 0], gt[0, :, :, 1],
                                color=color_gt, angles='xy', scale_units='xy', scale=quiver_scale,
                                width=quiver_width, alpha=quiver_alpha, label='Left GT')
    if plt_pred:
        left_pred_qv = axs[0].quiver(X, Y,
                                    pred[0, :, :, 0], pred[0, :, :, 1],
                                    color=color_pred, angles='xy', scale_units='xy', scale=quiver_scale,
                                    width=quiver_width, alpha=quiver_alpha, label='Left Pred')
    axs[0].set_title('Left Hand (GT & Pred)')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    # 只放一次图例，避免重复
    handles0, labels0 = axs[0].get_legend_handles_labels()
    by_label0 = dict(zip(labels0, handles0))
    axs[0].legend(by_label0.values(), by_label0.keys(), loc='upper right')

    # 右手
    if plt_gt:
        right_gt_qv = axs[1].quiver(X, Y,
                                    gt[0, :, :, 3], gt[0, :, :, 4],
                                    color=color_gt, angles='xy', scale_units='xy', scale=quiver_scale,
                                    width=quiver_width, alpha=quiver_alpha, label='Right GT')
    if plt_pred:
        right_pred_qv = axs[1].quiver(X, Y,
                                    pred[0, :, :, 3], pred[0, :, :, 4],
                                    color=color_pred, angles='xy', scale_units='xy', scale=quiver_scale,
                                    width=quiver_width, alpha=quiver_alpha, label='Right Pred')

    axs[1].set_title('Right Hand (GT & Pred)')
    axs[1].set_xlabel('X')
    handles1, labels1 = axs[1].get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    axs[1].legend(by_label1.values(), by_label1.keys(), loc='upper right')

    def update(i):
        # 关键：只更新 U/V，不创建/删除 artist
        res=[]
        # 更新标题以包含帧ID
        axs[0].set_title(f'Left Hand (GT & Pred) - Frame {i}')
        axs[1].set_title(f'Right Hand (GT & Pred) - Frame {i}')
        res.append(axs[0].title)
        res.append(axs[1].title)
        if plt_gt:
            left_gt_qv.set_UVC(   gt[i, :, :, 0],    gt[i, :, :, 1])
            res.append(left_gt_qv)
        if plt_pred:
            left_pred_qv.set_UVC( pred[i, :, :, 0],  pred[i, :, :, 1])
            res.append(left_pred_qv)
        if plt_gt:
            right_gt_qv.set_UVC(  gt[i, :, :, 3],    gt[i, :, :, 4])
            res.append(right_gt_qv)
        if plt_pred:
            right_pred_qv.set_UVC(pred[i, :, :, 3], pred[i, :, :, 4])
            res.append(right_pred_qv)
        # 返回这些 artist（blit=True 时需要；blit=False 也可返回）
        return res

    # 提示：quiver + blit 在不同后端表现不一；为稳妥可用 blit=False
    anim = FuncAnimation(fig, update, frames=T, interval=150, blit=False)
    plt.tight_layout()

    # 选择合适的 writer
    if save_path.lower().endswith('.mp4'):
        writer = FFMpegWriter(fps=fps)   # 需要系统安装 ffmpeg
    elif save_path.lower().endswith('.gif'):
        writer = PillowWriter(fps=fps)   # 纯 Python，无需 ffmpeg
    else:
        raise ValueError("save_path 后缀需为 .mp4 或 .gif")

    anim.save(save_path, writer=writer, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    root_path = '/home/tars/projects/tactile_wm/outputs/2026-02-07/20-53-09'
    data_path = os.path.join(root_path, 'infer_log')
    save_path = os.path.join(root_path, 'tactile_vis')
    save_img = os.path.join(root_path, 'obs_vis')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_img, exist_ok=True)
    file_list = sorted(os.listdir(data_path))
    for i in range(len(file_list)):
        data = np.load(os.path.join(data_path, file_list[i]), allow_pickle=True)
        tactile_pred = data.item()['pred_tactiles']
        image = data.item()['wrist_images'][-1]
        save_path_i = os.path.join(save_path, str("%04d"%i) + '.mp4')
        plot_tactile_grids_animation_v2(tactile_pred, tactile_pred, save_path_i, plt_gt=False)
        save_img_i = os.path.join(save_img, str("%04d"%i) + '.jpg')
        cv2.imwrite(save_img_i, data.item()['wrist_images'][-1][..., ::-1])
