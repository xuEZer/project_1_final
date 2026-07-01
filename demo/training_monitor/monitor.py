"""Demo: epoch-level monitoring of imputation & prediction metrics.

在每个 epoch 结束时收集训练集/验证集上的插补 loss 和预测 MAE，
以及测试集上的插补/预测 MAE，绘制变化曲线。

用法:
    python demo_monitor.py                              # 默认参数
    python demo_monitor.py --imputator.model_type SAITS --training.num_epoch 10
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 非交互后端，保存为 png
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path（支持从子目录运行）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 项目内部模块
# ---------------------------------------------------------------------------
from src.data_loader import get_tensor_data_loader, load_h5, preprocess_dataset
from src.logger import setup_logger
from src.models.extern.SAITS.utils import masked_mae_cal, masked_mse_cal
from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor

GLOBAL_CONFIG_PATH = _PROJECT_ROOT / "configs" / "global.ini"
FIG_SAVE_DIR = Path(__file__).resolve().parent / "figures"

# 演示时预测采样数（减小以加快速度；实际训练可用 100）
DEMO_NUM_SAMPLES = 5


# ===================================================================
# 辅助函数
# ===================================================================


def _parse_args() -> ConfigParser:
    """解析 CLI 覆盖并合并 config."""
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(GLOBAL_CONFIG_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument("--training.num_epoch", type=int)
    parser.add_argument("--training.warmup_epoch", type=int)
    parser.add_argument("--training.batch_size", type=int)
    parser.add_argument("--imputator.model_type", type=str)
    parser.add_argument("--predictor.model_type", type=str)
    parser.add_argument("--experiment.exp_name", type=str)
    known, _ = parser.parse_known_args()

    overrides = {}
    for section_key, value in vars(known).items():
        if value is not None:
            section, key = section_key.split(".", 1)
            overrides.setdefault(section, {})[key] = str(value)

    for section, kv in overrides.items():
        for key, val in kv.items():
            config.set(section, key, val)

    # 加载模板
    for section in ["imputator", "predictor"]:
        model_type = config.get(section, "model_type")
        template_path = _PROJECT_ROOT / "configs" / "template" / section / f"{model_type}.ini"
        if template_path.exists():
            config.read(template_path)

    # 解析 dataset 字段
    dataset = config.get("dataset", "dataset")
    column_index = dataset.split("_")[-1]
    item_id = dataset.split("_")[-2]
    config.set("dataset", "column_index", column_index)
    config.set("dataset", "item_id", item_id)
    dataset_name_freq = "_".join(dataset.split("_")[:-2])
    template_path = _PROJECT_ROOT / "configs" / "template" / "dataset" / f"{dataset_name_freq}.ini"
    if template_path.exists():
        config.read(template_path)

    return config


def _compute_epoch_metrics(
    joint_model: JointModel,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    compute_imp_mae: bool = False,
) -> dict[str, float]:
    """Compute imputation loss + prediction MAE/MSE over one DataLoader.

    对 DataLoader 中的每个 batch:
      - 调用 joint_model.forward() 获取 imp_loss / pred_data
      - 用 batch 中的 Y_N(未来真值) 计算 pred_mae
      - 若 compute_imp_mae=True, 用 batch 中的 X(完整真值) 计算 imp_mae

    Args:
        joint_model: 模型（eval 模式由调用方设置）。
        loader: 评测 DataLoader。
        device: 设备。
        use_amp: 是否 AMP。
        compute_imp_mae: 是否计算 imp_mae（需要 loader 输出 X 完整真值）。

    Returns:
        {"imp_loss": float, "pred_mae": float, "pred_mse": float}
        若 loader 为空则返回全 0。
    """
    imp_loss_sum = 0.0
    pred_mae_sum = 0.0
    pred_mse_sum = 0.0
    imp_mae_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            n_batches += 1
            tensors = [t.to(device) for t in batch]

            if len(batch) == 6:
                # train/val: (X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask)
                X_N, X_NA, X_NA_mask, X_A_mask, Y, Y_mask = tensors
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = joint_model.forward(X_NA, X_NA_mask, X_N, X_A_mask)
                # 预测误差: pred_data vs Y_N
                pred_target = Y
                pred_target_mask = 1 - Y_mask

            elif len(batch) == 4:
                if compute_imp_mae:
                    # with_gt  test: (X_N, X_N_mask, X, Y)
                    X_N, X_N_mask, X_gt, Y_gt = tensors
                    pred_target = Y_gt
                    pred_target_mask = torch.ones_like(Y_gt)
                else:
                    # no_gt   test: (X_N, X_N_mask, Y_N, Y_N_mask)
                    X_N, X_N_mask, pred_target, pred_target_mask = tensors
                    pred_target_mask = 1 - pred_target_mask
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = joint_model.forward(X_N, X_N_mask)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            imp_loss = output["imp_loss"]
            pred_data = output["pred_data"]

            if imp_loss is not None:
                imp_loss_sum += imp_loss.item()

            # 预测误差: pred_data vs pred_target
            if pred_data is not None and pred_data.shape == pred_target.shape:
                pred_mae_sum += masked_mae_cal(
                    pred_data, pred_target, pred_target_mask
                ).item()
                pred_mse_sum += masked_mse_cal(
                    pred_data, pred_target, pred_target_mask
                ).item()

            # 插补误差（仅 test 有完整真值 X）
            if compute_imp_mae and len(batch) == 4:
                imp_data = output["imp_data"]
                if imp_data is not None:
                    imp_mae_sum += masked_mae_cal(imp_data, X_gt, X_N_mask).item()

    n = max(1, n_batches)
    metrics = {
        "imp_loss": imp_loss_sum / n,
        "pred_mae": pred_mae_sum / n,
        "pred_mse": pred_mse_sum / n,
    }
    if compute_imp_mae:
        metrics["imp_mae"] = imp_mae_sum / n
    return metrics


def _plot_metrics(history: dict[str, list[float]], save_path: Path) -> None:
    """绘制插补与预测指标随 epoch 的变化图."""
    epochs = list(range(1, len(history.get("train_imp_loss", [])) + 1))
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图: 插补指标 ----
    ax = axes[0]
    if "train_imp_loss" in history:
        ax.plot(epochs, history["train_imp_loss"], "o-", label="Train imp loss")
    if "val_imp_loss" in history:
        ax.plot(epochs, history["val_imp_loss"], "s-", label="Val imp loss")
    if "test_imp_mae" in history:
        ax.plot(epochs, history["test_imp_mae"], "d-", label="Test imp MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE")
    ax.set_title("Imputation Metrics")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ---- 右图: 预测指标 ----
    ax = axes[1]
    if "train_pred_mae" in history:
        ax.plot(epochs, history["train_pred_mae"], "o-", label="Train pred MAE")
    if "val_pred_mae" in history:
        ax.plot(epochs, history["val_pred_mae"], "s-", label="Val pred MAE")
    if "test_pred_mae" in history:
        ax.plot(epochs, history["test_pred_mae"], "d-", label="Test pred MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Prediction Metrics")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Training Monitor — Imputation & Prediction", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {save_path}")


# ===================================================================
# 主流程
# ===================================================================


def demo() -> None:
    """Run the monitoring demo."""
    # ---- 配置与日志 ----
    config = _parse_args()
    device = config.get("training", "device")

    exp_name = config.get("experiment", "exp_name", fallback="demo_monitor")
    log_save_dir = Path(config.get("paths", "log_save_dir"))
    log_save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_save_dir / f"{exp_name}.log", log_name=exp_name)
    logger.info("=" * 60)
    logger.info("Demo: Training Monitor — Imputation & Prediction Metrics")
    logger.info("=" * 60)

    seed = int(config.get("experiment", "random_seed", fallback="42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Device: {device} | Random seed: {seed}")

    # ---- 数据准备 ----
    logger.info("--- Preprocessing / loading data ---")
    dataset_folder = preprocess_dataset(
        dataset_name=config.get("dataset", "dataset_name"),
        frequency=config.get("dataset", "frequency"),
        item_id=config.getint("dataset", "item_id"),
        column_index=config.getint("dataset", "column_index"),
        missing_rate=config.getfloat("dataset", "missing_rate"),
        artificial_missing_rate=config.getfloat("dataset", "artificial_missing_rate"),
        block_size=config.getint("dataset", "block_size"),
        lookback_len=config.getint("dataset", "lookback_len"),
        pred_len=config.getint("dataset", "pred_len"),
        val_ratio=config.getfloat("dataset", "val_ratio"),
        test_ratio=config.getfloat("dataset", "test_ratio"),
        random_seed=seed,
        logger=logger,
        save_path=Path(config.get("paths", "preprocessed_data_dir")),
    )
    dataset = load_h5(dataset_folder / "dataset.h5")

    config.set("dataset", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("dataset", "lookback_len", str(dataset["train"]["X_N"].shape[1]))

    target_batch_size = config.getint("training", "batch_size", fallback=32)
    micro_batch_size = target_batch_size  # demo 不搜索 batch size

    train_loader_dict, test_loader, no_gt = get_tensor_data_loader(
        dataset, batch_size=micro_batch_size
    )
    val_loader = train_loader_dict["val"]
    train_loader = train_loader_dict["train"]

    logger.info(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Test batches: {len(test_loader)}"
    )

    # ---- 模型初始化 ----
    logger.info("--- Building models ---")
    use_amp = config.getboolean("training", "use_amp", fallback=False)

    # 将 predictor 的 num_samples 暂时改为小值加速 demo
    config.set("predictor", "num_samples", str(DEMO_NUM_SAMPLES))

    joint_model = JointModel(
        imputer=imputator(config.get("imputator", "model_type"), config),
        predictor=predictor(config.get("predictor", "model_type"), config),
        model_save_dir=log_save_dir / "ckpt_demo",
        optimizer_name=config.get("training", "optimizer", fallback="AdamW"),
        lr=config.getfloat("training", "lr", fallback=1e-4),
        device=device,
        tb_summary_writer=SummaryWriter(log_dir=log_save_dir / "tb_demo"),
        logger=logger,
        use_amp=use_amp,
    ).to(device)

    logger.info(
        f"Imputer: {config.get('imputator', 'model_type')} | "
        f"Predictor: {config.get('predictor', 'model_type')} "
        f"(num_samples={DEMO_NUM_SAMPLES}, demo mode)"
    )

    # ---- 监控容器 ----
    history: dict[str, list[float]] = defaultdict(list)
    FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  训练：直接使用 JointModel.fit()，通过 on_epoch_end 回调收集指标
    #  训练逻辑（AMP / 梯度累积 / 早停）与 main.py 完全一致
    # ================================================================
    logger.info("--- Training begins ---")

    def _on_epoch_end(
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: JointModel,
    ) -> None:
        """Callback: 每个 epoch 结束后收集插补 + 预测指标."""
        model.eval()

        # 复用 JointModel 内部的 validate 来获取 val imp_loss
        train_metrics = _compute_epoch_metrics(
            model, train_loader, device, use_amp
        )
        val_pred_metrics = _compute_epoch_metrics(
            model, val_loader, device, use_amp
        )
        test_metrics = _compute_epoch_metrics(
            model,
            test_loader,
            device,
            use_amp,
            compute_imp_mae=(not no_gt),
        )

        history["train_imp_loss"].append(train_loss)
        history["train_pred_mae"].append(train_metrics["pred_mae"])
        history["val_imp_loss"].append(val_loss)
        history["val_pred_mae"].append(val_pred_metrics["pred_mae"])
        history["test_pred_mae"].append(test_metrics["pred_mae"])
        if not no_gt:
            history["test_imp_mae"].append(test_metrics.get("imp_mae", 0.0))

        logger.info(
            f"  ├─ val  imp_loss={val_loss:.6f}  "
            f"pred_MAE={val_pred_metrics['pred_mae']:.6f}\n"
            f"  └─ test pred_MAE={test_metrics['pred_mae']:.6f}"
        )

    joint_model.fit(
        data_loader=train_loader_dict,
        phase_name="demo",
        num_epoch=config.getint("training", "num_epoch", fallback=10),
        on_epoch_end=_on_epoch_end,
    )

    logger.info("--- Training complete ---")

    # ---- 最终图 ----
    final_plot = FIG_SAVE_DIR / "final.png"
    _plot_metrics(history, final_plot)
    logger.info(f"Final plot saved → {final_plot}")
    print(f"\n所有指标图已保存至 {FIG_SAVE_DIR.resolve()}")
    print(f"运行 `open {FIG_SAVE_DIR}/final.png` 查看最终结果")


if __name__ == "__main__":
    demo()
