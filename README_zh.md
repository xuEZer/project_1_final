# FOTSI

带插补的时序预测实验框架。支持两阶段基线 (B1) 和 TIP 联合训练 (B2)。

## 环境配置

```bash
uv sync
cd src/models/predictors/Sundial && uv sync && cd -
cd src/models/predictors/TimesFM25 && uv sync && cd -
```

国内用户设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 运行

```bash
./scripts/run_experiments.sh                  # B1 + B2, 全部 7 个数据集
./scripts/run_experiments.sh --failed-only    # 跳过已完成实验
./scripts/run_experiments.sh --dataset CPHL_30T --exp B2
./scripts/run_k_sweep.sh                      # TIP k 消融
./scripts/run_epsilon_sweep.sh                # TIP epsilon 消融
./scripts/run_missing_rate_robustness.sh      # B3 缺失率鲁棒性
```
