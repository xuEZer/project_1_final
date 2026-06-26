from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2026)

n = 60
impute_err = np.random.uniform(0.05, 1.0, n)

# Rising lower bound: minimum possible forecast error grows with imputation error.
floor_slope = 0.55
floor_intercept = 0.05
floor = floor_intercept + floor_slope * impute_err

# Fixed ceiling keeps the upper bound constant.
# Points fill the gap between floor and ceiling uniformly.
# At low impute_err the gap is large → wide spread (low & high forecast).
# At high impute_err the gap shrinks → cloud squeezes upward.
ceiling = 0.82
gap = ceiling - floor
frac = np.random.uniform(0, 1, size=n)
forecast_err = floor + frac * gap

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(impute_err, forecast_err, c="#3b6fb6", edgecolors="#2a5298",
           alpha=0.75, s=48, linewidth=0.5, zorder=5)

# Linear fit dashed line
m, b = np.polyfit(impute_err, forecast_err, 1)
xs = np.linspace(impute_err.min(), impute_err.max(), 100)
ax.plot(xs, m * xs + b, linestyle="--", color="#cc4444", linewidth=1.5, zorder=4)


ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.3, linestyle="--")

outdir = Path(__file__).resolve().parent
outdir.mkdir(exist_ok=True)
outpath = outdir / "imputation_forecast_decoupling.png"

fig.tight_layout(pad=1.2)
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {outpath}")
