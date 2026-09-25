"""Render the four one-factor sensitivity scans from the frozen experiment results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


OUTPUT = Path("paper/Figures/pilot_sensitivity_four_panel.pdf")
B1_MSE = 0.45952
B2_MSE = 0.39685
SCANS = (
    ("(a) Rank", r"Rank $r$", (2, 3, 4, 5, 6),
     (0.42162, B2_MSE, 0.43002, 0.44400, 0.45939), 3),
    ("(b) Difference step", r"Step $\varepsilon$",
     (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2),
     (0.46380, 0.40898, 0.41514, B2_MSE, 0.45790, 0.44305, 0.46402), 1e-3),
    ("(c) Confidence level", r"Confidence $\delta$", (.05, .1, .2, .3, .4, .5),
     (0.44696, 0.45189, 0.39764, B2_MSE, 0.43587, 0.45066), .3),
    ("(d) Probe batches", "Probe batches", (2, 3, 4, 5, 6),
     (0.54985, 0.47263, B2_MSE, 0.44042, 0.45952), 4),
)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "cm",
        "font.size": 26,
        "axes.linewidth": 2.1,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.5), sharey=True,
                             constrained_layout=True)
    fig.get_layout_engine().set(wspace=.10, w_pad=.08, h_pad=.08)
    for ax, (title, xlabel, x, mse, default_x) in zip(axes, SCANS):
        ax.axhline(B1_MSE, color="#64748B", linestyle="--", linewidth=3.8,
                   zorder=1)
        ax.plot(x, mse, color="#2563EB", marker="o", linewidth=5.0,
                markersize=13.0, zorder=3)
        ax.plot(default_x, B2_MSE, linestyle="none", marker="*",
                color="#059669", markersize=22, zorder=4)
        ax.set(title=title, xlabel=xlabel, ylim=(.375, .565))
        ax.title.set_fontsize(29)
        ax.xaxis.label.set_fontsize(26)
        ax.tick_params(axis="both", labelsize=24, width=2)
        ax.grid(axis="y", alpha=.2, linewidth=1.6)
        if default_x == 1e-3:
            ax.set_xscale("log")
            ax.set_xticks((1e-4, 1e-3, 1e-2))
            ax.set_xticklabels((r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"))
        else:
            ax.set_xticks(x)
        if default_x == .3:
            ax.set_xticklabels((".05", ".1", ".2", ".3", ".4", ".5"),
                               rotation=35, ha="right")
    axes[0].set_ylabel("Test forecast MSE", fontsize=26)
    axes[0].set_yticks((.40, .45, .50, .55))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
