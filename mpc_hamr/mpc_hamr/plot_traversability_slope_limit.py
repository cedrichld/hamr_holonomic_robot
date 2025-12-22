#!/usr/bin/env python3
"""
Plot traversability T as a function of slope with a dashed line showing
the maximum COMPA slope (0.6 rad ≈ 35°).

Equation:
    T = 0.9 * exp(-(slope / 0.4)^4) + 0.1 * exp(-(roughness / 0.03)^4)

For this plot, roughness is held constant (smooth-ish terrain).
"""

import numpy as np
import matplotlib.pyplot as plt

# -------- Figure / style config (ICRA-style single-column) --------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.figsize": (3.3, 2.4),  # single-column friendly
    "lines.linewidth": 1.6,
})

# Uncomment if you want full LaTeX rendering and have LaTeX installed
# plt.rcParams["text.usetex"] = True


def traversability(slope, roughness):
    """
    Traversability function:

        T = 0.9 * exp(-(slope / 0.4)^4)
          + 0.1 * exp(-(roughness / 0.03)^4)

    slope     : array-like, in radians
    roughness : scalar or array, same shape as slope
    """
    term_slope = 0.9 * np.exp(- (slope / 0.4) ** 4)
    term_rough = 0.1 * np.exp(- (roughness / 0.03) ** 4)
    return term_slope + term_rough


def main():
    # Domain for slope [rad]
    slope_min = 0.0
    slope_max = 0.9  # keep 0.6 rad nicely inside the range
    slope = np.linspace(slope_min, slope_max, 400)

    # Fix roughness for this 1D slice (smooth terrain)
    roughness_value = 0.0  # change if you want another slice

    T = traversability(slope, roughness_value)

    # Max COMPA slope
    slope_max_compa = 0.5  # [rad]
    slope_max_compa_deg = np.rad2deg(slope_max_compa)
    traversability_cutoff = 0.2

    fig, ax = plt.subplots()

    # Main curve
    ax.plot(
        slope,
        T,
        label=fr"$T(\theta)$, roughness $= {roughness_value:.1f}$"
    )

    # Dashed vertical line at max slope
    ax.axvline(
        x=slope_max_compa,
        linestyle="--",
        linewidth=1.4,
        label=fr"Max COMPA slope $~0.5\,\mathrm{{rad}} \approx {slope_max_compa_deg:.0f}^\circ$",
    )
    ax.axhline(
        y=traversability_cutoff,
        linestyle="--",
        color="red",
        linewidth=1.4,
        label=fr"Traversability cost cutoff = ${traversability_cutoff:.1f}$",
    )

    # Optional annotation near the top of the vertical line
    # ax.annotate(
    #     fr"$0.6\,\mathrm{{rad}} \approx {slope_max_compa_deg:.0f}^\circ$",
    #     xy=(slope_max_compa, traversability(slope_max_compa, roughness_value)),
    #     xytext=(slope_max_compa + 0.02, 0.85),
    #     textcoords="data",
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    # )

    # Axis labels
    ax.set_xlabel(r"Slope $\theta$ [rad]")
    ax.set_ylabel(r"Traversability score $T$")

    # Limits & grid
    ax.set_xlim(slope_min, slope_max)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle=":")

    # Legend without frame (clean for paper)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    # Save as high-quality outputs for LaTeX
    fig.savefig("traversability_vs_slope.pdf", bbox_inches="tight")
    fig.savefig("traversability_vs_slope.png", dpi=300, bbox_inches="tight")

    # Show for interactive use (disable in batch)
    plt.show()


if __name__ == "__main__":
    main()
