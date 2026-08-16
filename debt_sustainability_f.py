"""
Debt sustainability: tilde_a = lowest nonpositive root of f(a) = 0.

Baseline parameters:
    r = 4%, rho = 8%, g = 2, lambda = 0.1, c = 0.25, y_tilde = 1
    y = 1 (normalization; output scales linearly with y)

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
Y = 1.0  # fixed normalization (not varied in the 6 graphs)

CENTRAL = dict(
    r=0.04,
    rho=0.08,
    g=2.0,
    lam=0.1,
    c=0.27,
    y_tilde=1.0,
)


def regime(r, rho, g, lam, c):
    """Return 1, 2, or 3 for the active branch."""
    dr = rho - r
    bound1 = (g * (1 - c) - 1) * lam
    bound2 = (g - 1) * lam
    if dr <= bound1:
        return 1
    if dr <= bound2:
        return 2
    return 3


def f(a, r, rho, g, lam, c, y_tilde, y=Y):
    """
    f(a) piecewise value function minus reference level.

    Regime 1 (rho-r <= [g(1-c)-1] lambda):
        gy/rho * lambda/(lambda+rho-r) * [1 + ra/y - c(1+ra/y)^((rho+lambda)/r)]
        - g y_tilde/rho * lambda/(lambda+rho-r) * [1-c]

    Regime 2 ([g(1-c)-1]lambda <= rho-r <= [g-1]lambda):
        gy lambda/(rho(lambda+rho-r)) * (1+ra/y)
        - y[lambda(g-1)+r-rho]/(rho(lambda+rho-r)) * (1+ra/y)^((rho+lambda)/r)
        - g y_tilde lambda/(rho(lambda+rho-r))
        + y_tilde[lambda(g-1)+r-rho]/(rho(lambda+rho-r))

    Regime 3 ([g-1]lambda <= rho-r):
        (y + r a)/rho - y_tilde/rho
    """
    reg = regime(r, rho, g, lam, c)
    x = r * a / y

    if reg == 3:
        return (y + r * a - y_tilde) / rho

    d = lam + rho - r
    exp = (rho + lam) / r

    if 1 + x <= 0:
        return np.nan

    if reg == 1:
        pref = (g * y / rho) * (lam / d)
        ref = (g * y_tilde / rho) * (lam / d) * (1 - c)
        return pref * (1 + x - c * (1 + x) ** exp) - ref

    pref = (g * y * lam) / (rho * d)
    coef = y * (lam * (g - 1) + r - rho) / (rho * d)
    ref = (g * y_tilde * lam) / (rho * d) - y_tilde * (lam * (g - 1) + r - rho) / (rho * d)
    return pref * (1 + x) - coef * (1 + x) ** exp - ref


def find_tilde_a(r, rho, g, lam, c, y_tilde, y=Y, a_min=None, a_max=0.0, n_scan=4000):
    """
    Lowest nonpositive solution to f(a) = 0 on [a_min, 0].
    Returns 0 if no nonpositive root exists.
    """
    reg = regime(r, rho, g, lam, c)

    # Regime 3: linear, unique root
    if reg == 3:
        root = (y_tilde - y) / r
        return root if root <= 0 else 0.0

    # For regimes 1-2, need 1 + r*a/y > 0 for real-valued power
    if a_min is None:
        a_min = -0.999 * y / r

    def fn(a):
        return f(a, r, rho, g, lam, c, y_tilde, y)

    grid = np.linspace(a_min, a_max, n_scan)
    vals = np.array([fn(a) for a in grid])

    roots = []
    for i in range(len(grid) - 1):
        lo, hi = grid[i], grid[i + 1]
        flo, fhi = vals[i], vals[i + 1]
        if abs(flo) < 1e-12 and lo < 1e-12:
            roots.append(lo)
        elif flo * fhi < 0:
            try:
                rt = brentq(fn, lo, hi, xtol=1e-12)
                if rt <= 1e-10:  # nonpositive (allow tiny numerical noise)
                    roots.append(rt)
            except ValueError:
                pass

    if not roots:
        return 0.0

    roots.sort()
    # Deduplicate
    unique = [roots[0]]
    for rt in roots[1:]:
        if rt - unique[-1] > 1e-5:
            unique.append(rt)
    return unique[0]


def sweep(param, values, y=Y):
    out = []
    for v in values:
        p = {**CENTRAL, param: v}
        out.append(
            find_tilde_a(
                r=p["r"],
                rho=p["rho"],
                g=p["g"],
                lam=p["lam"],
                c=p["c"],
                y_tilde=p["y_tilde"],
                y=y,
            )
        )
    return np.array(out, dtype=float)


def sweep_regime(param, values):
    """Regime (1, 2, or 3) at each swept parameter value."""
    out = []
    for v in values:
        p = {**CENTRAL, param: v}
        out.append(regime(p["r"], p["rho"], p["g"], p["lam"], p["c"]))
    return np.array(out, dtype=int)


REGIME_COLORS = {1: "#d6eaf8", 2: "#fdebd0", 3: "#fadbd8"}
REGIME_LABELS = {1: "Patient Gov", 2: "Impatient Gov", 3: "Very Impatient Gov"}


def shade_regimes(ax, vals, regs):
    """Draw colored background spans for each regime."""
    i = 0
    while i < len(vals):
        reg = regs[i]
        j = i + 1
        while j < len(vals) and regs[j] == reg:
            j += 1
        left = vals[i] if i == 0 else (vals[i - 1] + vals[i]) / 2
        right = vals[j - 1] if j == len(vals) else (vals[j - 1] + vals[j]) / 2
        ax.axvspan(left, right, color=REGIME_COLORS[reg], alpha=0.55, zorder=0)
        i = j


def regime_legend_handles(regs):
    """Legend patches for regimes present in a panel."""
    return [
        Patch(
            facecolor=REGIME_COLORS[r],
            alpha=0.55,
            edgecolor="gray",
            linewidth=0.5,
            label=REGIME_LABELS[r],
        )
        for r in sorted(set(regs))
    ]


def main():
    ranges = {
        "r":   np.linspace(0.03, 0.05, 100),
        "rho": np.linspace(0.045, 0.135, 100),
        "g":   np.linspace(1.6, 2.2, 100),
        "lam": np.linspace(0.001, 0.3, 100),
        "c":   np.linspace(0.15, 0.4, 100),
    }


    labels = {
        "r": r"$r$",
        "rho": r"$\rho$",
        "g": r"$g$",
        "lam": r"$\lambda$",
        "c": r"$c$",
    }

    descriptions = {
        "r": "Risk free rate",
        "rho": "Government discount rate",
        "g": "Marginal value of spending exigencies",
        "lam": "Poisson rate of spending exigencies",
        "c": "fire-sale cost",
    }

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    axes = axes.ravel()

    central_tilde = find_tilde_a(
        CENTRAL["r"], CENTRAL["rho"], CENTRAL["g"], CENTRAL["lam"],
        CENTRAL["c"], CENTRAL["y_tilde"], y=Y,
    )
    reg_central = regime(CENTRAL["r"], CENTRAL["rho"], CENTRAL["g"], CENTRAL["lam"], CENTRAL["c"])

    # fig.suptitle(
    #     rf"Baseline parameters: $r={100 * CENTRAL['r']:.0f}\%$, $\rho={100 * CENTRAL['rho']:.0f}\%$, "
    #     rf"$g={CENTRAL['g']}$, $\lambda={CENTRAL['lam']}$, $c={CENTRAL['c']}$, "
    #     rf"$\tilde{{y}}={CENTRAL['y_tilde']}$",
    #     fontsize=12,
    # )

    for ax, (param, vals) in zip(axes, ranges.items()):
        regs = sweep_regime(param, vals)
        shade_regimes(ax, vals, regs)

        tilde = sweep(param, vals)
        (line,) = ax.plot(vals, tilde, lw=3, color="steelblue", zorder=3, label=r"Sustainable debt $\tilde{a}$")
        ax.axhline(0, color="black", lw=0.8, alpha=0.4, zorder=2)
        (baseline,) = ax.plot(
            CENTRAL[param], central_tilde, "ro", ms=10, zorder=4, label=r"Baseline $\tilde{a}$"
        )
        ax.legend(
            handles=regime_legend_handles(regs) + [line, baseline],
            fontsize=12,
            loc="best",
        )
        ax.set_xlabel(labels[param], fontsize=15)
        ax.set_ylabel(r"$\tilde{a}$", fontsize=15)
        ax.set_title(f"{labels[param]} {descriptions[param]}", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.3)

        if param in ("r", "rho"):
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{100 * x:.1f}%")
            )

    axes[-1].axis("off")

    plt.tight_layout()
    out = "debt_sustainability_tilde_a_6panels.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print(f"Baseline regime: {reg_central}")
    print(f"Baseline tilde_a:  {central_tilde}")


def hat_y_label(y_tilde, y=Y):
    """Format inline label as y_hat = 0.99y, etc."""
    ratio = y_tilde / y
    if np.isclose(ratio, 1.0):
        return r"$\hat{y} = y$"
    ratio_str = f"{ratio:.2f}".rstrip("0").rstrip(".")
    return rf"$\hat{{y}} = {ratio_str}y$"


def plot_tilde_a_vs_c():
    """Second figure: tilde(a) vs c for several y_tilde values."""
    y_tilde_values = np.linspace(1, 0.9, 11)
    c_vals = np.linspace(0.0, 0.4, 100)
    blue = "steelblue"

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    label_positions = np.linspace(0.12, 0.88, len(y_tilde_values))

    for i, yt in enumerate(y_tilde_values):
        tilde = np.array(
            [
                find_tilde_a(
                    CENTRAL["r"],
                    CENTRAL["rho"],
                    CENTRAL["g"],
                    CENTRAL["lam"],
                    c,
                    yt,
                    y=Y,
                )
                for c in c_vals
            ]
        )
        ax.plot(c_vals, tilde, lw=3, color=blue)
        idx = int(label_positions[i] * (len(c_vals) - 1))
        label = hat_y_label(yt, y=Y)
        ax.text(
            c_vals[idx],
            tilde[idx],
            label,
            color=blue,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        )

    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$c$ fire-sale cost", fontsize=15)
    ax.set_ylabel(r"$\tilde{a}$", fontsize=15)
    # ax.set_title(
    #     rf"Sustainable debt $\tilde{{a}}$ vs $c$  "
    #     rf"(baseline: $r={100 * CENTRAL['r']:.0f}\%$, $\rho={100 * CENTRAL['rho']:.0f}\%$, "
    #     rf"$g={CENTRAL['g']}$, $\lambda={CENTRAL['lam']}$)",
    #     fontsize=16,
    # )
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "debt_sustainability_tilde_a_vs_c.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    # Quick sanity check
    print("f(0) at central =", f(0, **CENTRAL))
    print("f(-1) at central =", f(-1, **CENTRAL))
    main()
    plot_tilde_a_vs_c()
