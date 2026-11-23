
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Parameters
lam = 0.1
g = 2
r = 0.04
rho = 0.08
y = 1

# Grid
c_values = np.linspace(0, 0.2, 401)
y_tilde_values = np.linspace(0.9, 1.0, 6)

# Nonpositive implicit solution
def a_tilde(c, y_tilde):

    def equation(a):
        x = 1 + r*a/y
        return x - c*x**((rho+lam)/r) \
               - (y_tilde/y)*(1-c)

    return brentq(equation, -y/r, 0)

# Font settings
plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 36,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'pdf.fonttype': 42
})

# Large figure
fig, ax = plt.subplots(figsize=(14, 8))

for y_tilde in y_tilde_values:

    a_values = [a_tilde(c, y_tilde)
                for c in c_values]

    line, = ax.plot(
        c_values, a_values, linewidth=4
    )

    # Label each curve
    c_label = 0.145

    ax.text(
        c_label,
        a_tilde(c_label, y_tilde),
        fr'$\tilde{{y}}={y_tilde:.2f}$',
        color=line.get_color(),
        fontsize=30,
        ha='center',
        va='center',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.9,
            pad=1.3
        )
    )

# Axes
ax.set_xlabel(r'$c$', labelpad=14)
ax.set_ylabel(r'$\tilde{a}$', labelpad=15)

ax.set_xlim(0, 0.2)
ax.set_ylim(-5.7, 0.5)
ax.set_xticks(np.linspace(0, 0.2, 5))

ax.tick_params(axis='both', length=8, width=1.5)
ax.grid(True, alpha=0.25, linewidth=1.1)

plt.tight_layout()

# Save as vector PDF
plt.savefig(
    "a_tilde.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
