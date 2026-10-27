
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# DATOS EXPERIMENTALES
g = 9.81
m = np.array([0.148, 0.346, 0.596, 1.093, 2.081])
P = np.cumsum(m) * g

l0 = 0.332
l = np.array([0.340, 0.352, 0.373, 0.437, 0.584])
a = np.array([0.01480, 0.01410, 0.01330, 0.01215, 0.01060])
b = np.array([0.00635, 0.00585, 0.00560, 0.00535, 0.00480])

# DESCARGA: mismos valores en orden inverso
P_d = P[::-1]
l_d = l[::-1]
a_d = a[::-1]
b_d = b[::-1]

# MAGNITUDES
dl = l - l0
dl_d = l_d - l0

epsilon = dl / l0
epsilon_d = dl_d / l0

sigma = P / (a * b) / 1000  # kPa
sigma_d = P_d / (a_d * b_d) / 1000

# INCERTIDUMBRES
u_m = 0.0005
u_l = 0.0005
u_a = 0.00002
u_b = 0.00002

u_P = g * np.sqrt(np.arange(1, 6)) * u_m
u_dl = np.full(5, np.sqrt(2) * u_l)

u_epsilon = np.sqrt(
    (u_l / l0)**2 + (l * u_l / l0**2)**2
)

u_sigma = sigma * np.sqrt(
    (u_P / P)**2 +
    (u_a / a)**2 +
    (u_b / b)**2
)

# CARPETA DE SALIDA
salida = Path(__file__).resolve().parent.parent / "figuras"
salida.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11
})

def graficar(x, y, x_d, y_d, ux, uy, xlabel, ylabel, nombre):
    pendiente, intercepto = np.polyfit(x, y, 1)
    pred = pendiente * x + intercepto

    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)

    xx = np.linspace(min(x), max(x), 300)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(
        x, y, xerr=ux, yerr=uy,
        fmt="none", ecolor="gray",
        capsize=3, linewidth=1
    )

    # Descarga: cuadrados huecos
    ax.scatter(
        x_d, y_d, s=115, marker="s",
        facecolors="none", edgecolors="#2563a6",
        linewidths=1.5, label="Descarga", zorder=3
    )

    # Carga: circulos negros
    ax.scatter(
        x, y, s=32, color="black",
        label="Carga", zorder=4
    )

    ax.plot(
        xx, pendiente * xx + intercepto,
        "--", color="#c43c39",
        linewidth=1.6, label="Ajuste lineal"
    )

    # Mostrar solo los valores de Y
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.2f}", (xi, yi),
            xytext=(7, -12),
            textcoords="offset points",
            fontsize=9
        )

    ecuacion = (
        rf"$y={pendiente:.3f}x{intercepto:+.3f}$"
        "\n"
        rf"$R^2={r2:.4f}$"
    )

    ax.text(
        0.05, 0.95, ecuacion,
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", edgecolor="gray")
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(salida / f"{nombre}.png", dpi=300, bbox_inches="tight")
    fig.savefig(salida / f"{nombre}.pdf", bbox_inches="tight")

    plt.close(fig)

    print(f"{nombre}: pendiente={pendiente:.4f}, "
          f"intercepto={intercepto:.4f}, R²={r2:.5f}")

# GRAFICA A: PESO VS ELONGACION
graficar(
    dl, P, dl_d, P_d,
    u_dl, u_P,
    r"Elongación, $\Delta\ell$ (m)",
    r"Peso aplicado, $P$ (N)",
    "peso_elongacion_liga"
)

# GRAFICA B: ESFUERZO VS DEFORMACION
graficar(
    epsilon, sigma, epsilon_d, sigma_d,
    u_epsilon, u_sigma,
    r"Deformación unitaria, $\varepsilon$",
    r"Esfuerzo calculado, $\sigma$ (kPa)",
    "esfuerzo_deformacion_liga"
)
