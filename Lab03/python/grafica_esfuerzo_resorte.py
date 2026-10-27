
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================
# 1. DATOS EXPERIMENTALES
# ================================================

g = 9.81  # m/s²

# Masas individuales (kg)
masas = np.array([
    0.148, 0.346, 0.596, 1.093, 2.081
])

# Carga acumulativa
M = np.cumsum(masas)
P = M * g

# Dimensiones iniciales (m)
l0 = 0.203
d0 = 0.01420

# CARGA: orden de menor a mayor peso
l_carga = np.array([
    0.204, 0.226, 0.272, 0.359, 0.532
])

d_carga = np.array([
    0.01419, 0.01418, 0.01416,
    0.01415, 0.01410
])

# DESCARGA: orden de mayor a menor peso
l_descarga = np.array([
    0.532, 0.359, 0.272, 0.226, 0.204
])

d_descarga = np.array([
    0.01410, 0.01415, 0.01416,
    0.01418, 0.01419
])

P_descarga = P[::-1]

# ================================================
# 2. ESFUERZO Y DEFORMACION UNITARIA
# ================================================

# Area circular calculada con el diametro medido
A_carga = np.pi * d_carga**2 / 4
A_descarga = np.pi * d_descarga**2 / 4

# Esfuerzo calculado (Pa)
sigma_carga = P / A_carga
sigma_descarga = P_descarga / A_descarga

# Convertir esfuerzo a kPa
sigma_carga = sigma_carga / 1000
sigma_descarga = sigma_descarga / 1000

# Deformacion longitudinal relativa
epsilon_carga = (l_carga - l0) / l0
epsilon_descarga = (l_descarga - l0) / l0

# ================================================
# 3. INCERTIDUMBRES
# ================================================

u_m = 0.0005   # kg
u_l = 0.0005   # m
u_d = 0.00002  # m

# Incertidumbre de las masas acumuladas
u_M = np.sqrt(np.arange(1, 6)) * u_m

# Incertidumbre de los pesos
u_P = g * u_M
u_P_descarga = u_P[::-1]

# Incertidumbre de la deformacion
u_epsilon_carga = np.sqrt(
    (u_l / l0)**2 +
    (l_carga * u_l / l0**2)**2
)

u_epsilon_descarga = np.sqrt(
    (u_l / l0)**2 +
    (l_descarga * u_l / l0**2)**2
)

# Incertidumbre del esfuerzo en kPa
u_sigma_carga = sigma_carga * np.sqrt(
    (u_P / P)**2 +
    (2 * u_d / d_carga)**2
)

u_sigma_descarga = sigma_descarga * np.sqrt(
    (u_P_descarga / P_descarga)**2 +
    (2 * u_d / d_descarga)**2
)

# ================================================
# 4. REGRESION LINEAL
# ================================================

# Ajuste por minimos cuadrados ordinarios
# sigma = a * epsilon + b

a, b = np.polyfit(
    epsilon_carga,
    sigma_carga,
    1
)

sigma_predicho = a * epsilon_carga + b

SS_res = np.sum(
    (sigma_carga - sigma_predicho)**2
)

SS_tot = np.sum(
    (sigma_carga - np.mean(sigma_carga))**2
)

R2 = 1 - SS_res / SS_tot

x_ajuste = np.linspace(
    min(epsilon_carga),
    max(epsilon_carga),
    300
)

y_ajuste = a * x_ajuste + b

# ================================================
# 5. GRAFICA
# ================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "figure.dpi": 150
})

fig, ax = plt.subplots(figsize=(8, 5.5))

# Barras de incertidumbre
ax.errorbar(
    epsilon_carga,
    sigma_carga,
    xerr=u_epsilon_carga,
    yerr=u_sigma_carga,
    fmt="none",
    ecolor="gray",
    elinewidth=1,
    capsize=3,
    zorder=2
)

# DESCARGA: cuadrados azules huecos
ax.scatter(
    epsilon_descarga,
    sigma_descarga,
    marker="s",
    s=115,
    facecolors="none",
    edgecolors="#2563a6",
    linewidths=1.5,
    label="Descarga",
    zorder=3
)

# CARGA: circulos negros
ax.scatter(
    epsilon_carga,
    sigma_carga,
    marker="o",
    s=32,
    color="black",
    label="Carga",
    zorder=4
)

# Recta de ajuste
ax.plot(
    x_ajuste,
    y_ajuste,
    linestyle="--",
    color="#c43c39",
    linewidth=1.6,
    label="Ajuste lineal",
    zorder=1
)

# Ecuacion y R²
ecuacion = (
    rf"$\sigma = {a:.3f}\varepsilon {b:+.3f}$"
    "\n"
    rf"$R^2 = {R2:.4f}$"
)

ax.text(
    0.05,
    0.95,
    ecuacion,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    bbox=dict(
        facecolor="white",
        edgecolor="gray",
        alpha=0.95,
        boxstyle="round,pad=0.5"
    )
)

# Etiquetas: solo valores del eje Y
for i in range(len(sigma_carga)):
    ax.annotate(
        f"{sigma_carga[i]:.2f}",
        xy=(epsilon_carga[i], sigma_carga[i]),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=9,
        color="black"
    )

# Ejes
ax.set_xlabel(
    r"Deformacion unitaria, $\varepsilon$"
)

ax.set_ylabel(
    r"Esfuerzo calculado, $\sigma$ (kPa)"
)

ax.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.5
)

ax.legend(frameon=False, loc="lower right")

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.tight_layout()

# ================================================
# 6. GUARDAR IMAGEN
# ================================================

carpeta = (
    Path(__file__).resolve().parent.parent
    / "figuras"
)

carpeta.mkdir(parents=True, exist_ok=True)

fig.savefig(
    carpeta / "esfuerzo_deformacion_resorte.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    carpeta / "esfuerzo_deformacion_resorte.pdf",
    bbox_inches="tight"
)

# ================================================
# 7. RESULTADOS
# ================================================

print("\nESFUERZO Y DEFORMACION DEL RESORTE")
print("-" * 65)

for i in range(5):
    print(
        f"Etapa {i+1}: "
        f"P = {P[i]:.5f} N | "
        f"epsilon = {epsilon_carga[i]:.6f} | "
        f"sigma = {sigma_carga[i]:.3f} kPa"
    )

print("\nREGRESION LINEAL")
print(f"Pendiente = {a:.5f} kPa")
print(f"Intercepto = {b:.5f} kPa")
print(f"R² = {R2:.6f}")

print("\nGrafica guardada correctamente.")
