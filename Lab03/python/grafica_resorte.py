
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# 1. DATOS EXPERIMENTALES
# ==================================================

g = 9.81  # Aceleracion de la gravedad (m/s^2)

# Masas individuales (kg)
m = np.array([0.148, 0.346, 0.596, 1.093, 2.081])

# Masas acumuladas y pesos aplicados
M = np.cumsum(m)
P = M * g

# Longitud inicial del resorte (m)
l0 = 0.203

# Longitudes de carga, ordenadas por peso creciente (m)
l_carga = np.array([
    0.204, 0.226, 0.272, 0.359, 0.532
])

# Longitudes de descarga, ordenadas por peso decreciente (m)
l_descarga = np.array([
    0.532, 0.359, 0.272, 0.226, 0.204
])

# Elongaciones
dl_carga = l_carga - l0
dl_descarga = l_descarga - l0

# Pesos durante la descarga: orden inverso
P_descarga = P[::-1]

# ==================================================
# 2. INCERTIDUMBRES
# ==================================================

u_m = 0.0005  # kg
u_l = 0.0005  # m

# Incertidumbre de las masas acumuladas
u_M = np.sqrt(np.arange(1, 6)) * u_m

# Incertidumbre de los pesos
u_P = g * u_M
u_P_descarga = u_P[::-1]

# Incertidumbre de la elongacion
u_dl = np.sqrt(2) * u_l

# ==================================================
# 3. REGRESION LINEAL
# ==================================================

# Ajuste P = k * Delta_l + b
k, b = np.polyfit(dl_carga, P, 1)

P_predicho = k * dl_carga + b

SS_res = np.sum((P - P_predicho)**2)
SS_tot = np.sum((P - np.mean(P))**2)

R2 = 1 - SS_res / SS_tot

# Recta de regresion
x_ajuste = np.linspace(
    min(dl_carga),
    max(dl_carga),
    300
)

y_ajuste = k * x_ajuste + b

# ==================================================
# 4. CONFIGURACION GRAFICA
# ==================================================

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
    dl_carga,
    P,
    xerr=u_dl,
    yerr=u_P,
    fmt="none",
    ecolor="gray",
    elinewidth=1,
    capsize=3,
    zorder=2
)

# DESCARGA: cuadrados azules huecos
# Se dibujan primero porque coinciden con los puntos de carga.
ax.scatter(
    dl_descarga,
    P_descarga,
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
    dl_carga,
    P,
    marker="o",
    s=32,
    color="black",
    label="Carga",
    zorder=4
)

# Mostrar únicamente los valores del eje Y
for i in range(len(P)):
    ax.annotate(
        f"{P[i]:.2f}",
        xy=(dl_carga[i], P[i]),
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=9,
        color="black"
    )
    
# Recta de regresion lineal
ax.plot(
    x_ajuste,
    y_ajuste,
    linestyle="--",
    color="#c43c39",
    linewidth=1.6,
    label="Ajuste lineal",
    zorder=1
)

# Ecuacion y coeficiente de determinacion
ecuacion = (
    rf"$P = {k:.3f}\,\Delta\ell {b:+.3f}$"
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

# Etiquetas
ax.set_xlabel(r"Elongacion, $\Delta\ell$ (m)")
ax.set_ylabel(r"Peso aplicado, $P$ (N)")

ax.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.5
)

ax.legend(
    frameon=False,
    loc="lower right"
)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.tight_layout()

# ==================================================
# 5. GUARDAR LA GRAFICA
# ==================================================

carpeta = (
    Path(__file__).resolve().parent.parent
    / "figuras"
)

carpeta.mkdir(parents=True, exist_ok=True)

fig.savefig(
    carpeta / "peso_elongacion_resorte.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    carpeta / "peso_elongacion_resorte.pdf",
    bbox_inches="tight"
)

# ==================================================
# 6. MOSTRAR RESULTADOS
# ==================================================

print("\nDATOS EXPERIMENTALES")
print("-" * 65)

for i in range(5):
    print(
        f"Etapa {i+1}: "
        f"M = {M[i]:.3f} kg | "
        f"P = {P[i]:.5f} N | "
        f"Delta_l = {dl_carga[i]:.3f} m"
    )

print("\nAJUSTE LINEAL")
print(f"Pendiente k = {k:.5f} N/m")
print(f"Intercepto b = {b:.5f} N")
print(f"R^2 = {R2:.6f}")

print("\nGrafica guardada en:")
print(carpeta / "peso_elongacion_resorte.png")
