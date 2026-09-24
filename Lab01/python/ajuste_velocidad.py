import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LECTURA DE LOS DATOS EXPERIMENTALES
# ============================================================

archivo = "../datos/LABORATORIO_1_mar02.xlsx"

df = pd.read_excel(
    archivo,
    sheet_name="pruerba 1 rotacion"
)

# Extraer las columnas de tiempo y velocidad
t = pd.to_numeric(df["Time"], errors="coerce")
v = pd.to_numeric(df["Speed at A Run 1"], errors="coerce")

# Eliminar filas vacías o no numéricas
datos = pd.DataFrame({
    "t": t,
    "v": v
}).dropna()

t = datos["t"].to_numpy()
v = datos["v"].to_numpy()

print(f"Número de datos analizados: {len(t)}")


# ============================================================
# 2. AJUSTE LINEAL
#
# Modelo:
#
#           v(t) = v0 + a t
#
# donde:
#   a  = aceleración lineal
#   v0 = ordenada al origen
# ============================================================

coeficientes, covarianza = np.polyfit(
    t,
    v,
    1,
    cov=True
)

a = coeficientes[0]
v0 = coeficientes[1]


# ============================================================
# 3. INCERTIDUMBRE DE LA PENDIENTE
# ============================================================

delta_a = np.sqrt(covarianza[0, 0])


# ============================================================
# 4. COEFICIENTE DE DETERMINACIÓN R²
# ============================================================

v_pred = a * t + v0

ss_res = np.sum((v - v_pred) ** 2)
ss_tot = np.sum((v - np.mean(v)) ** 2)

R2 = 1 - ss_res / ss_tot


# ============================================================
# 5. RESULTADOS NUMÉRICOS
# ============================================================

print("\nRESULTADOS DEL AJUSTE LINEAL")
print("------------------------------------------")
print(f"a       = {a:.8f} m/s²")
print(f"Δa      = {delta_a:.8f} m/s²")
print(f"v0      = {v0:.8f} m/s")
print(f"R²      = {R2:.6f}")
print("------------------------------------------")

print(
    f"\nResultado de la aceleración:\n"
    f"a = ({a:.8f} ± {delta_a:.8f}) m/s²"
)


# ============================================================
# 6. RECTA DEL AJUSTE
# ============================================================

t_recta = np.linspace(
    t.min(),
    t.max(),
    300
)

v_recta = a * t_recta + v0


# ============================================================
# 7. GRÁFICA
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

# Datos experimentales
ax.scatter(
    t,
    v,
    s=28,
    label="Datos experimentales",
    zorder=3
)

# Recta obtenida mediante regresión lineal
ax.plot(
    t_recta,
    v_recta,
    linewidth=1.8,
    label="Ajuste lineal",
    zorder=2
)


# ============================================================
# 8. INFORMACIÓN DEL AJUSTE EN LA GRÁFICA
# ============================================================

texto = (
    r"$v(t)=v_0+at$"
    "\n"
    rf"$a=({a:.6f}\pm{delta_a:.6f})\ "
    r"\mathrm{m/s^2}$"
    "\n"
    rf"$v_0={v0:.6f}\ \mathrm{{m/s}}$"
    "\n"
    rf"$R^2={R2:.5f}$"
)

ax.text(
    0.05,
    0.95,
    texto,
    transform=ax.transAxes,
    verticalalignment="top"
)


# ============================================================
# 9. FORMATO DE LA GRÁFICA
# ============================================================

ax.set_xlabel(
    r"Tiempo, $t$ (s)"
)

ax.set_ylabel(
    r"Velocidad, $v$ (m/s)"
)

ax.set_title(
    "Velocidad en función del tiempo: sistema disco–anillo"
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()


# ============================================================
# 10. GUARDAR FIGURA
# ============================================================

fig.savefig(
    "../figuras/ajuste_velocidad.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "../figuras/ajuste_velocidad.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nGráficas guardadas correctamente:")
print("../figuras/ajuste_velocidad.pdf")
print("../figuras/ajuste_velocidad.png")