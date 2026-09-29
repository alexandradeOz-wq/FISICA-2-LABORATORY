import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
FIGURAS = BASE.parent / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)

M = 1.8402       # kg
L = 1.001        # m
b = 0.037        # m
g = 9.81         # m/s²

u_ell = 0.0005   # m

ell = np.array([
    0.503,
    0.454,
    0.403,
    0.353,
    0.304,
    0.253,
    0.204,
    0.153,
    0.102,
    0.053
])

N = np.array([
    20, 20, 20, 20, 20,
    20, 20, 10, 10, 10
])

datos_1 = np.array([
    [33.62, 33.62, 33.62],
    [32.90, 32.82, 32.90],
    [32.22, 32.06, 32.22],
    [32.01, 31.89, 31.64],
    [31.72, 31.82, 31.62],
    [32.11, 32.21, 32.33],
    [33.24, 33.24, 32.92],
    [17.71, 17.99, 17.76],
    [20.55, 20.37, 20.28],
    [26.44, 27.21, 26.70]
])

datos_2 = np.array([
    [33.57, 33.34, 33.28],
    [32.78, 32.96, 32.65],
    [32.30, 32.31, 32.91],
    [31.69, 32.11, 31.96],
    [31.94, 31.90, 32.40],
    [31.95, 32.31, 32.63],
    [33.28, 33.42, 33.14],
    [17.01, 17.63, 17.76],
    [20.33, 20.53, 20.53],
    [26.86, 27.04, 26.76]
])

datos_3 = np.array([
    [33.41, 33.17, 31.58],
    [30.74, 30.82, 31.62],
    [32.22, 32.33, 32.34],
    [31.63, 31.99, 32.01],
    [31.96, 31.90, 31.94],
    [31.95, 32.30, 31.60],
    [33.20, 33.52, 33.20],
    [17.00, 16.91, 17.56],
    [20.32, 20.43, 20.54],
    [26.70, 27.05, 27.30]
])

IG_teorico = (1 / 12) * M * (L**2 + b**2)

# Distancia correspondiente al período mínimo teórico
ell_min_teorico = np.sqrt(IG_teorico / M)

# Período mínimo teórico
T_min_teorico = 2 * np.pi * np.sqrt(
    (IG_teorico + M * ell_min_teorico**2)
    / (M * g * ell_min_teorico)
)

def analizar_integrante(datos, nombre):

    # Cada tiempo corresponde a N oscilaciones
    periodos = datos / N[:, np.newaxis]

    # Promedio de las tres repeticiones
    T = np.mean(periodos, axis=1)

    # Desviación estándar
    s_T = np.std(periodos, axis=1, ddof=1)

    # Incertidumbre tipo A de la media
    u_T = s_T / np.sqrt(3)

    T2 = T**2
    ell2 = ell**2

    # Momento de inercia experimental
    I1 = (M * g * ell * T2) / (4 * np.pi**2)

    u_T2 = 2 * T * u_T
    u_ell2 = 2 * ell * u_ell

    # Propagación para I1 considerando M y g constantes
    u_I1 = I1 * np.sqrt(
        (u_ell / ell)**2
        +
        (2 * u_T / T)**2
    )

    coeficientes, covarianza = np.polyfit(
        ell2,
        I1,
        1,
        cov=True
    )

    M_exp = coeficientes[0]
    IG_exp = coeficientes[1]

    u_M_exp = np.sqrt(covarianza[0, 0])
    u_IG_exp = np.sqrt(covarianza[1, 1])

    I1_ajuste = M_exp * ell2 + IG_exp

    # Coeficiente de determinación R²
    SS_res = np.sum((I1 - I1_ajuste)**2)
    SS_tot = np.sum((I1 - np.mean(I1))**2)

    R2 = 1 - SS_res / SS_tot

    error_M = abs(M_exp - M) / M * 100

    error_IG = (
        abs(IG_exp - IG_teorico)
        / IG_teorico
        * 100
    )

    indice_min = np.argmin(T)

    ell_min_medido = ell[indice_min]
    T_min_medido = T[indice_min]

    izquierda = np.where(ell < ell_min_teorico)[0]
    derecha = np.where(ell > ell_min_teorico)[0]

    mejor_diferencia = np.inf
    mejor_par = None

    for i in izquierda:
        for j in derecha:

            diferencia = abs(T[i] - T[j])

            if diferencia < mejor_diferencia:
                mejor_diferencia = diferencia
                mejor_par = (i, j)

    i_igual, j_igual = mejor_par

    tabla = pd.DataFrame({
        "ell (m)": ell,
        "T (s)": T,
        "u_T (s)": u_T,
        "T² (s²)": T2,
        "I1 (kg m²)": I1,
        "u_I1 (kg m²)": u_I1,
        "ell² (m²)": ell2
    })

    print("\n")
    print("=" * 78)
    print(nombre.upper())
    print("=" * 78)

    print(
        tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n--- PERÍODO MÍNIMO ---")

    print(
        f"ell mínimo entre puntos medidos = "
        f"{ell_min_medido:.6f} m"
    )

    print(
        f"T mínimo entre puntos medidos   = "
        f"{T_min_medido:.6f} s"
    )

    print("\n--- PUNTOS CON PERÍODO APROXIMADAMENTE IGUAL ---")

    print(
        f"ell_1 = {ell[i_igual]:.6f} m, "
        f"T_1 = {T[i_igual]:.6f} s"
    )

    print(
        f"ell_2 = {ell[j_igual]:.6f} m, "
        f"T_2 = {T[j_igual]:.6f} s"
    )

    print(
        f"|T_1 - T_2| = "
        f"{abs(T[i_igual] - T[j_igual]):.6f} s"
    )

    print("\n--- AJUSTE LINEAL DE STEINER ---")

    print(
        f"M experimental  = "
        f"{M_exp:.6f} ± {u_M_exp:.6f} kg"
    )

    print(
        f"IG experimental = "
        f"{IG_exp:.6f} ± {u_IG_exp:.6f} kg m²"
    )

    print(f"R²              = {R2:.6f}")

    print(
        f"Error masa       = "
        f"{error_M:.2f} %"
    )

    print(
        f"Error IG         = "
        f"{error_IG:.2f} %"
    )

    return {
        "T": T,
        "u_T": u_T,
        "T2": T2,
        "u_T2": u_T2,
        "ell2": ell2,
        "u_ell2": u_ell2,
        "I1": I1,
        "u_I1": u_I1,
        "M_exp": M_exp,
        "u_M_exp": u_M_exp,
        "IG_exp": IG_exp,
        "u_IG_exp": u_IG_exp,
        "R2": R2,
        "error_M": error_M,
        "error_IG": error_IG,
        "ell_min_medido": ell_min_medido,
        "T_min_medido": T_min_medido,
        "i_igual": i_igual,
        "j_igual": j_igual
    }


r1 = analizar_integrante(
    datos_1,
    "Integrante 1"
)

r2 = analizar_integrante(
    datos_2,
    "Integrante 2"
)

r3 = analizar_integrante(
    datos_3,
    "Integrante 3"
)

print("\n")
print("=" * 78)
print("RESULTADOS TEÓRICOS")
print("=" * 78)

print(
    f"I_G teórico        = "
    f"{IG_teorico:.6f} kg m²"
)

print(
    f"ell mínimo teórico = "
    f"{ell_min_teorico:.6f} m"
)

print(
    f"T mínimo teórico   = "
    f"{T_min_teorico:.6f} s"
)

orden = np.argsort(ell)

ell_g = ell[orden]
T_g = r1["T"][orden]
u_T_g = r1["u_T"][orden]

fig, ax = plt.subplots(
    figsize=(7.2, 5.0)
)

# Datos experimentales
ax.errorbar(
    ell_g,
    T_g,
    yerr=u_T_g,
    fmt="o",
    capsize=4,
    label="Datos experimentales"
)

# Curva teórica
ell_curva = np.linspace(
    0.04,
    0.55,
    500
)

T_teorico = 2 * np.pi * np.sqrt(
    (
        IG_teorico
        +
        M * ell_curva**2
    )
    /
    (
        M
        *
        g
        *
        ell_curva
    )
)

ax.plot(
    ell_curva,
    T_teorico,
    linewidth=1.5,
    label="Modelo teórico"
)

# Mínimo teórico
ax.axvline(
    ell_min_teorico,
    linestyle="--",
    linewidth=1,
    label=(
        rf"$\ell_{{\min}}="
        rf"{ell_min_teorico:.3f}\,\mathrm{{m}}$"
    )
)

ax.set_xlabel(
    r"Distancia al centro de masa, $\ell$ (m)"
)

ax.set_ylabel(
    r"Período, $T$ (s)"
)

ax.set_title(
    r"Período de oscilación en función de $\ell$"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURAS / "periodo_vs_distancia_integrante1.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIGURAS / "periodo_vs_distancia_integrante1.pdf",
    bbox_inches="tight"
)

plt.close(fig)

x = r1["ell2"]
y = r1["I1"]
uy = r1["u_I1"]

orden2 = np.argsort(x)

x = x[orden2]
y = y[orden2]
uy = uy[orden2]

x_ajuste = np.linspace(
    0,
    max(x) * 1.05,
    300
)

y_ajuste = (
    r1["M_exp"] * x_ajuste
    +
    r1["IG_exp"]
)

fig, ax = plt.subplots(
    figsize=(7.2, 5.0)
)

# Datos experimentales
ax.errorbar(
    x,
    y,
    yerr=uy,
    fmt="o",
    capsize=4,
    label="Datos experimentales"
)

# Ajuste lineal
ax.plot(
    x_ajuste,
    y_ajuste,
    linewidth=1.5,
    label="Ajuste lineal"
)

# Texto de la regresión
texto = (
    rf"$I_1 = "
    rf"({r1['M_exp']:.4f})\ell^2"
    rf" + ({r1['IG_exp']:.4f})$"
    "\n"
    rf"$R^2={r1['R2']:.5f}$"
)

ax.text(
    0.05,
    0.93,
    texto,
    transform=ax.transAxes,
    verticalalignment="top"
)

ax.set_xlabel(
    r"$\ell^2$ (m$^2$)"
)

ax.set_ylabel(
    r"$I_1$ (kg m$^2$)"
)

ax.set_title(
    r"Momento de inercia en función de $\ell^2$"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURAS / "inercia_vs_distancia2_integrante1.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIGURAS / "inercia_vs_distancia2_integrante1.pdf",
    bbox_inches="tight"
)

plt.close(fig)

print("\n")
print("=" * 78)
print("RESUMEN DE LOS TRES INTEGRANTES")
print("=" * 78)

print("\nINTEGRANTE 1")
print(
    f"M_exp  = {r1['M_exp']:.6f} ± "
    f"{r1['u_M_exp']:.6f} kg"
)
print(
    f"IG_exp = {r1['IG_exp']:.6f} ± "
    f"{r1['u_IG_exp']:.6f} kg m²"
)
print(f"R²     = {r1['R2']:.6f}")

print("\nINTEGRANTE 2")
print(
    f"M_exp  = {r2['M_exp']:.6f} ± "
    f"{r2['u_M_exp']:.6f} kg"
)
print(
    f"IG_exp = {r2['IG_exp']:.6f} ± "
    f"{r2['u_IG_exp']:.6f} kg m²"
)
print(f"R²     = {r2['R2']:.6f}")

print("\nINTEGRANTE 3")
print(
    f"M_exp  = {r3['M_exp']:.6f} ± "
    f"{r3['u_M_exp']:.6f} kg"
)
print(
    f"IG_exp = {r3['IG_exp']:.6f} ± "
    f"{r3['u_IG_exp']:.6f} kg m²"
)
print(f"R²     = {r3['R2']:.6f}")


print("\n")
print("=" * 78)
print("ARCHIVOS GENERADOS")
print("=" * 78)

print(
    FIGURAS
    / "periodo_vs_distancia_integrante1.png"
)

print(
    FIGURAS
    / "periodo_vs_distancia_integrante1.pdf"
)

print(
    FIGURAS
    / "inercia_vs_distancia2_integrante1.png"
)

print(
    FIGURAS
    / "inercia_vs_distancia2_integrante1.pdf"
)

def generar_graficas(resultado, numero_integrante):

    orden = np.argsort(ell)

    ell_g = ell[orden]
    T_g = resultado["T"][orden]
    u_T_g = resultado["u_T"][orden]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.errorbar(
        ell_g,
        T_g,
        yerr=u_T_g,
        fmt="o",
        capsize=4,
        label="Datos experimentales"
    )

    # Curva teórica
    ell_curva = np.linspace(0.04, 0.55, 500)

    T_teorico = 2 * np.pi * np.sqrt(
        (IG_teorico + M * ell_curva**2)
        /
        (M * g * ell_curva)
    )

    ax.plot(
        ell_curva,
        T_teorico,
        linewidth=1.5,
        label="Modelo teórico"
    )

    ax.axvline(
        ell_min_teorico,
        linestyle="--",
        linewidth=1,
        label=(
            rf"$\ell_{{\min}}="
            rf"{ell_min_teorico:.3f}\,\mathrm{{m}}$"
        )
    )

    ax.set_xlabel(
        r"Distancia al centro de masa, $\ell$ (m)"
    )

    ax.set_ylabel(
        r"Período, $T$ (s)"
    )

    ax.set_title(
        rf"Período de oscilación en función de $\ell$"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURAS /
        f"periodo_vs_distancia_integrante{numero_integrante}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURAS /
        f"periodo_vs_distancia_integrante{numero_integrante}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    x = resultado["ell2"]
    y = resultado["I1"]
    uy = resultado["u_I1"]

    orden = np.argsort(x)

    x = x[orden]
    y = y[orden]
    uy = uy[orden]

    x_ajuste = np.linspace(
        0,
        max(x) * 1.05,
        300
    )

    y_ajuste = (
        resultado["M_exp"] * x_ajuste
        +
        resultado["IG_exp"]
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.errorbar(
        x,
        y,
        yerr=uy,
        fmt="o",
        capsize=4,
        label="Datos experimentales"
    )

    ax.plot(
        x_ajuste,
        y_ajuste,
        linewidth=1.5,
        label="Ajuste lineal"
    )

    texto = (
        rf"$I_1="
        rf"({resultado['M_exp']:.4f})\ell^2"
        rf"+({resultado['IG_exp']:.4f})$"
        "\n"
        rf"$R^2={resultado['R2']:.5f}$"
    )

    ax.text(
        0.05,
        0.93,
        texto,
        transform=ax.transAxes,
        verticalalignment="top"
    )

    ax.set_xlabel(
        r"$\ell^2$ (m$^2$)"
    )

    ax.set_ylabel(
        r"$I_1$ (kg m$^2$)"
    )

    ax.set_title(
        r"Momento de inercia en función de $\ell^2$"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURAS /
        f"inercia_vs_distancia2_integrante{numero_integrante}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURAS /
        f"inercia_vs_distancia2_integrante{numero_integrante}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

generar_graficas(r2, 2)
generar_graficas(r3, 3)

print("\nGráficas de los integrantes 2 y 3 generadas correctamente.")
