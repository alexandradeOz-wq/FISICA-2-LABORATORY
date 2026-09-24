import numpy as np

# ============================================================
# CÁLCULO DEL MOMENTO DE INERCIA
# Sistema disco--anillo
# ============================================================

# Aceleración de la gravedad
g = 9.81  # m/s²


# ============================================================
# 1. DATOS EXPERIMENTALES DEL SISTEMA
# ============================================================

m = 0.2017          # kg
dm = 0.0001         # kg

d = 0.01310         # m
dd = 0.00005        # m

a = 0.00494258      # m/s²
da = 0.00003125     # m/s²


# ============================================================
# 2. DATOS DEL DISCO
# ============================================================

M_D = 1.4544        # kg
dM_D = 0.0001       # kg

R_D = 0.11430       # m
dR_D = 0.00050      # m


# ============================================================
# 3. DATOS DEL ANILLO
# ============================================================

M_A = 1.4466        # kg
dM_A = 0.0001       # kg

R_i = 0.053575      # m
dR_i = 0.000050     # m

R_e = 0.063825      # m
dR_e = 0.000050     # m


# ============================================================
# 4. MOMENTO DE INERCIA EXPERIMENTAL
#
# I = m d² (g-a) / (4a)
# ============================================================

I_exp = m * d**2 * (g - a) / (4 * a)


# ============================================================
# 5. INCERTIDUMBRE DE I EXPERIMENTAL
# ============================================================

dI_dm = d**2 * (g - a) / (4 * a)

dI_dd = m * d * (g - a) / (2 * a)

dI_da = -m * d**2 * g / (4 * a**2)

delta_I_exp = np.sqrt(
    (dI_dm * dm)**2
    + (dI_dd * dd)**2
    + (dI_da * da)**2
)


# ============================================================
# 6. MOMENTO DE INERCIA TEÓRICO DEL DISCO
#
# I_D = (1/2) M R²
# ============================================================

I_D = 0.5 * M_D * R_D**2

dI_D_dM = 0.5 * R_D**2
dI_D_dR = M_D * R_D

delta_I_D = np.sqrt(
    (dI_D_dM * dM_D)**2
    + (dI_D_dR * dR_D)**2
)


# ============================================================
# 7. MOMENTO DE INERCIA TEÓRICO DEL ANILLO
#
# I_A = (1/2) M (Ri² + Re²)
# ============================================================

I_A = 0.5 * M_A * (R_i**2 + R_e**2)

dI_A_dM = 0.5 * (R_i**2 + R_e**2)
dI_A_dRi = M_A * R_i
dI_A_dRe = M_A * R_e

delta_I_A = np.sqrt(
    (dI_A_dM * dM_A)**2
    + (dI_A_dRi * dR_i)**2
    + (dI_A_dRe * dR_e)**2
)


# ============================================================
# 8. MOMENTO DE INERCIA TEÓRICO TOTAL
# ============================================================

I_teo = I_D + I_A

delta_I_teo = np.sqrt(
    delta_I_D**2
    + delta_I_A**2
)


# ============================================================
# 9. DISCREPANCIA PORCENTUAL
# ============================================================

discrepancia = abs(I_exp - I_teo) / I_teo * 100


# ============================================================
# 10. DISCREPANCIA NORMALIZADA
#
# Indica cuántas incertidumbres combinadas separan
# el resultado experimental del teórico.
# ============================================================

z = abs(I_exp - I_teo) / np.sqrt(
    delta_I_exp**2 + delta_I_teo**2
)


# ============================================================
# 11. RESULTADOS
# ============================================================

print("\n==============================================")
print("      RESULTADOS DEL MOMENTO DE INERCIA")
print("==============================================")

print("\nEXPERIMENTAL: SISTEMA DISCO--ANILLO")
print(f"I_exp  = {I_exp:.8f} kg·m²")
print(f"ΔI_exp = {delta_I_exp:.8f} kg·m²")
print(
    f"I_exp  = ({I_exp:.8f} ± "
    f"{delta_I_exp:.8f}) kg·m²"
)

print("\nTEÓRICO: DISCO")
print(f"I_D    = {I_D:.8f} kg·m²")
print(f"ΔI_D   = {delta_I_D:.8f} kg·m²")

print("\nTEÓRICO: ANILLO")
print(f"I_A    = {I_A:.8f} kg·m²")
print(f"ΔI_A   = {delta_I_A:.8f} kg·m²")

print("\nTEÓRICO: SISTEMA DISCO--ANILLO")
print(f"I_teo  = {I_teo:.8f} kg·m²")
print(f"ΔI_teo = {delta_I_teo:.8f} kg·m²")
print(
    f"I_teo  = ({I_teo:.8f} ± "
    f"{delta_I_teo:.8f}) kg·m²"
)

print("\nCOMPARACIÓN")
print(f"Discrepancia porcentual = {discrepancia:.2f} %")
print(f"Discrepancia normalizada z = {z:.2f}")

print("\n==============================================")