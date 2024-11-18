# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:10:26 2024

@author: Gabriela Silva, Augusto C. Lima and Julio C. S. Da Silva
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import romberg

# Variables
# X1 andd X2 - Range of the orbitals energies evaluated  
Z = 0.0001
X1 = -0.25
X2 = -0.08
Y = 0.0005
W = 0.001

# Definition of the Transmission Function 
def F0(E, T1, T2, E1, Z):
    return (Y * Y * T1 * T2) / ((E - E1) ** 2 + (Y * T2 + Y * T1 + Z) ** 2)

# Variables (MO energies and Overlap contributions)

#GRAPH_pyrene-nonmodif
A0P1H1, B0P1H1, E0P1H1 = 0.38613, 0.22775, -0.14488
A0P1H0, B0P1H0, E0P1H0 = 0.01842, 0.96316, -0.14486
A0P1L0, B0P1L0, E0P1L0 = 0.05275, 0.89450, -0.12096
A0P1L1, B0P1L1, E0P1L1 = 0.18760, 0.62480, -0.12094

#1-NO2
A1P1H1, B1P1H1, E1P1H1 = 0.32416, 0.35167, -0.14738
A1P1H0, B1P1H0, E1P1H0 = 0.22821, 0.54359, -0.14482
A1P1L0, B1P1L0, E1P1L0 = 0.22006, 0.55987, -0.12345
A1P1L1, B1P1L1, E1P1L1 = 0.49287, 0.01426, -0.12090

#2-NO2
A2P1H1, B2P1H1, E2P1H1 = 0.32826, 0.34348, -0.14797
A2P1H0, B2P1H0, E2P1H0 = 0.16999, 0.66003, -0.14482
A2P1L0, B2P1L0, E2P1L0 = 0.20054, 0.59892, -0.12404
A2P1L1, B2P1L1, E2P1L1 = 0.46704, 0.06593, -0.12090

#3-NO2
A3P1H1, B3P1H1, E3P1H1 = 0.13373, 0.73254, -0.14796
A3P1H0, B3P1H0, E3P1H0 = 0.41696, 0.16608, -0.14483
A3P1L0, B3P1L0, E3P1L0 = 0.32147, 0.35707, -0.12403
A3P1L1, B3P1L1, E3P1L1 = 0.48957, 0.02087, -0.12091

#4-NO2
A4P1H1, B4P1H1, E4P1H1 = 0.45454, 0.09093, -0.14745
A4P1H0, B4P1H0, E4P1H0 = 0.38421, 0.23158, -0.14466
A4P1L0, B4P1L0, E4P1L0 = 0.28570, 0.42859, -0.12346
A4P1L1, B4P1L1, E4P1L1 = 0.21770, 0.56460, -0.12082


# Definition of the Functions to compute T(E)

def F0T1(E):
    return (F0(E, A0P1H1, B0P1H1, E0P1H1, Z) +
            F0(E, A0P1H0, B0P1H0, E0P1H0, Z) +
            F0(E, A0P1L0, B0P1L0, E0P1L0, Z) +
            F0(E, A0P1L1, B0P1L1, E0P1L1, Z))

def F1T1(E):
    return (F0(E, A1P1H1, B1P1H1, E1P1H1, Z) +
            F0(E, A1P1H0, B1P1H0, E1P1H0, Z) +
            F0(E, A1P1L0, B1P1L0, E1P1L0, Z) +
            F0(E, A1P1L1, B1P1L1, E1P1L1, Z))

def F2T1(E):
    return (F0(E, A2P1H1, B2P1H1, E2P1H1, Z) +
            F0(E, A2P1H0, B2P1H0, E2P1H0, Z) +
            F0(E, A2P1L0, B2P1L0, E2P1L0, Z) +
            F0(E, A2P1L1, B2P1L1, E2P1L1, Z))

def F3T1(E):
    return (F0(E, A3P1H1, B3P1H1, E3P1H1, Z) +
            F0(E, A3P1H0, B3P1H0, E3P1H0, Z) +
            F0(E, A3P1L0, B3P1L0, E3P1L0, Z) +
            F0(E, A3P1L1, B3P1L1, E3P1L1, Z))

def F4T1(E):
    return (F0(E, A4P1H1, B4P1H1, E4P1H1, Z) +
            F0(E, A4P1H0, B4P1H0, E4P1H0, Z) +
            F0(E, A4P1L0, B4P1L0, E4P1L0, Z) +
            F0(E, A4P1L1, B4P1L1, E4P1L1, Z))



# Generation of points to DS1 variable
E_values = np.linspace(-0.25, -0.00, 1000)
DS1 = {
    "E": W * E_values,
    "F0T1": [F0T1(W * E) for E in E_values],
    "F1T1": [F1T1(W * E) for E in E_values],
    "F2T1": [F2T1(W * E) for E in E_values],
    "F3T1": [F3T1(W * E) for E in E_values],
    "F4T1": [F4T1(W * E) for E in E_values],
        
}


# Plot of graphics 
plt.plot(E_values, [F0T1(E) for E in E_values], label='GPG-NM', color=(0.1,0.1,0.8), linewidth=3)
plt.plot(E_values, [F1T1(E) for E in E_values], label='GPG-1NO_2', color=(0.2,0.5,0.5), linewidth=3)
plt.plot(E_values, [F2T1(E) for E in E_values], label='GPG-2NO_2', color=(0.8,0.3,0.1), linewidth=3)
plt.plot(E_values, [F3T1(E) for E in E_values], label='GPG-3NO_2', color=(0.8,0.7,0.2), linewidth=3)
plt.plot(E_values, [F4T1(E) for E in E_values], label='GPG-4NO_2', color=(0.6,0.3,0.3), linewidth=3)

plt.yscale('log')
plt.xlabel("Energia (a.u.)", fontsize=15, fontweight='bold')  # Definindo tamanho e negrito para o label do eixo x
plt.ylabel("Transmissão T(E)", fontsize=15, fontweight='bold')   # Definindo tamanho e negrito para o label do eixo y
#plt.title("Função de Transmissão")
plt.legend()

# 
plt.savefig('transmission_GPG_NO2.tiff', format='tiff', dpi=600)

plt.show()

# Fermi energy level (Hartree)
EF0 = -0.132910
EF1 = -0.134135
EF2 = -0.134430
EF3 = -0.134430
EF4 = -0.134060

# Conductance G/G0 = T(Efermi)
print("CONDUTÂNCIA:")
print(" GPG-NM",F0T1(EF0), "\n GPG-1NO_2",F1T1(EF1), "\n GPG-2NO_2",F2T1(EF2), "\n GPG-3NO_2",F3T1(EF3), "\n GPG-4NO_2",F4T1(EF4))
print()

# Electric Current
def FAT0(E, V): 
   return (1 / (1 + np.exp(-(E - (EF0 + 0.5 * (1.9 * V) * 0.0368)) / 0.00094))) - (1 / (1 + np.exp(-(E - (EF0 - 0.5 * (1.9 * V) * 0.0368)) / 0.00094)))

def FAT1(E, V): 
   return (1 / (1 + np.exp(-(E - (EF1 + 0.5 * (1.9 * V) * 0.0368)) / 0.00094))) - (1 / (1 + np.exp(-(E - (EF1 - 0.5 * (1.9 * V) * 0.0368)) / 0.00094)))

def FAT2(E, V): 
   return (1 / (1 + np.exp(-(E - (EF2 + 0.5 * (1.9 * V) * 0.0368)) / 0.00094))) - (1 / (1 + np.exp(-(E - (EF2 - 0.5 * (1.9 * V) * 0.0368)) / 0.00094)))

def FAT3(E, V): 
   return (1 / (1 + np.exp(-(E - (EF3 + 0.5 * (1.9 * V) * 0.0368)) / 0.00094))) - (1 / (1 + np.exp(-(E - (EF3 - 0.5 * (1.9 * V) * 0.0368)) / 0.00094)))

def FAT4(E, V): 
   return (1 / (1 + np.exp(-(E - (EF4 + 0.5 * (1.9 * V) * 0.0368)) / 0.00094))) - (1 / (1 + np.exp(-(E - (EF4 - 0.5 * (1.9 * V) * 0.0368)) / 0.00094)))


DC0 = [
    [0.02 * V, 10 * romberg(lambda E: -662361.795 * FAT0(E, 0.02 * V) * F0T1(E), -0.16, -0.14)]
    for V in range(71)
]

DC1 = [
    [0.02 * V, 10 * romberg(lambda E: -662361.795 * FAT1(E, 0.02 * V) * F1T1(E), -0.16, -0.14)]
    for V in range(71)
]

DC2 = [
    [0.02 * V, 10 * romberg(lambda E: -662361.795 * FAT2(E, 0.02 * V) * F2T1(E), -0.16, -0.14)]
    for V in range(71)
]

DC3 = [
    [0.02 * V, 10 * romberg(lambda E: -662361.795 * FAT3(E, 0.02 * V) * F3T1(E), -0.16, -0.14)]
    for V in range(71)
]

DC4 = [
    [0.02 * V, 10 * romberg(lambda E: -662361.795 * FAT4(E, 0.02 * V) * F4T1(E), -0.16, -0.14)]
    for V in range(71)
]


# Graphic of the electric current

plt.plot([point[0] for point in DC0], [point[1] for point in DC0], marker='o', linestyle='-', label='GPG-NM', color=(0.1,0.1,0.8))
plt.plot([point[0] for point in DC1], [point[1] for point in DC1], marker='o', linestyle='-', label='GPG-1NO_2', color=(0.2,0.5,0.5))
plt.plot([point[0] for point in DC2], [point[1] for point in DC2], marker='o', linestyle='-', label='GPG-2NO_2', color=(0.8,0.3,0.1))
plt.plot([point[0] for point in DC3], [point[1] for point in DC3], marker='o', linestyle='-', label='GPG-3NO_2', color=(0.8,0.7,0.2))
plt.plot([point[0] for point in DC4], [point[1] for point in DC4], marker='o', linestyle='-', label='GPG-4NO_2', color=(0.6,0.3,0.3))

#plt.title('Corrente Elétrica')

plt.xlabel('Potencial (V)', fontsize=15, fontweight='bold')
plt.ylabel('Corrente (nA)', fontsize=15, fontweight='bold')
plt.ylim([0, 2500])
plt.xlim([0, 1])

plt.legend()

plt.grid(True)

# Salvando o gráfico como TIFF com resolução de 600 DPI
plt.savefig('Corrente_GPG_NO2.tiff', format='tiff', dpi=600)
plt.show()

# Seebeck Coefficient

L = 2.4410e-8  
e = 1.60217663410e-19
E = sp.symbols('E')

# Transformando função númerica em função simbólica
F0T1_sym = sp.Lambda(E, F0T1(E))
F1T1_sym = sp.Lambda(E, F1T1(E))
F2T1_sym = sp.Lambda(E, F2T1(E))
F3T1_sym = sp.Lambda(E, F3T1(E))
F4T1_sym = sp.Lambda(E, F4T1(E))

# Derivative of T(E) 
df0_sym = sp.diff(F0T1_sym(E),E)
df1_sym = sp.diff(F1T1_sym(E),E)
df2_sym = sp.diff(F2T1_sym(E),E)
df3_sym = sp.diff(F3T1_sym(E),E)
df4_sym = sp.diff(F4T1_sym(E),E)

#Transformando função simbólica em função númerica 
df0 = sp.lambdify(E, df0_sym, 'numpy')
df1 = sp.lambdify(E, df1_sym, 'numpy')
df2 = sp.lambdify(E, df2_sym, 'numpy')
df3 = sp.lambdify(E, df3_sym, 'numpy')
df4 = sp.lambdify(E, df4_sym, 'numpy')


# Seebeck Coefficient
def S0(T):
    return (-L) * e * T * (1/F0T1(EF0)) * (1/4.35e-18*df0(EF0))

def S1(T):
    return (-L) * e * T * (1/F1T1(EF1)) * (1/4.35e-18*df1(EF1))

def S2(T):
    return (-L) * e * T * (1/F2T1(EF2)) * (1/4.35e-18*df2(EF2))

def S3(T):
    return (-L) * e * T * (1/F3T1(EF3)) * (1/4.35e-18*df3(EF3))

def S4(T):
    return (-L) * e * T * (1/F4T1(EF4)) * (1/4.35e-18*df4(EF4))


print("COEFICIENTE DE SEEBACK:")
print(" GPG-NM:",S0(30),"\n GPG-1NO_2:", S1(30),"\n GPG-2NO_2:", S2(30),"\n GPG-3NO_2:", S3(30),"\n GPG-4NO_2:", S4(30))
