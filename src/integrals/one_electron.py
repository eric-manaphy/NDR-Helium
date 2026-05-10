import numpy as np
from scipy.special import gamma

def overlap_integral(bi, bj, deriv=False):
    
    if bi.l != bj.l or bi.m != bj.m:
        return (0.0, 0.0, 0.0) if deriv else 0.0
    
    l, z1, z2 = bi.l, bi.zeta, bj.zeta
    a = z1 + z2
    g3 = gamma(2*l + 3)
    s = g3 / a**(2*l + 3)
    
    if not deriv:
        return s
    
    ds_da = -(2*l + 3) * g3 / a**(2*l + 4)
    return s, ds_da, ds_da

def potential_integral(bi, bj, Z, deriv=False):

    if bi.l != bj.l or bi.m != bj.m:
        return (0.0, 0.0, 0.0) if deriv else 0.0
        
    l, z1, z2 = bi.l, bi.zeta, bj.zeta
    a = z1 + z2
    g2 = gamma(2*l + 2)
    v = -Z * g2 / a**(2*l + 2)
    
    if not deriv:
        return v
        
    dv_da = Z * (2*l + 2) * g2 / a**(2*l + 3)
    return v, dv_da, dv_da

def kinetic_integral(bi, bj, deriv=False):
    
    if bi.l != bj.l or bi.m != bj.m:
        return (0.0, 0.0, 0.0) if deriv else 0.0

    l, z1, z2 = bi.l, bi.zeta, bj.zeta
    a = z1 + z2
    g1, g2, g3 = gamma(2*l + 3), gamma(2*l + 2), gamma(2*l + 1)

    term1 = z1 * z2 * g1 / a**(2*l + 3)
    term2 = l * g2 / a**(2*l + 1)
    term3 = l * (2*l + 1) * g3 / a**(2*l + 1)
    t = 0.5 * (term1 - term2 + term3)

    if not deriv:
        return t

    n1 = 2*l + 3
    dT1_dz1 = (z2 * g1 / a**n1) - (n1 * z1 * z2 * g1 / a**(n1 + 1))
    dT1_dz2 = (z1 * g1 / a**n1) - (n1 * z1 * z2 * g1 / a**(n1 + 1))
    
    dT2_da = l * g2 * (-(2*l + 1)) * a**(-(2*l + 2))
    dT3_da = l * (2*l + 1) * g3 * (-(2*l + 1)) * a**(-(2*l + 2))

    dt_dz1 = 0.5 * (dT1_dz1 - dT2_da + dT3_da)
    dt_dz2 = 0.5 * (dT1_dz2 - dT2_da + dT3_da)
    
    return t, dt_dz1, dt_dz2

# --- Matrix Builders ---

def S_matrix(basis_functions):
    n = len(basis_functions)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = overlap_integral(basis_functions[i], basis_functions[j])
            S[i, j] = S[j, i] = val
    return S

def T_matrix(basis_functions):
    n = len(basis_functions)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = kinetic_integral(basis_functions[i], basis_functions[j])
            T[i, j] = T[j, i] = val
    return T

def V_matrix(basis_functions, Z):
    n = len(basis_functions)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = potential_integral(basis_functions[i], basis_functions[j], Z)
            V[i, j] = V[j, i] = val
    return V

def H_matrix(basis, Z):
    return T_matrix(basis) + V_matrix(basis, Z)