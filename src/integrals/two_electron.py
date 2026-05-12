import numpy as np
from scipy.special import gamma
from sympy.physics.wigner import wigner_3j
from math import factorial
from functools import lru_cache

@lru_cache(maxsize=None)
def Y_matrix_element(li, mi, L, M, lj, mj):
    """Calculates the angular coupling element using Wigner-3j symbols."""
    if (-mi + M + mj) != 0:
        return 0.0
    
    # Selection rule for l: triangle inequality and parity
    if not (abs(li - lj) <= L <= li + lj) or (li + L + lj) % 2 != 0:
        return 0.0

    phase = (-1)**mi
    pref = np.sqrt((2*li + 1)*(2*L + 1)*(2*lj + 1) / (4*np.pi))

    return float(
        phase
        * pref
        * wigner_3j(li, L, lj, 0, 0, 0)
        * wigner_3j(li, L, lj, -mi, M, mj)
    )

def angular_part(bi, bj, bk, bl, L):
    """Sum over M for the angular part of the ERI."""
    s = 0.0
    # M must satisfy selection rules for both pairs
    for M in range(-L, L+1):
        a = Y_matrix_element(bi.l, bi.m, L, M, bj.l, bj.m)
        b = Y_matrix_element(bk.l, bk.m, L, M, bl.l, bl.m)
        if a != 0 and b != 0:
            s += a * np.conjugate(b)
    return (4 * np.pi / (2 * L + 1)) * float(s)

def radial_repulsion_integral(bi, bj, bk, bl, L, deriv=False):

    a_pow = bi.l + bj.l + 2
    b_pow = bk.l + bl.l + 2
    p = bi.zeta + bj.zeta
    q = bk.zeta + bl.zeta

    # --- Part 1: I1 ---
    n1 = a_pow + L
    pref1 = factorial(n1) / p**(n1 + 1)
    term0_1 = gamma(b_pow - L) / q**(b_pow - L)

    sum1, dsum1_dp, dsum1_dq = 0.0, 0.0, 0.0
    for k in range(0, n1 + 1):
        nu = b_pow - L + k
        base = (p**k / factorial(k)) * gamma(nu) / (p + q)**nu
        sum1 += base
        if deriv:
            dbase_dp = base * (k/p - nu/(p+q)) if p != 0 else -base * nu/(p+q)
            dbase_dq = base * (-nu/(p+q))
            dsum1_dp += dbase_dp
            dsum1_dq += dbase_dq

    I1 = pref1 * (term0_1 - sum1)

    # --- Part 2: I2 ---
    n2 = a_pow - L - 1
    pref2 = factorial(n2) / p**(a_pow - L)

    sum2, dsum2_dp, dsum2_dq = 0.0, 0.0, 0.0
    for k in range(0, n2 + 1):
        nu = b_pow + L + k + 1
        base = (p**k / factorial(k)) * gamma(nu) / (p + q)**nu
        sum2 += base
        if deriv:
            dbase_dp = base * (k/p - nu/(p+q)) if p != 0 else -base * nu/(p+q)
            dbase_dq = base * (-nu/(p+q))
            dsum2_dp += dbase_dp
            dsum2_dq += dbase_dq

    I2 = pref2 * sum2
    R = I1 + I2

    if not deriv:
        return R

    # Derivatives of prefactors and terms
    dI1_dp = (pref1 * (-(n1+1)/p)) * (term0_1 - sum1) - pref1 * dsum1_dp
    dI1_dq = pref1 * (term0_1 * (-(b_pow-L)/q) - dsum1_dq)
    dI2_dp = (pref2 * (-(a_pow-L)/p)) * sum2 + pref2 * dsum2_dp
    dI2_dq = pref2 * dsum2_dq

    return R, (dI1_dp + dI2_dp), (dI1_dq + dI2_dq)

def electron_repulsion_integral(bi, bj, bk, bl, deriv=False):
    Lmin = max(abs(bi.l - bj.l), abs(bk.l - bl.l))
    Lmax = min(bi.l + bj.l, bk.l + bl.l)
    
    total_val = 0.0
    total_grad_bi = 0.0 # dERI / d_zeta_i
    
    for L in range(Lmin, Lmax + 1):
        if (bi.l + bj.l + L) % 2 == 0 and (bk.l + bl.l + L) % 2 == 0:
            A_L = angular_part(bi, bj, bk, bl, L)
            if abs(A_L) > 1e-15:
                if deriv:
                    # Capture the radial derivatives you already implemented
                    R_L, dR_dp, dR_dq = radial_repulsion_integral(bi, bj, bk, bl, L, deriv=True)
                    total_val += A_L * R_L
                    # Since p = zeta_i + zeta_j, dR/d_zeta_i = dR/dp
                    total_grad_bi += A_L * dR_dp
                else:
                    R_L = radial_repulsion_integral(bi, bj, bk, bl, L, deriv=False)
                    total_val += A_L * R_L
                    
    if deriv:
        return float(total_val), float(total_grad_bi)
    return float(total_val)

def build_eri_tensor(basis):
    n = len(basis)
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    eri[i, j, k, l] = electron_repulsion_integral(basis[i], basis[j], basis[k], basis[l])
    return eri

def build_JK(eri, D):
    # J_ij = sum_kl D_kl * (ij|kl)
    # K_ij = sum_kl D_kl * (ik|jl)
    J = np.einsum('kl,ijkl->ij', D, eri, optimize=True)
    K = np.einsum('kl,ikjl->ij', D, eri, optimize=True)
    return J, -0.5 * K # Note: Exchange factor handled in Fock build usually