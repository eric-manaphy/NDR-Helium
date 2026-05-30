import numpy as np
from scipy.linalg import eigh

def build_density(C, occ):
    return C @ np.diag(occ) @ C.T

def group_basis_by_lm(basis):
    lm_indices = {}
    for i, bf in enumerate(basis):
        lm_indices.setdefault((bf.l, bf.m), []).append(i)

    l_values = sorted(set(bf.l for bf in basis))
    radial_indices = {}
    for l in l_values:
        radial_indices[l] = lm_indices[(l, 0)]
    return l_values, radial_indices, lm_indices

def fractional_occupations(eps, n_electrons, degeneracy_tol=1e-8):
    n = len(eps)
    occ = np.zeros(n)
    idx_sorted = np.argsort(eps)
    eps_sorted = eps[idx_sorted]
    remaining_electrons = n_electrons

    i = 0
    while remaining_electrons > 0 and i < n:
        j = i + 1
        while j < n and abs(eps_sorted[j] - eps_sorted[i]) < degeneracy_tol:
            j += 1
        block = idx_sorted[i:j]
        block_size = len(block)
        block_occupancy = 2.0 * block_size

        if remaining_electrons >= block_occupancy:
            occ[block] = 2.0
            remaining_electrons -= block_occupancy
        else:
            occ[block] = remaining_electrons / block_size
            remaining_electrons = 0.0
        i = j
    return occ

def scf(basis, Z, N_elec, max_iter=150, conv=1e-7, damping=0.3):
    from src.integrals.one_electron import S_matrix, H_matrix
    from src.integrals.two_electron import build_eri_tensor, build_JK

    n = len(basis)
    S = S_matrix(basis)
    H_core = H_matrix(basis, Z)
    eri = build_eri_tensor(basis)
    
    l_values, radial_indices, lm_indices = group_basis_by_lm(basis)

    # Initial Guess
    C_full = np.zeros((n, n))
    eps_full = np.zeros(n)
    col = 0
    for l in l_values:
        idx_rad = radial_indices[l]
        eps_l, C_l = eigh(H_core[np.ix_(idx_rad, idx_rad)], S[np.ix_(idx_rad, idx_rad)])
        for a in range(len(eps_l)):
            for m in range(-l, l + 1):
                C_full[lm_indices[(l, m)], col] = C_l[:, a]
                eps_full[col] = eps_l[a]
                col += 1

    D = build_density(C_full, fractional_occupations(eps_full, N_elec))
    E_old = 0.0

    for it in range(1, max_iter + 1):
        J, K = build_JK(eri, D)
        F = H_core + J + K

        C_new_full = np.zeros((n, n))
        eps_new_full = np.zeros(n)
        col = 0
        for l in l_values:
            idx_rad = radial_indices[l]
            eps_l, C_l = eigh(F[np.ix_(idx_rad, idx_rad)], S[np.ix_(idx_rad, idx_rad)])
            for a in range(len(eps_l)):
                for m in range(-l, l + 1):
                    C_new_full[lm_indices[(l, m)], col] = C_l[:, a]
                    eps_new_full[col] = eps_l[a]
                    col += 1

        occ = fractional_occupations(eps_new_full, N_elec)
        D_trial = build_density(C_new_full, occ)
        D_new = (1 - damping) * D + damping * D_trial

        # Energy calculation
        J_final, K_final = build_JK(eri, D_new)
        one_e = np.trace(D_new @ H_core)
        two_e = 0.5 * np.trace(D_new @ (J_final + K_final))
        E_tot = one_e + two_e
        
        if abs(E_tot - E_old) < conv:
            print(f"1e: {one_e}")
            print(f"2e: {two_e}")
            sort_idx = np.argsort(eps_new_full)
            eps_new_full = eps_new_full[sort_idx]
            C_new_full = C_new_full[:, sort_idx]
            return {
                "E_total": E_tot,
                "orb_energies": eps_new_full,
                "density": D_new,
                "Fock": F,
                "coefficients": C_new_full,
                "H_core": H_core,
                "n_elec": N_elec,
                "eri": eri
            }

        D, E_old = D_new, E_tot

    sort_idx = np.argsort(eps_new_full)
    C_sorted = C_new_full[:, sort_idx]
    eps_sorted = eps_new_full[sort_idx]

    return {
        "E_total": E_tot, 
        "density": D, 
        "orb_energies": eps_sorted,
        "coefficients": C_sorted,
        "H_core": H_core, 
        "eri": eri, 
        "n_elec": N_elec,
        "converged": False
    }