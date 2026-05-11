import numpy as np

def build_param_maps(basis):
    # Local import to avoid circular dependency
    from src.solver.hf_scf import group_basis_by_lm

    l_values, radial_indices, lm_indices = group_basis_by_lm(basis)
    n = len(basis)
    ao_to_param = np.full(n, -1, dtype=int)
    param_to_aos = []

    p_index = 0
    for l in l_values:
        n_rad = len(radial_indices[l])
        for a in range(n_rad):
            aos = []
            for m in range(-l, l + 1):
                idx_list = lm_indices[(l, m)]
                mu = idx_list[a]   # AO index for this radial shell and m
                aos.append(mu)
                ao_to_param[mu] = p_index
            param_to_aos.append(aos)
            p_index += 1

    return l_values, radial_indices, lm_indices, param_to_aos, ao_to_param

def pack_params(*zeta_lists):
    flat = []
    for zl in zeta_lists:
        if zl is not None and len(zl) > 0:
            flat.extend(zl)
    return np.log(np.array(flat, dtype=float))

def unpack_params(x_vector, counts):
    zetas = np.exp(x_vector)
    zeta_lists = []
    cursor = 0
    for count in counts:
        zeta_lists.append(zetas[cursor : cursor + count])
        cursor += count
    return zeta_lists

def ao_to_mo_transform(eri_ao, C):

    n = C.shape[0]
    tmp = np.dot(C.T, eri_ao.reshape(n, -1)).reshape(n, n, n, n)
    
    tmp = tmp.transpose(1, 0, 2, 3).reshape(n, -1)
    tmp = np.dot(C.T, tmp).reshape(n, n, n, n)
    
    tmp = tmp.transpose(2, 0, 1, 3).reshape(n, -1)
    tmp = np.dot(C.T, tmp).reshape(n, n, n, n)
    
    tmp = tmp.transpose(3, 0, 1, 2).reshape(n, -1)
    tmp = np.dot(C.T, tmp).reshape(n, n, n, n)
    

    return tmp.transpose(2, 3, 0, 1)

def build_spin_orbital_integrals(h_mo, eri_mo):

    n_mo = h_mo.shape[0]
    n_spin = 2 * n_mo
    h_spin = np.zeros((n_spin, n_spin))
    
    for p in range(n_mo):
        for q in range(n_mo):
            h_spin[2*p, 2*q] = h_spin[2*p+1, 2*q+1] = h_mo[p, q]

    g_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    val = eri_mo[p, r, q, s]
            
                    g_spin[2*p, 2*q, 2*r, 2*s] = val
                    g_spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = val

                    g_spin[2*p, 2*q+1, 2*r, 2*s+1] = val
                    g_spin[2*p+1, 2*q, 2*r+1, 2*s] = val
                    
    return h_spin, g_spin

def get_excitation_level(det, ref_det):
    '''
    Return 0 for C_0, 1 for Single, 2 for Doubles, etc.
    Each excitation moves one electron from an occupied to a virtual orbital,
    changing two bits in the determinant.
    '''
    return (det ^ ref_det).bit_count() // 2