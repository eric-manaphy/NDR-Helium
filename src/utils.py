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

# def ao_to_mo_transform(eri_ao, C):
#     aos = eri_ao.shape[0]
#     mos = C.shape[1]

#     scratch = np.zeros((aos, aos, aos, aos))
#     scratch2 = np.zeros((aos, aos, aos, aos))
#     eri_mo = np.zeros((aos, aos, aos, aos))

#     for m in range(aos):
#         for n in range(aos):
#             for k in range(aos):
#                 for l in range(aos):
#                     tmp = eri_ao[m, n, k, l]
#                     for p in range(mos):
#                         scratch[m,n,k,p] += tmp * C[l, p]
    
#     for m in range(aos):
#         for n in range(aos):
#             for k in range(aos):
#                 for p in range(mos):
#                     tmp = scratch[m,n,k,p]
#                     for q in range(mos):
#                         scratch2[m,n,q,p] += tmp * C[k, q]

#     scratch = np.zeros((aos, aos, aos, aos))

#     for m in range(aos):
#         for n in range(aos):
#             for p in range(mos):
#                 for q in range(mos):
#                     tmp = scratch2[m,n,p,q]
#                     for r in range(mos):
#                         scratch[m,r,p,q] += tmp * C[n, r]

#     for m in range(aos):
#         for p in range(mos):
#             for q in range(mos):
#                 for r in range(mos):
#                     tmp = scratch[m,p,q,r]
#                     for s in range(mos):
#                         eri_mo[s,p,q,r] += tmp * C[m, s]

#     return eri_mo

def ao_to_mo_transform(eri_ao, C):

    n = C.shape[0]
    tmp = np.dot(C.T, eri_ao.reshape(n, -1)).reshape(n, n, n, n)
    
    tmp = tmp.transpose(1, 0, 2, 3).reshape(n, -1)
    tmp = np.dot(C.T, tmp).reshape(n, n, n, n)
    
    tmp = tmp.transpose(2, 0, 1, 3).reshape(n, -1)
    tmp = np.dot(C.T, tmp).reshape(n, n, n, n)
    
    tmp = tmp.transpose(3, 0, 1, 2).reshape(n, -1)
    tmp = np.dot(C.T, tmp).reshape(n, n, n, n)
    

    return tmp.transpose(3, 2, 1, 0)

def build_spin_orbital_integrals(h_mo, eri_mo):

    n_mo = h_mo.shape[0]
    n_spin = 2 * n_mo
    h_spin = np.zeros((n_spin, n_spin))
    
    for p in range(n_mo):
        for q in range(n_mo):
            h_spin[2*p, 2*q] = h_spin[2*p+1, 2*q+1] = h_mo[p, q]

    g_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for p in range(n_mo):
        for r in range(n_mo):
            for q in range(n_mo):
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

def get_multi_index(i, j, a, b, occ, virt):
    '''
    Returns index of doubles CI coefficient vector.
    Requires four indices and number of occupied and virtual orbitals looped over
    '''
    occ_coeff = virt * (virt - 1) // 2
    shift =  occ_coeff * occ * (occ - 1) // 2
    # j > i and b > a highest index
    if j > i and b > a:
        return 3*shift + (j * (j - 1) // 2 + i) * occ_coeff + (b - occ)*(b - occ - 1) // 2 + a - occ
    # j > i and b < a second-highest index
    elif j > i:
        return 2*shift + (j * (j - 1) // 2 + i) * occ_coeff + (a - occ)*(a - occ - 1) // 2 + b - occ
    # j < i and b > a third-highest index
    elif b > a:
        return shift + (i * (i - 1) // 2 + j) * occ_coeff + (b - occ)*(b - occ - 1) // 2 + a - occ
    # Default case, but this shouldn't be used
    else:
        return (i * (i - 1) // 2 + j) * occ_coeff + (a - occ)*(a - occ - 1) // 2 + b - occ
    
def flatten_index(i, j, a, b, occ, virt):
    '''
    Returns the corresponding general index (j < i and b < a)
    Requires four indices and number of occupied and virtual orbitals looped over
    '''
    if j > i:
        i, j = j, i
    if b > a:
        a, b = b, a
    return (i * (i - 1) // 2 + j) * virt * (virt - 1) // 2 + (a - occ)*(a - occ - 1) // 2 + b - occ + 1

def calculate_1rdm(ci_vec, occ, virt):
    n_spin = occ + virt
    one_rdm = np.zeros((n_spin, n_spin), dtype=np.float64)
    np.fill_diagonal(one_rdm[:occ, :occ], 1)

    # Occ-occ off-diagonal
    # Not relevant for 2-electron
    # n_virt = virt * (virt - 1) // 2
    # for i in range(1, occ):
    #     for j in range(0, i):
    #         val = int(i == j)

    #         i_idx = flatten_index(i, 0, occ+1, occ, occ, virt)
    #         j_idx = flatten_index(j, 0, occ+1, occ, occ, virt)
    #         for k in range(0, j):
    #             for a in range(occ + 1, n_spin):
    #                 for b in range(occ, a):
    #                     val -= ci_vec[i_idx]*ci_vec[j_idx]
    #                     i_idx += 1
    #                     j_idx += 1
            
    #         i_idx = flatten_index(i, j+1, occ+1, occ, occ, virt)
    #         j_idx = flatten_index(j, j+1, occ+1, occ, occ, virt)
    #         for k in range(j+1, i):
    #             for a in range(occ + 1, n_spin):
    #                 for b in range(occ, a):
    #                     val += ci_vec[i_idx]*ci_vec[j_idx] # Negative due to antisymmetry
    #                     i_idx += 1
    #                     j_idx += 1
    #             j_idx += (k-1)*n_virt # Need to account for skipping indices

    #         i_idx = flatten_index(i, i+1, occ+1, occ, occ, virt)
    #         j_idx = flatten_index(j, i+1, occ+1, occ, occ, virt)
    #         for k in range(i+1, occ):
    #             for a in range(occ + 1, n_spin):
    #                 for b in range(occ, a):
    #                     val -= ci_vec[i_idx]*ci_vec[j_idx]
    #                     i_idx += 1
    #                     j_idx += 1
    #             i_idx += (k-1)*n_virt
    #             j_idx += (k-1)*n_virt
    #         one_rdm[i, j] = one_rdm[j, i] = val
    
    # Occ-occ diagonal
    v_idx = 1
    for i in range(1, occ):
        for k in range(0, i):
            val = 0
            for a in range(occ + 1, n_spin):
                for b in range(occ, a):
                    val += abs(ci_vec[v_idx]) ** 2
                    v_idx += 1
            one_rdm[i, i] -= val
            one_rdm[k, k] -= val

    # Virt-virt off-diagonal
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ + 1, n_spin):
                for b in range(occ, a):
                    val = 0

                    a_idx = flatten_index(i, j, a, occ, occ, virt)
                    b_idx = flatten_index(i, j, b, occ, occ, virt)
                    for c in range(occ, b):
                        val += ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += 1
                        b_idx += 1
                    
                    a_idx += 1
                    b_idx = flatten_index(i, j, b, b+1, occ, virt)
                    for c in range(b+1, a):
                        val -= ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += 1
                        b_idx += c - occ
                    
                    a_idx = flatten_index(i, j, a, a+1, occ, virt)
                    b_idx = flatten_index(i, j, b, a+1, occ, virt)
                    for c in range(a+1, virt):
                        val += ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += c - occ
                        b_idx += c - occ
                    one_rdm[a, b] += val
                    one_rdm[b, a] += val

    # Virt-virt diagonal
    v_idx = 1
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ + 1, n_spin):
                for c in range(occ, a):
                    val = abs(ci_vec[v_idx]) ** 2
                    one_rdm[a, a] += val
                    one_rdm[c, c] += val
                    v_idx += 1
    
    return one_rdm

def calculate_JK(one_rdm, g_spin, n_spin):
    J = np.zeros((n_spin, n_spin), dtype=np.float64)
    K = np.zeros((n_spin, n_spin), dtype=np.float64)
    
    for p in range(n_spin):
        for q in range(n_spin):
            J_sum = 0
            K_sum = 0
            for r in range(n_spin):
                for s in range(n_spin):
                    d = one_rdm[r, s]
                    J_sum += g_spin[p, r, q, s] * d
                    K_sum += g_spin[p, r, s, q] * d
            J[p, q] += J_sum
            K[p, q] += -K_sum
    
    return J, K

def build_EKT_Fock(h, one_rdm, g_spin, two_rdm):
    n_spin = h.shape[0]
    one_e = np.zeros((n_spin, n_spin), dtype=np.float64)
    two_e = np.zeros((n_spin, n_spin), dtype=np.float64)
    F = np.zeros((n_spin, n_spin), dtype=np.float64)
    for p in range(n_spin):
        for q in range(n_spin):
            one_sum = 0
            two_sum = 0
            for r in range(n_spin):
                one_sum += h[p, r] * one_rdm[r, q]
                for s in range(n_spin):
                    for t in range(n_spin):
                        two_sum += (g_spin[r,s,t,p] - g_spin[r,s,p,t]) * two_rdm[r,s,t,q]
            one_e[p,q] += one_sum
            two_e[p,q] += two_sum
    F = one_e + two_e
    return F, one_e, two_e
    