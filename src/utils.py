import numpy as np
from itertools import combinations

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
    return (i * (i - 1) // 2 + j) * virt * (virt - 1) // 2 + (a - occ)*(a - occ - 1) // 2 + b - occ

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

                    a_idx = flatten_index(i, j, a, occ, occ, virt) + 1
                    b_idx = flatten_index(i, j, b, occ, occ, virt) + 1
                    for c in range(occ, b):
                        val += ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += 1
                        b_idx += 1
                    
                    a_idx += 1
                    b_idx = flatten_index(i, j, b, b+1, occ, virt) + 1
                    for c in range(b+1, a):
                        val -= ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += 1
                        b_idx += c - occ
                    
                    a_idx = flatten_index(i, j, a, a+1, occ, virt) + 1
                    b_idx = flatten_index(i, j, b, a+1, occ, virt) + 1
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

def calculate_1rdm_from_full(ci_vec, occ, virt):
    n_spin = occ + virt
    n_singles = occ * virt
    one_rdm = np.zeros((n_spin, n_spin), dtype=np.float64)
    np.fill_diagonal(one_rdm[:occ, :occ], 1)

    # Singles Portion

    # Occ-occ off-diagonal
    for i in range(1, occ):
        for j in range(0, i):
            val = 0
            for a in range(occ, n_spin):
                a_idx = a - occ
                val += ci_vec[i * virt + a_idx + 1] * ci_vec[j * virt + a_idx + 1]
            one_rdm[i, j] -= val
            one_rdm[j, i] -= val

    # Occ-occ diagonal
    v_idx = 1
    for i in range(occ):
        for a in range(occ, n_spin):
            val += abs(ci_vec[v_idx]) ** 2
            v_idx += 1
        one_rdm[i, i] -= val
        val = 0

    # Virt-Virt off-diagonal
    for i in range(occ):
        i_idx = i * virt
        for a in range(occ + 1, n_spin):
            for b in range(occ, a):
                a_idx = a - occ
                b_idx = b - occ
                val = ci_vec[i_idx + a_idx + 1] * ci_vec[i_idx + b_idx + 1]
                one_rdm[a, b] += val
                one_rdm[b, a] += val

    # Virt-Virt diagonal
    v_idx = 1
    for i in range(occ):
        for a in range(occ, n_spin):
                one_rdm[a, a] += abs(ci_vec[v_idx]) ** 2
                v_idx += 1

    # Off-diagonal
    offdiag = -ci_vec[1:n_singles + 1] * ci_vec[0]
    for i in range(occ):
        for a in range(occ, n_spin):
            for j in range(0, i):
                for b in range(occ, a):
                    sub_idx_1 = i * virt + (a - occ)
                    sub_idx_2 = j * virt + (b - occ)
                    v_idx = flatten_index(i, j, a, b, occ, virt) + 1 + n_singles
                    offdiag[sub_idx_1] +=  ci_vec[v_idx] * ci_vec[sub_idx_2 + 1]
                for b in range(a + 1, n_spin):
                    sub_idx_1 = i * virt + (a - occ)
                    sub_idx_2 = j * virt + (b - occ)
                    v_idx = flatten_index(i, j, a, b, occ, virt) + 1 + n_singles
                    offdiag[sub_idx_1] -=  ci_vec[v_idx] * ci_vec[sub_idx_2 + 1]
            for j in range(i + 1, occ):
                for b in range(occ, a):
                    sub_idx_1 = i * virt + (a - occ)
                    sub_idx_2 = j * virt + (b - occ)
                    v_idx = flatten_index(i, j, a, b, occ, virt) + 1 + n_singles
                    offdiag[sub_idx_1] -=  ci_vec[v_idx] * ci_vec[sub_idx_2 + 1]
                for b in range(a + 1, n_spin):
                    sub_idx_1 = i * virt + (a - occ)
                    sub_idx_2 = j * virt + (b - occ)
                    v_idx = flatten_index(i, j, a, b, occ, virt) + 1 + n_singles
                    offdiag[sub_idx_1] +=  ci_vec[v_idx] * ci_vec[sub_idx_2 + 1]
    # Not sure how to account for phase
    offdiag[n_singles // 2:n_singles] = -offdiag[n_singles // 2:n_singles]
    offdiag = offdiag.reshape((2, -1))
    one_rdm[0:occ, occ:n_spin] = offdiag
    one_rdm[occ:n_spin, 0:occ] = offdiag.T

    # Doubles Portion

    # Occ-occ diagonal
    v_idx = 1 + n_singles
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

                    a_idx = flatten_index(i, j, a, occ, occ, virt) + 1 + n_singles
                    b_idx = flatten_index(i, j, b, occ, occ, virt) + 1 + n_singles
                    for c in range(occ, b):
                        val += ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += 1
                        b_idx += 1
                    
                    a_idx += 1
                    b_idx = flatten_index(i, j, b, b+1, occ, virt) + 1 + n_singles
                    for c in range(b+1, a):
                        val -= ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += 1
                        b_idx += c - occ
                    
                    a_idx = flatten_index(i, j, a, a+1, occ, virt) + 1 + n_singles
                    b_idx = flatten_index(i, j, b, a+1, occ, virt) + 1 + n_singles
                    for c in range(a+1, n_spin):
                        val += ci_vec[a_idx] * ci_vec[b_idx]
                        a_idx += c - occ
                        b_idx += c - occ
                    one_rdm[a, b] += val
                    one_rdm[b, a] += val

    # Virt-virt diagonal
    v_idx = 1 + n_singles
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ, n_spin):
                v_idx = flatten_index(i, j, a, occ, occ, virt) + 1 + n_singles
                for c in range(occ, a):
                    val = abs(ci_vec[v_idx]) ** 2
                    one_rdm[a, a] += val
                    v_idx += 1
                v_idx = flatten_index(i, j, a, a+1, occ, virt) + 1 + n_singles
                for c in range(a+1, n_spin):
                    val = abs(ci_vec[v_idx]) ** 2
                    one_rdm[a, a] += val
                    v_idx += c - occ
    
    return one_rdm

def calculate_2rdm_from_full(two_rdm, ci_vec, occ, virt, trdm=0):
    '''
    Calculates the irreducible part
    '''
    n_spin = occ + virt
    n_singles = occ * virt
    d_0 = 1 + n_singles
    # two_rdm = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)

    # ijkl
    v_idx_1 = 0
    v_idx_2 = 0
    for i in range(1, occ):
        for j in range(0, i):
            cur_idx = v_idx_1
            for l in range(1, occ):
                for k in range(0, l):
                    v_idx_1 = cur_idx
                    val = 0
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            # Might be a factor of 0.5 or 2 here?
                            val += 0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[d_0 + v_idx_2]
                            v_idx_1 += 1
                            v_idx_2 += 1
                    two_rdm[i, j, k, l] -= val
                    two_rdm[j, i, l, k] -= val
                    two_rdm[j, i, k, l] += val
                    two_rdm[i, j, l, k] += val

    # abcd
    v_idx_1 = 0
    v_idx_2 = 0
    for i in range(1, occ):
        for j in range(0, i):
            cur_idx = v_idx_2
            for a in range(occ+1, n_spin):
                for b in range(occ, a):
                    v_idx_2 = cur_idx
                    for d in range(occ+1, n_spin):   
                        for c in range(occ, d):
                            # Might be a factor of 0.5 or 2 here?
                            # For some reason this is negative; I'll have to double check
                            val = -0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[d_0 + v_idx_2]
                            two_rdm[a, b, c, d] += val
                            two_rdm[b, a, c, d] -= val
                            two_rdm[a, b, d, c] -= val
                            two_rdm[b, a, d, c] += val
                            v_idx_2 += 1
                    v_idx_1 += 1

    # ijab (and abij)
    v_idx_1 = 0
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ+1, n_spin):
                for b in range(occ, a):
                    # Check if negative
                    val = -0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[0]
                    v_idx_1 += 1
                    two_rdm[i, j, a, b] -= val
                    two_rdm[j, i, b, a] -= val
                    two_rdm[a, b, i, j] -= val
                    two_rdm[b, a, j, i] -= val
                    two_rdm[j, i, a, b] += val
                    two_rdm[i, j, b, a] += val
                    two_rdm[a, b, j, i] += val
                    two_rdm[b, a, i, j] += val


    # ijka (and every order of three occ one virt)
    v_idx_1 = 0 # Double
    v_idx_2 = 0 # Single
    for i in range(1, occ):
        for j in range(0, i):
            cur_idx_1 = v_idx_1 # Set start for doubles for given (i, j)
            cur_idx_1_temp = v_idx_1
            for k in range(occ):
                cur_idx_2 = v_idx_2 # Set start for singles for given k
                cur_idx_1 = cur_idx_1_temp # Reset start for doubles
                for a in range(occ, n_spin):
                    v_idx_1 = cur_idx_1 # Reset doubles idx
                    v_idx_2 = cur_idx_2 # Reset singles for given k
                    val = 0
                    # Might need a factor of 0.5
                    for b in range(occ, a):
                        val += 0.5 * ci_vec[1 + v_idx_2] * ci_vec[d_0 + v_idx_1]
                        v_idx_1 += 1
                        v_idx_2 += 1
                    cur_idx_1 = v_idx_1 # Set new start for doubles
                    v_idx_1 += a - occ # Switch from (a, a-1) to (a+1, a)
                    v_idx_2 += 1 # Account for the jump where b != a
                    for b in range(a+1, n_spin):
                        val += 0.5 * ci_vec[1 + v_idx_2] * -ci_vec[d_0 + v_idx_1]
                        v_idx_1 += b - occ # Jump from (a+1, a) to (a+2, a) etc.
                        v_idx_2 += 1
                    two_rdm[i, j, k, a] -= val
                    two_rdm[j, i, a, k] -= val
                    two_rdm[k, a, i, j] -= val
                    two_rdm[a, k, j, i] -= val
                    two_rdm[j, i, k, a] += val
                    two_rdm[i, j, a, k] += val
                    two_rdm[k, a, j, i] += val
                    two_rdm[a, k, i, j] += val
            v_idx_2 = 0 # Reset singles entirely

    # abci
    v_idx_1 = 0 # Double
    v_idx_2 = 0 # Single
    cur_idx_1 = v_idx_1
    for i in range(0, occ):
        for j in range(0, i):
            v_idx_2 = j * virt
            for c in range(occ, n_spin):
                v_idx_1 = cur_idx_1
                for a in range(occ+1, n_spin):
                    for b in range(occ, a):
                        val = -0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[1 + v_idx_2]
                        two_rdm[a, b, c, i] -= val
                        two_rdm[b, a, i, c] -= val
                        two_rdm[c, i, a, b] -= val
                        two_rdm[i, c, b, a] -= val
                        two_rdm[b, a, c, i] += val
                        two_rdm[a, b, i, c] += val
                        two_rdm[c, i, b, a] += val
                        two_rdm[i, c, a, b] += val
                        v_idx_1 += 1
                v_idx_2 += 1
            cur_idx_1 = v_idx_1
        for j in range(i+1, occ):
            v_idx_2 = j * virt
            cur_idx_1_temp = flatten_index(j, i, occ+1, occ, occ, virt)
            for c in range(occ, n_spin):
                v_idx_1 = cur_idx_1_temp
                for a in range(occ+1, n_spin):
                    for b in range(occ, a):
                        val = 0.5 * -ci_vec[d_0 + v_idx_1] * ci_vec[1 + v_idx_2]
                        two_rdm[a, b, c, i] -= val
                        two_rdm[b, a, i, c] -= val
                        two_rdm[c, i, a, b] -= val
                        two_rdm[i, c, b, a] -= val
                        two_rdm[b, a, c, i] += val
                        two_rdm[a, b, i, c] += val
                        two_rdm[c, i, b, a] += val
                        two_rdm[i, c, a, b] += val
                        v_idx_1 += 1
                v_idx_2 += 1

    # iajb
    v_idx_1 = 0 # ja
    v_idx_2 = 0 # ib
    # Since this is 2-electron, the only cases we have
    # for (j,k) and (i,k) are (1,0) (1,0) and (0,1) (0,1)
    # which are the same by (anti)symmetry, so it's
    # really not worth it to code the occ loops right now
    for a in range(occ, n_spin):
        for b in range(occ, a+1):
            v_idx_1 = flatten_index(1,0,a,occ,occ,virt)
            v_idx_2 = flatten_index(1,0,b,occ,occ,virt)
            val = 0
            for c in range(occ, b):
                val += -0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[d_0 + v_idx_2]
                v_idx_1 += 1
                v_idx_2 += 1
            v_idx_2 = flatten_index(1,0,b,b+1,occ,virt)
            v_idx_1 = flatten_index(1,0,a,b+1,occ,virt)
            for c in range(b+1, a):
                val += 0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[d_0 + v_idx_2]
                v_idx_1 += 1
                v_idx_2 += c - occ # Jump from (b+1, b) to (b+2, b) etc.
            v_idx_2 = flatten_index(1,0,b,a+1,occ,virt)
            v_idx_1 = flatten_index(1,0,a,a+1,occ,virt)
            for c in range(a+1, n_spin):
                val += -0.5 * ci_vec[d_0 + v_idx_1] * ci_vec[d_0 + v_idx_2]
                v_idx_1 += c - occ
                v_idx_2 += c - occ
            # i = j = 0
            a_idx = (a - occ) # + 0 * virt
            b_idx = (b - occ)
            new_val = val # + 0.5 * ci_vec[a_idx + 1] * ci_vec[b_idx + 1]
            two_rdm[0,a,0,b] += new_val
            two_rdm[a,0,b,0] += new_val
            two_rdm[0,a,b,0] -= new_val  
            two_rdm[a,0,0,b] -= new_val

            if(a != b):
                two_rdm[0,b,0,a] += new_val
                two_rdm[b,0,a,0] += new_val
                two_rdm[0,b,a,0] -= new_val
                two_rdm[b,0,0,a] -= new_val

            # i = j = 1
            a_idx += virt
            b_idx += virt
            new_val = val # + 0.5 * ci_vec[a_idx + 1] * ci_vec[b_idx + 1]
            two_rdm[1,a,1,b] += new_val
            two_rdm[a,1,b,1] += new_val
            two_rdm[1,a,b,1] -= new_val
            two_rdm[a,1,1,b] -= new_val
            
            if(a != b):
                two_rdm[1,b,1,a] += new_val
                two_rdm[b,1,a,1] += new_val
                two_rdm[1,b,a,1] -= new_val
                two_rdm[b,1,1,a] -= new_val

    return two_rdm

                            

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

def _apply_one_body(det, p, q):
    if not (det & (1 << q)): return None, 0
    phase1 = (-1) ** (det & ((1 << q) - 1)).bit_count()
    det_tmp = det ^ (1 << q)
    
    if det_tmp & (1 << p): return None, 0
    phase2 = (-1) ** (det_tmp & ((1 << p) - 1)).bit_count()
    return det_tmp | (1 << p), phase1 * phase2

def calculate_1rdm_full_from_external(ci_vector, n_spin, n_elec):

    dets = []
    for occ in combinations(range(n_spin), n_elec):
        det = 0
        for p in occ:
            det |= (1 << p)
        dets.append(det)
    det_map = {det: i for i, det in enumerate(dets)}

    rdm = np.zeros((n_spin, n_spin))

    # Loop over determinants in the basis
    for j, det_j in enumerate(dets):
        c_j = ci_vector[j]
        if abs(c_j) < 1e-12: 
            continue

        # for each determinant, apply a_p^dagger a_q for all spin-orbitals pairs p, q
        for q in range(n_spin):
            if not (det_j & (1 << q)):
                continue

            for p in range(n_spin):

                det_i, phase = _apply_one_body(det_j, p, q)
                if det_i is not None and det_i in det_map:
                    i = det_map[det_i]
                    c_i = ci_vector[i]
                    rdm[p, q] += c_i * c_j * phase

    return rdm