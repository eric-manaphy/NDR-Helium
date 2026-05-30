    # if mode in ["hf", "opt"]:
    #     C = scf_res["coefficients"]
    #     f_mo = C.T @ scf_res["Fock"] @ C
    #     eri_mo = ao_to_mo_transform(scf_res["eri"], scf_res["coefficients"])
    #     f_spin, g_spin = build_spin_orbital_integrals(f_mo, eri_mo)
    #     e = np.diag(f_spin)
    #     n_spin = f_spin.shape[0]
    #     occ = scf_res["n_elec"]
    #     virt = n_spin - occ
    #     n_occ = occ*(occ-1) // 2
    #     n_virt = virt*(virt-1) // 2
    #     full_dim = n_occ * n_virt + 1
    #     solution_dim = 1
    #     basis_dim = 1
    #     e_diff = np.empty(full_dim, dtype=np.float64, order='F')
    #     mp2 = np.empty(full_dim, dtype=np.float64, order='F')

    #     print(e)

    #     v_idx = 1
    #     e_mp2 = 0
    #     for i in range(1, occ):
    #         for j in range(0, i):
    #             for a in range(occ+1, n_spin):
    #                 for b in range(occ, a):
    #                     sum = e[a]+e[b]-e[i]-e[j]
    #                     mp2[v_idx] = -(g_spin[a,b,i,j] - g_spin[a,b,j,i]) / sum
    #                     e_mp2 += (g_spin[a,b,i,j] - g_spin[a,b,j,i]) ** 2 / sum
    #                     e_diff[v_idx] = sum
    #                     v_idx += 1
    #     e_diff[0] = 0.000001
    #     mp2[0] = 0.000001

    #     # mat_prod = np.array([[0, -(g_spin[0,1,2,3]-g_spin[0,1,3,2])], [-(g_spin[2,3,0,1]-g_spin[2,3,1,0]), e[2]+e[3]-e[0]-e[1]-(g_spin[0,1,1,0]-g_spin[0,1,0,1])+(g_spin[2,3,3,2]-g_spin[2,3,2,3])]])
    #     # print(mat_prod)
    #     # print(np.linalg.eigh(mat_prod))
    #     lk.initialize()
    #     index = lk.add_space(lk.Kind.REAL, lk.Structure.SYMMETRIC, lk.Equation.EIGENVALUE,
    #                          full_dim, solution_dim, basis_dim)
    #     # vectors = np.zeros(full_dim, dtype=np.float64, order='F')
    #     # vectors[5] = 1.0
    #     # vectors = np.random.rand(full_dim)
    #     # lk.set_real_space_vectors(index, vectors)
    #     lk.set_space_preconditioner(index, 'c')
    #     lk.set_space_diagonal(index, e_diff)
    #     lk.set_real_space_vectors_from_diagonal(index, basis_dim, e_diff)
    #     # lk.set_space_diagonal(index, mp2)
    #     # lk.set_real_space_vectors(index, mp2)
    #     # lk.set_real_space_vectors(index, np.asarray([-0.9993283 ,  0.03664628]))
    #     def multiply2(vectors, products):
    #         # Calculate c_0, first term, and last term
    #         c_0 = vectors[0]
    #         d_0 = 0
    #         # foe = 0 # First-order energy
    #         v_idx = 1
    #         for i in range(1, occ):
    #             for j in range(0, i):
    #                 # foe -= (g_spin[i,j,i,j] - g_spin[i,j,j,i])
    #                 for a in range(occ+1, n_spin):
    #                     for b in range(occ, a):
    #                         ijab = vectors[v_idx]
    #                         d_0 += -(g_spin[i,j,a,b]-g_spin[i,j,b,a])*ijab
    #                         # Not sure if I can guarantee products is already initalized
    #                         # So I'll just initialize (the relevant indices) here
    #                         products[v_idx] = (e_diff[v_idx])*ijab - (g_spin[a,b,i,j]-g_spin[a,b,j,i])*c_0
    #                         v_idx += 1
    #         # print(f"1st order energy: {foe}")
    #         products[0] = d_0
    #         # Calculate occ subloop (second term)
    #         curr_idx = 1
    #         for i in range(1, occ):
    #             for j in range(0, i):
    #                 v_sub_idx = 1
    #                 for l in range(1, occ):
    #                     for k in range(0, l):
    #                         v_idx = curr_idx
    #                         for a in range(occ+1, n_spin):
    #                             for b in range(occ, a):
    #                                 products[v_idx] -= vectors[v_sub_idx]*(g_spin[l,k,j,i]-g_spin[k,l,j,i])
    #                                 v_idx += 1
    #                                 v_sub_idx += 1
    #                 curr_idx = v_idx

    #         # Calculate virt subloop (second term)
    #         v_idx = 1
    #         curr_idx = 1
    #         for i in range(1, occ):
    #             for j in range(0, i):
    #                 for a in range(occ+1, n_spin):
    #                     for b in range(occ, a):
    #                         v_sub_idx = curr_idx
    #                         ijab = 0
    #                         for d in range(occ+1, n_spin):
    #                             for c in range(occ, d):
    #                                 ijab += vectors[v_sub_idx]*(g_spin[a,b,c,d]-g_spin[a,b,d,c])
    #                                 v_sub_idx += 1
    #                         products[v_idx] += ijab
    #                         v_idx += 1
    #                 curr_idx = v_sub_idx

    #         return 0
    #     # def multiply2(vectors, products):
    #     #     np.copyto(products, mat_prod @ vectors)
    #     #     return 0
    #     lk.solve_real_equation(index, multiply2)
    #     print(lk.get_space_eigenvalues(index, solution_dim))
    #     ci_vec = lk.get_real_space_solutions(index, full_dim, solution_dim).flatten()
    #     print(ci_vec)
    #     lk.finalize()

    #     one_rdm = np.zeros((n_spin, n_spin), dtype=np.float64)

    #     # Occ-occ off-diagonal
    #     # Not relevant for 2-electron
    #     # for i in range(1, occ):
    #     #     for j in range(0, i):
    #     #         val = int(i == j)

    #     #         i_idx = flatten_index(i, 0, occ+1, occ, occ, virt)
    #     #         j_idx = flatten_index(j, 0, occ+1, occ, occ, virt)
    #     #         for k in range(0, j):
    #     #             for a in range(occ + 1, n_spin):
    #     #                 for b in range(occ, a):
    #     #                     val -= ci_vec[i_idx]*ci_vec[j_idx]
    #     #                     i_idx += 1
    #     #                     j_idx += 1
                
    #     #         i_idx = flatten_index(i, j+1, occ+1, occ, occ, virt)
    #     #         j_idx = flatten_index(j, j+1, occ+1, occ, occ, virt)
    #     #         for k in range(j+1, i):
    #     #             for a in range(occ + 1, n_spin):
    #     #                 for b in range(occ, a):
    #     #                     val += ci_vec[i_idx]*ci_vec[j_idx] # Negative due to antisymmetry
    #     #                     i_idx += 1
    #     #                     j_idx += 1
    #     #             j_idx += (k-1)*n_virt # Need to account for skipping indices

    #     #         i_idx = flatten_index(i, j+1, occ+1, occ, occ, virt)
    #     #         j_idx = flatten_index(j, j+1, occ+1, occ, occ, virt)
    #     #         for k in range(i+1, occ):
    #     #             for a in range(occ + 1, n_spin):
    #     #                 for b in range(occ, a):
    #     #                     val -= ci_vec[i_idx]*ci_vec[j_idx]
    #     #                     i_idx += 1
    #     #                     j_idx += 1
    #     #             i_idx += (k-1)*n_virt
    #     #             j_idx += (k-1)*n_virt
    #     #         one_rdm[i, j] = one_rdm[j, i] = val
        
    #     # Occ-occ diagonal
    #     np.fill_diagonal(one_rdm[:occ, :occ], 1)
    #     v_idx = 1
    #     for i in range(1, occ):
    #         for k in range(0, i):
    #             val = 0
    #             for a in range(occ + 1, n_spin):
    #                 for b in range(occ, a):
    #                     val += abs(ci_vec[v_idx]) ** 2
    #                     v_idx += 1
    #             one_rdm[i, i] -= val
    #             one_rdm[k, k] -= val

    #     # Virt-virt off-diagonal
    #     for i in range(1, occ):
    #         for j in range(0, i):
    #             for a in range(occ + 1, n_spin):
    #                 for b in range(occ, a):
    #                     val = 0

    #                     a_idx = flatten_index(i, j, a, occ, occ, virt)
    #                     b_idx = flatten_index(i, j, b, occ, occ, virt)
    #                     for c in range(occ, b):
    #                         val += ci_vec[a_idx] * ci_vec[b_idx]
    #                         a_idx += 1
    #                         b_idx += 1
                        
    #                     a_idx += 1
    #                     b_idx = flatten_index(i, j, b, b+1, occ, virt)
    #                     for c in range(b+1, a):
    #                         val -= ci_vec[a_idx] * ci_vec[b_idx]
    #                         a_idx += 1
    #                         b_idx += c - occ
                        
    #                     a_idx = flatten_index(i, j, a, a+1, occ, virt)
    #                     b_idx = flatten_index(i, j, b, a+1, occ, virt)
    #                     for c in range(a+1, virt):
    #                         val += ci_vec[a_idx] * ci_vec[b_idx]
    #                         a_idx += c - occ
    #                         b_idx += c - occ
    #                     one_rdm[a, b] += val
    #                     one_rdm[b, a] += val

    #     # Virt-virt diagonal
    #     v_idx = 1
    #     for i in range(1, occ):
    #         for j in range(0, i):
    #             for a in range(occ + 1, n_spin):
    #                 for c in range(occ, a):
    #                     val = abs(ci_vec[v_idx]) ** 2
    #                     one_rdm[a, a] += val
    #                     one_rdm[c, c] += val
    #                     v_idx += 1

    #     print(one_rdm)
    #     occs, natural_orbitals, ndr_coeffs = get_natural_orbitals(one_rdm, N_elec)
    #     print(np.trace(one_rdm))
    #     print("\nNatural Orbital Occupation Numbers:")
    #     for i, occ in enumerate(occs):
    #         print(f"  NO {i+1}: {occ:.8f}")
    #     # print(natural_orbitals)
    #     # print(ndr_coeffs)

    #     J = np.zeros((n_spin, n_spin))
    #     K = np.zeros((n_spin, n_spin))

    #     for p in range(n_spin):
    #         for q in range(n_spin):
    #             J_sum = 0
    #             K_sum = 0
    #             for r in range(n_spin):
    #                 for s in range(n_spin):
    #                     d = one_rdm[r, s]
    #                     J_sum += g_spin[p, r, q, s] * d
    #                     K_sum += g_spin[p, r, s, q] * d
    #             J[p, q] += J_sum
    #             K[p, q] += -K_sum

    #     # J, K = build_JK(g_spin, one_rdm)
    #     h_ao = scf_res["H_core"]
    #     h_mo = C.T @ h_ao @ C
    #     h_spin, _ = build_spin_orbital_integrals(h_mo, eri_mo)
    #     F = h_spin + J + K
    #     eigenvalues, eigenvectors = eigh(F)
    #     print(eigenvalues)

    #     return 0
