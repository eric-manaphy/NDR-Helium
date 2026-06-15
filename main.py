import numpy as np
import json
from scipy.linalg import eigh, block_diag
from src.zetas import Zetas
from src import make_basis
from src.solver.hf_scf import scf, group_basis_by_lm
# from src.integrals.two_electron import build_JK
from src.solver.ci_full import FullCISolver
from src.solver.orbital_opt import optimize_hf_orbitals
from src.solver.NDR import calculate_1rdm_full, get_natural_orbitals, calculate_2rdm
from src.utils import ao_to_mo_transform, build_spin_orbital_integrals, flatten_index, calculate_1rdm, calculate_JK, build_EKT_Fock
from src.ndr import libkrylov as lk

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def list_to_np(obj):
    for key, value in obj.items():
        if isinstance(value, list):
            obj[key] = np.asarray(value, dtype=np.float64)
        elif isinstance(value, dict):
            obj[key] = list_to_np(value)
    return obj

def multiply(vectors, products, occ, virt, g_spin, e_diff, f_c):
    # Calculate c_0, first term, and last term
    c_0 = vectors[0]
    d_0 = 0

    n_spin = occ+virt
    n_virt = virt * (virt - 1) // 2

    v_idx = 1
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ+1, n_spin):
                for b in range(occ, a):
                    ijab = vectors[v_idx]
                    d_0 += -(g_spin[i,j,a,b]-g_spin[i,j,b,a])*ijab
                    # Not sure if I can guarantee products is already initalized
                    # So I'll just initialize (the relevant indices) here
                    products[v_idx] = (e_diff[v_idx])*ijab - (g_spin[a,b,i,j]-g_spin[a,b,j,i])*c_0
                    v_idx += 1
    products[0] = d_0
    # Calculate occ subloop (second term)
    curr_idx = 1
    for i in range(1, occ):
        for j in range(0, i):
            v_sub_idx = 1
            for l in range(1, occ):
                for k in range(0, l):
                    v_idx = curr_idx
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            products[v_idx] -= vectors[v_sub_idx]*(g_spin[l,k,j,i]-g_spin[k,l,j,i])
                            v_idx += 1
                            v_sub_idx += 1
            curr_idx = v_idx

    # Calculate virt subloop (second term)
    v_idx = 1
    curr_idx = 1
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ+1, n_spin):
                for b in range(occ, a):
                    v_sub_idx = curr_idx
                    ijab = 0
                    for d in range(occ+1, n_spin):
                        for c in range(occ, d):
                            ijab += vectors[v_sub_idx]*(g_spin[a,b,c,d]-g_spin[a,b,d,c])
                            v_sub_idx += 1
                    products[v_idx] += ijab
                    v_idx += 1
            curr_idx = v_sub_idx

    if f_c is not None:
        v_idx = 1
        curr_idx = 1
        for i in range(1, occ):
            for j in range(0, i):
                for k in range(j+1, occ):
                    v_idx = curr_idx
                    j_idx = flatten_index(k, j, occ+1, occ, occ, virt)
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            products[v_idx] += f_c[k, i] * vectors[j_idx]
                            v_idx += 1
                            j_idx += 1
                for k in range(i+1, occ):
                    v_idx = curr_idx
                    i_idx = flatten_index(k, i, occ+1, occ, occ, virt)
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            products[v_idx] -= f_c[k, j] * vectors[i_idx]
                            v_idx += 1
                            i_idx += 1
                for k in range(0, j):
                    v_idx = curr_idx
                    j_idx = flatten_index(j, k, occ+1, occ, occ, virt)
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            products[v_idx] -= f_c[k, i] * vectors[j_idx]
                            v_idx += 1
                            j_idx += 1
                for k in range(0, i):
                    v_idx = curr_idx
                    i_idx = flatten_index(i, k, occ+1, occ, occ, virt)
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            products[v_idx] += f_c[k, j] * vectors[i_idx]
                            v_idx += 1
                            i_idx += 1
                curr_idx = v_idx

        v_idx = 1
        for i in range(1, occ):
            for j in range(0, i):
                for a in range(occ+1, n_spin):
                    for b in range(occ, a):
                        val = 0
                        for c in range(b+1, n_spin):
                            val -= f_c[c, a] * vectors[flatten_index(i, j, c, b, occ, virt)]
                        for c in range(a+1, n_spin):
                            val += f_c[c, b] * vectors[flatten_index(i, j, c, a, occ, virt)]
                        v_sub_idx = flatten_index(i, j, b, occ, occ, virt)
                        for c in range(occ, b):
                            val += f_c[c, a] * vectors[v_sub_idx]
                            v_sub_idx += 1
                        v_sub_idx = flatten_index(i, j, a, occ, occ, virt)
                        for c in range(occ, a):
                            val -= f_c[c, b] * vectors[v_sub_idx]
                            v_sub_idx += 1
                        products[v_idx] += val
                        v_idx += 1

        # v_idx = 1
        # curr_idx = 1
        # for i in range(1, occ):
        #     for j in range(0, i):
        #         j_idx = flatten_index(j, 0, occ+1, occ, occ, virt)
        #         for k in range(0, j):
        #             v_idx = curr_idx
        #             for a in range(occ+1, n_spin):
        #                 for b in range(occ, a):
        #                     products[v_idx] -= f_c[k, j] * vectors[j_idx]
        #                     v_idx += 1
        #                     j_idx += 1

        #         i_idx = curr_idx
        #         for k in range(j, i):
        #             v_idx = curr_idx
        #             for a in range(occ+1, n_spin):
        #                 for b in range(occ, a):
        #                     products[v_idx] += f_c[k, j] * vectors[i_idx]
        #                     v_idx += 1
        #                     i_idx += 1
                
        #         # k = i
        #         v_idx = curr_idx
        #         for a in range(occ+1, n_spin):
        #             for b in range(occ, a):
        #                 products[v_idx] += f_c[i, i] * vectors[v_idx]
        #                 v_idx += 1

        #         for k in range(i+1, occ):
        #             v_idx = curr_idx
        #             i_idx = flatten_index(k, i, occ+1, occ, occ, virt)
        #             j_idx = flatten_index(k, j, occ+1, occ, occ, virt)
        #             for a in range(occ+1, n_spin):
        #                 for b in range(occ, a):
        #                     products[v_idx] += -f_c[k, j] * vectors[i_idx] + f_c[k, i] * vectors[j_idx]
        #                     v_idx += 1
                
        #         curr_idx = v_idx

        # v_idx = 1
        # for i in range(1, occ):
        #     for j in range(0, i):
        #         for a in range(occ+1, n_spin):
        #             for b in range(occ, a):
        #                 val = 0

        #                 v_sub_idx = flatten_index(i, j, b, occ, occ, virt)
        #                 for c in range(occ, b):
        #                     val -= f_c[c, a] * vectors[v_sub_idx]
        #                     v_sub_idx += 1

        #                 v_sub_idx = v_idx
        #                 for c in range(b, a):
        #                     val += f_c[c, b] * vectors[v_sub_idx]
        #                     v_sub_idx += 1
                        
        #                 v_sub_idx = v_idx

        #                 # c = a
        #                 val -= f_c[a, a] * vectors[v_sub_idx]
        #                 v_sub_idx -= a - occ

        #                 for c in range(a+1, n_spin):
        #                     val += f_c[c, a] * vectors[v_sub_idx] - f_c[c, b] * vectors[v_sub_idx + (a - b)]
        #                     v_sub_idx += c - occ

        #                 products[v_idx] += val
        #                 v_idx += 1

    return 0

def run_calculation(Z, N_elec, zetas, mode="hf"):
    """
    mode: "hf"          -> Single Hartree-Fock run
          "opt"         -> Optimize zetas for HF energy
          "fci"         -> Run HF then Full CI
    """
    print(f"\n{'='*60}")
    print(f" SYSTEM: Z={Z}, N={N_elec} | MODE: {mode.upper()}")
    print(f"{'='*60}")

    if mode == "opt":
        print("--> Starting Orbital Optimization...")
        res = optimize_hf_orbitals(Z, N_elec, zetas)
        working_zetas = res['zetas']
        print(f"--> Optimization Converged. Final Energy: {res['energy']:.10f}")

        print("Optimized Zetas:")
        for i, zeta in enumerate(working_zetas[0]):
            print(f"  Zeta {i+1}: {zeta:.6f}")
    else:
        working_zetas = zetas

    # 2. RUN BASE HF 
    print("--> Running SCF...")
    basis = make_basis(*working_zetas)
    scf_res = scf(basis, Z, N_elec, conv=1e-8)
    
    print(f"HF Total Energy: {scf_res['E_total']:.10f} Ha")

    if mode in ["hf", "opt"]:
        return scf_res

    # 3. FCI & NDR MODE
    if mode in ["fci", "ndr"]:
        print("--> Running Full Configuration Interaction...")
        ci_engine = FullCISolver(scf_res)
        ci_res = ci_engine.solve()
        
        print(f"FCI Total Energy: {ci_res['E_ci']:.10f} Ha")
        print(f"Correlation Energy: {ci_res['E_correlation']:.10f} Ha")

        ground_state_vector = ci_res['vectors'][:, 0]
        print(f"Leading CI Coefficients: {ground_state_vector[0]:.4f}, {ground_state_vector[1]:.4f}, ...")

        print("\n--> Verifying Eigenvector")
        H_ci = ci_res['H_ci']
        E_ci = ci_res['E_ci']

        residual = np.dot(H_ci, ground_state_vector) - E_ci * ground_state_vector
        residual_norm = np.linalg.norm(residual)
        print(f"  FCI Hamiltonian Residual Norm (||HC - EC||): {residual_norm:.12e}")

        if residual_norm < 1e-10:
            print("  [SUCCESS] Eigenvector is mathematically exact for this configuration space.")
        else:
            print("  [WARNING] High residual detected. Check alignment of your operators.")

        print(f"\nCISD Components extracted from FCI:")
        
        coeffs = ci_res['cisd_coeffs']
        
        print(f"  REFERENCE (C0): {coeffs['C_0']:.6f}")
        
        for level in ['singles', 'doubles']:
            data = coeffs.get(level, [])
            if data:
                print(f"  {level.upper()}:")
                
                sorted_excitations = sorted(data, key=lambda x: abs(x[1]), reverse=True)
                for det, coeff in sorted_excitations: 
                    print(f"    Det {bin(det)}: {coeff:.6f}")

        if mode == "ndr":
            print("\n--> Analyzing Natural Determinant Reference (NDR)...")
            ground_state_vec = ci_res['vectors'][:, 0]
            
            # Compute the exact 1RDM from CI coefficients
            rdm = calculate_1rdm_full(ci_engine, ground_state_vec)
            print(f"1RDM Computed. Trace (electrons): {np.trace(rdm):.4f}")

            trdm = calculate_2rdm(ci_engine, ground_state_vec)

            # Diagonalize 1RDM to find Natural Orbitals and Occupations
            occs, natural_orbitals, ndr_coeffs = get_natural_orbitals(rdm, N_elec)
            
            print("\nNatural Orbital Occupation Numbers:")
            for i, occ in enumerate(occs):
                print(f"  NO {i+1}: {occ:.8f}")

            # The NDR consists of the most occupied natural orbitals
            print(f"\nNDR constructed from {N_elec} most occupied natural orbitals.")
            
            # Add NDR data to results dictionary
            # ci_res['1rdm'] = rdm
            # ci_res['no_occupations'] = occs
            # ci_res['natural_orbitals'] = natural_orbitals

            C = scf_res["coefficients"]
            h_ao = scf_res["H_core"]
            h_mo = C.T @ h_ao @ C
            eri_mo = ao_to_mo_transform(scf_res["eri"], C)
            h_spin, g_spin = build_spin_orbital_integrals(h_mo, eri_mo)
            e = [e_spin for e_orb in scf_res["orb_energies"] for e_spin in (e_orb, e_orb)]
            n_spin = h_spin.shape[0]
            occ = scf_res["n_elec"]
            virt = n_spin - occ
            n_occ = occ*(occ-1) // 2
            n_virt = virt*(virt-1) // 2
            full_dim = n_occ * n_virt + 1
            solution_dim = 1
            basis_dim = 1
            e_diff = np.empty(full_dim, dtype=np.float64, order='F')

            rdm_0 = np.zeros((n_spin, n_spin), dtype=np.float64)
            np.fill_diagonal(rdm_0[:occ, :occ], 1)

            J, K = calculate_JK(rdm, g_spin, n_spin)
            f_spin = h_spin + J + K

            print(f'FCI F: {f_spin}')

            J, K = calculate_JK(rdm_0, g_spin, n_spin)
            f_0 = h_spin + J + K
            print(f'FCI F0: {f_0}')

            eigenvalues, eigenvectors = eigh(f_spin)

            print(f"NDR orbital energies: {eigenvalues}")

            scf_res["fci_coefficients"] = eigenvectors
            scf_res["fci_orb_energies"] = eigenvalues
            scf_res["fci_Fock"] = f_spin
            scf_res["fci_density"] = rdm
            scf_res["fci_F_0"] = f_0
            scf_res["fci_two_rdm"] = trdm
            # scf_res["F_c"] = (f_spin - f_0)
            # scf_res["H_core"] = h_spin
            # scf_res["g_spin"] = g_spin

            scf_res.update(ci_res)

        return scf_res

    return scf_res

def run_ci(scf_res, iteration = 1):
    print(f"Current CI Iteration: {iteration}")
    f_c = None
    if iteration > 1:
        C_fci = scf_res["fci_coefficients"]
        e = scf_res["fci_orb_energies"]
        f_c = C_fci.T @ (scf_res["fci_Fock"] - scf_res["fci_F_0"]) @ C_fci
        h_spin = C_fci.T @ result["h_spin"] @ C_fci
        g_spin = C_fci.T @ C_fci.T @ result["g_spin"] @ C_fci @ C_fci
    else:
        C = scf_res["coefficients"]
        e = [e_spin for e_orb in scf_res["orb_energies"] for e_spin in (e_orb, e_orb)]
        h_ao = scf_res["H_core"]
        h_mo = C.T @ h_ao @ C
        eri_mo = ao_to_mo_transform(scf_res["eri"], C)
        h_spin, g_spin = build_spin_orbital_integrals(h_mo, eri_mo)

    n_spin = h_spin.shape[0]
    occ = scf_res["n_elec"]
    virt = n_spin - occ
    n_occ = occ*(occ-1) // 2
    n_virt = virt*(virt-1) // 2
    full_dim = n_occ * n_virt + 1
    solution_dim = 1
    basis_dim = 1
    e_diff = np.empty(full_dim, dtype=np.float64, order='F')

    v_idx = 1
    for i in range(1, occ):
        for j in range(0, i):
            for a in range(occ+1, n_spin):
                for b in range(occ, a):
                    sum = e[a]+e[b]-e[i]-e[j]
                    e_diff[v_idx] = sum
                    v_idx += 1
    e_diff[0] = 0.000001

    # print(g_spin)
    # mat_prod = np.array([[0, -(g_spin[0,1,2,3]-g_spin[0,1,3,2])], [-(g_spin[2,3,0,1]-g_spin[2,3,1,0]), e[2]+e[3]-e[0]-e[1]-(g_spin[0,1,1,0]-g_spin[0,1,0,1])+(g_spin[2,3,3,2]-g_spin[2,3,2,3])]])
    # print(mat_prod)
    # print(np.linalg.eigh(mat_prod))

    # Set up and run libkrylov
    lk.initialize()
    lk.set_real_option("max_residual_norm", 10e-6)
    index = lk.add_space(lk.Kind.REAL, lk.Structure.SYMMETRIC, lk.Equation.EIGENVALUE,
                            full_dim, solution_dim, basis_dim)
    lk.set_space_preconditioner(index, 'j')
    lk.set_space_diagonal(index, e_diff)
    lk.set_real_space_vectors_from_diagonal(index, basis_dim, e_diff)
    lk.solve_real_equation(index, multiply, occ, virt, g_spin, e_diff, f_c)
    final_eig = lk.get_space_eigenvalues(index, solution_dim)
    print(f"Final eigenvalue: {final_eig}")
    with open("eigs", "a") as f:
        f.write(f"{final_eig}\n")
    ci_vec = lk.get_real_space_solutions(index, full_dim, solution_dim).flatten()
    # print(ci_vec)
    lk.finalize()

    # Calculate 1RDM
    one_rdm = calculate_1rdm(ci_vec, occ, virt)
    one_rdm_0 = np.zeros((n_spin, n_spin), dtype=np.float64)
    np.fill_diagonal(one_rdm_0[:occ, :occ], 1)

    occs, natural_orbitals, ndr_coeffs = get_natural_orbitals(one_rdm, scf_res["n_elec"])
    print(f"Trace: {np.trace(one_rdm)}")
    print("\nNatural Orbital Occupation Numbers:")
    for i, occ in enumerate(occs[:6]):
        print(f"  NO {i+1}: {occ:.8f}")

    J, K = calculate_JK(one_rdm, g_spin, n_spin)
    f_spin = h_spin + J + K

    # np.savetxt('it_1rdm.csv', one_rdm, "%.8f", ',')

    J, K = calculate_JK(one_rdm_0, g_spin, n_spin)
    f_0 = h_spin + J + K

    eigenvalues, eigenvectors = eigh(f_spin)
    
    print(f"NDR orbital energies: {eigenvalues}")

    scf_res["fci_coefficients"] = eigenvectors
    scf_res["fci_orb_energies"] = eigenvalues
    scf_res["fci_Fock"] = f_spin
    scf_res["fci_density"] = one_rdm
    scf_res["fci_F_0"] = f_0
    scf_res["F_c"] = f_spin - f_0
    scf_res["h_spin"] = h_spin
    scf_res["g_spin"] = g_spin

    return scf_res

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    Z = 2
    N = 2

    my_zetas = Zetas.TZVP

    # factor = 1.7/0.7

    # my_zetas = [
    #     [z / factor for z in shell] for shell in my_zetas
    # ]

    # result = run_calculation(Z, N, my_zetas, mode="ndr")

    # with open("fullci_tzvp_new.json", 'w') as file:
    #     json.dump(result, file, cls=NumpyEncoder)
    with open("fullci_tzvp_new.json", 'r') as file:
        result = json.load(file, object_hook=list_to_np)
    
    # print(f"HF Energies: {result['orb_energies']}")

    occ = N
    C = result["coefficients"]
    h_core = result["H_core"]
    h_mo = C.T @ h_core @ C
    eri = result["eri"]
    eri_mo = ao_to_mo_transform(result["eri"], result["coefficients"])
    h_spin, g_spin = build_spin_orbital_integrals(h_mo, eri_mo)
    n_spin = h_spin.shape[0]

    result["h_spin"] = h_spin
    result["g_spin"] = g_spin

    # print("F_ekt")
    # trdm = result["fci_two_rdm"]
    rdm = result["fci_density"]
    # F_ekt, F_ekt_1, F_ekt_2 = build_EKT_Fock(h_spin, rdm, g_spin, trdm)

    J, K = calculate_JK(rdm, g_spin, n_spin)

    F = h_spin + J + K

    result["fci_Fock"] = F

    eigenvalues, eigenvectors = eigh(rdm)
    print(eigenvalues)
    c0 = eigenvectors[:, -1]
    c1 = eigenvectors[:, -2]

    result["fci_coefficients"] = eigenvectors

    rdm_0 = np.outer(c0, c0) + np.outer(c1, c1)
    J, K = calculate_JK(rdm_0, g_spin, n_spin)
    F_0 = h_spin + J + K

    result["fci_F_0"] = F_0

    # eigenvalues, eigenvectors = eigh(F_ekt)

    # result["fci_orb_energies"] = eigenvalues
    run_ci(result, 2)

    # print("Canonical")
    # F_no = eigenvectors.T @ F @ eigenvectors
    # occ_vals, occ_vecs = eigh(F_no[:occ, :occ])
    # virt_vals, virt_vecs = eigh(F_no[occ:, occ:])
    # print(occ_vals)
    # print(virt_vals)
    
    # result["coefficients"] = block_diag(occ_vecs, virt_vecs)
    # result["fci_orb_energies"] = np.concatenate([occ_vals, virt_vals])
    # result["fci_Fock"] = F_no
    # result["fci_F_0"] = eigenvectors.T @ F_0 @ eigenvectors

    # result = run_ci(result, 2)