import numpy as np
from scipy.special import comb
from src import make_basis
from src.solver.hf_scf import scf
from src.solver.ci_full import FullCISolver
from src.solver.orbital_opt import optimize_hf_orbitals
from src.solver.NDR import calculate_1rdm, get_natural_orbitals
from src.utils import ao_to_mo_transform, build_spin_orbital_integrals, get_multi_index
import libkrylov as lk

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
    else:
        working_zetas = zetas

    # 2. RUN BASE HF 
    print("--> Running SCF...")
    basis = make_basis(*working_zetas)
    scf_res = scf(basis, Z, N_elec)
    
    print(f"HF Total Energy: {scf_res['E_total']:.10f} Ha")

    if mode in ["hf", "opt"]:
        f_mo = scf_res["coefficients"].T @ scf_res["Fock"] @ scf_res["coefficients"]
        eri_mo = ao_to_mo_transform(scf_res["eri"], scf_res["coefficients"])
        f_spin, g_spin = build_spin_orbital_integrals(f_mo, eri_mo)
        e = np.diag(f_spin)
        n_spin = f_spin.shape[0]
        occ = scf_res["n_elec"]
        virt = n_spin - occ
        # full_dim = occ*(occ-1)*virt*(virt-1)
        full_dim = comb(virt, occ, exact=True) + 1
        solution_dim = 1
        basis_dim = 1
        e_mat = np.empty(full_dim, dtype=np.float64, order='F')

        v_idx = 0
        for i in range(1, occ):
            for j in range(0, i):
                for a in range(occ+1, n_spin):
                    for b in range(occ, a):
                        sum = e[a]+e[b]-e[i]-e[j]
                        e_mat[v_idx] = sum
                        v_idx += 1
        e_mat[0] = 0.01

        lk.initialize()
        index = lk.add_space(lk.real_kind, lk.symmetric_structure, lk.eigenvalue_equation,
                             full_dim, solution_dim, basis_dim)
        # vectors = np.zeros(full_dim, dtype=np.float64, order='F')
        # vectors[5] = 1.0
        # vectors = np.random.rand(full_dim)
        # lk.set_real_space_vectors(index, vectors)
        lk.set_space_preconditioner(index, 'd')
        lk.set_space_diagonal(index, full_dim, e_mat)
        lk.set_real_space_vectors_from_diagonal(index, full_dim, basis_dim, e_mat)
        def multiply(m, n, vectors, products):
            # Calculate c_0, first term, and last term
            c_0 = vectors[0]
            d_0 = 0
            v_idx = 1
            for i in range(1, occ):
                for j in range(0, i):
                    for a in range(occ+1, n_spin):
                        for b in range(occ, a):
                            ijab = vectors[v_idx]
                            d_0 += 2*(g_spin[i,j,a,b]-g_spin[i,j,b,a])*ijab
                            # Not sure if I can guarantee products is already initalized
                            # So I'll just initialize (the relevant indices) here
                            products[v_idx] = e_mat[v_idx]*ijab - (g_spin[a,b,i,j]-g_spin[a,b,j,i])*c_0/2
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

            return 0
        lk.solve_real_equation(index, multiply)
        print(lk.get_space_eigenvalues(index, solution_dim))
        print(lk.get_real_space_solutions(index, full_dim, solution_dim))
        lk.finalize()
        return 0

    # 3. FCI & NDR MODE
    if mode in ["fci", "ndr"]:
        print("--> Running Full Configuration Interaction...")
        ci_engine = FullCISolver(scf_res)
        ci_res = ci_engine.solve()
        
        print(f"FCI Total Energy: {ci_res['E_ci']:.10f} Ha")
        print(f"Correlation Energy: {ci_res['E_correlation']:.10f} Ha")

        ground_state_vector = ci_res['vectors'][:, 0]
        print(f"Leading CI Coefficients: {ground_state_vector[0]:.4f}, {ground_state_vector[1]:.4f}, ...")

        
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
            rdm = calculate_1rdm(ci_engine, ground_state_vec)
            print(f"1RDM Computed. Trace (electrons): {np.trace(rdm):.4f}")

            # Diagonalize 1RDM to find Natural Orbitals and Occupations
            occs, natural_orbitals, ndr_coeffs = get_natural_orbitals(rdm, N_elec)
            
            print("\nNatural Orbital Occupation Numbers (top 6):")
            for i, occ in enumerate(occs[:6]):
                print(f"  NO {i+1}: {occ:.8f}")

            # The NDR consists of the most occupied natural orbitals
            print(f"\nNDR constructed from {N_elec} most occupied natural orbitals.")
            
            # Add NDR data to results dictionary
            ci_res['1rdm'] = rdm
            ci_res['no_occupations'] = occs
            ci_res['natural_orbitals'] = natural_orbitals

        return ci_res

    return scf_res

if __name__ == "__main__":

    Z = 2
    N = 2

    my_zetas = [
        [4.5, 2.0, 0.6], 
    ]

    result = run_calculation(Z, N, my_zetas, mode="opt")
    # h_mo = result["coefficients"].T @ result["H_core"] @ result["coefficients"]
    # eri_mo = ao_to_mo_transform(result["eri"], result["coefficients"])
    # h_spin, g_spin = build_spin_orbital_integrals(h_mo, eri_mo)
    # f_mo = result["coefficients"].T @ result["Fock"] @ result["coefficients"]
    # eri_mo = ao_to_mo_transform(result["eri"], result["coefficients"])
    # f_spin, _ = build_spin_orbital_integrals(f_mo, eri_mo)