import numpy as np
from src import make_basis
from src.solver.hf_scf import scf
from src.solver.ci_full import FullCISolver
from src.solver.orbital_opt import optimize_hf_orbitals
from src.solver.NDR import calculate_1rdm, get_natural_orbitals 

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
    scf_res = scf(basis, Z, N_elec)
    
    print(f"HF Total Energy: {scf_res['E_total']:.10f} Ha")

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
                for det, coeff in sorted_excitations[:5]: 
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
        [8.955016, 2.975601, 1.477575, 0.706409, 0.207456, 0.101581],
        [1.0],
        [0.5]
    ]

    result = run_calculation(Z, N, my_zetas, mode="ndr")
