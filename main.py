import numpy as np
from src import make_basis
from src.solver.hf_scf import scf
from src.solver.ci_full import FullCISolver
from src.solver.orbital_opt import optimize_hf_orbitals

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

    # 2. RUN BASE HF (Required for both 'hf' and 'fci')
    print("--> Running SCF...")
    basis = make_basis(*working_zetas)
    scf_res = scf(basis, Z, N_elec)
    
    print(f"HF Total Energy: {scf_res['E_total']:.10f} Ha")

    # 3. FCI MODE
    if mode == "fci":
        print("--> Running Full Configuration Interaction...")
        ci_engine = FullCISolver(scf_res)
        ci_res = ci_engine.solve()
        
        print(f"FCI Total Energy: {ci_res['E_ci']:.10f} Ha")
        print(f"Correlation Energy: {ci_res['E_correlation']:.10f} Ha")
        return ci_res

    return scf_res

if __name__ == "__main__":

    Z = 2
    N = 2

    my_zetas = [
        [4.5, 2.0, 0.6], 
    ]

    result = run_calculation(Z, N, my_zetas, mode="fci")

    if "zetas" in str(result):
         print("\nFinal Optimized Zetas:")
         for i, l in enumerate(['s', 'p', 'd']):
             if i < len(my_zetas): print(f"  {l}: {np.round(result, 4)}")