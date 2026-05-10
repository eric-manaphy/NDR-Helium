import numpy as np
from scipy.optimize import minimize
from src.solver.hf_scf import scf
from src.utils import pack_params, unpack_params
from src import make_basis # Assuming your basis generator is here

def objective_function(x, counts, Z, N_elec):

    try:
        zeta_lists = unpack_params(x, counts)
        

        basis = make_basis(*zeta_lists)
        
        result = scf(basis, Z, N_elec, conv=1e-8, damping=0.3)
        
        if not result.get("converged", True):
            return 100.0 
            
        return result["E_total"]

    except np.linalg.LinAlgError:
        return 100.0

def optimize_hf_orbitals(Z, N_elec, initial_zetas):

    counts = [len(z) for z in initial_zetas]
    x0 = pack_params(*initial_zetas)
    
    res = minimize(
        objective_function,
        x0,
        args=(counts, Z, N_elec),
        method='BFGS',
        options={'disp': True, 'gtol': 1e-5}
    )
    
    final_zetas = unpack_params(res.x, counts)
    return {
        "zetas": final_zetas,
        "energy": res.fun,
        "success": res.success,
        "message": res.message
    }

if __name__ == "__main__":

    Z, N = 3, 3
    init_zetas = [
        [4.0, 1.5, 0.5], 
        [1.0]           
    ]
    
    results = optimize_hf_orbitals(Z, N, init_zetas)
    
    print(f"\nOptimization Results for Z={Z}:")
    print(f"Final HF Energy: {results['energy']:.8f} Ha")
    for i, z_list in enumerate(results['zetas']):
        print(f"L={i} Zetas: {np.round(z_list, 4)}")