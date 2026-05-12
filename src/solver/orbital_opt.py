import numpy as np
from scipy.optimize import minimize
from src.integrals.one_electron import kinetic_integral, overlap_integral, potential_integral
from src.integrals.two_electron import electron_repulsion_integral
from src.solver.hf_scf import scf
from src.utils import pack_params, unpack_params, build_param_maps
from src import make_basis 

def objective_function(x, counts, Z, N_elec):
    grad = np.zeros(len(x))
    try:
        zeta_lists = unpack_params(x, counts)
        

        basis = make_basis(*zeta_lists)
        
        result = scf(basis, Z, N_elec, conv=1e-8, damping=0.3)
        
        if not result.get("converged", True):
            return 100.0, grad
            
        E_tot = result["E_total"]
        D = result["density"]
        F = result["Fock"]

        W = D @ F @ D


        _, _, _, param_to_aos, _ = build_param_maps(basis)

        


        for p_idx, ao_indices in enumerate(param_to_aos):
            dE_dzeta = 0.0
            for mu in ao_indices:
                bi = basis[mu]
                for nu in range(len(basis)):

                    bj = basis[nu]

                    # Derivative of one-electron integrals
                    _, dt_di, _ = kinetic_integral(bi, bj, deriv=True)
                    _, dv_di, _ = potential_integral(bi, bj, Z, deriv=True)
                    _, ds_di, _ = overlap_integral(bi, bj, deriv=True)

                    # Contribution to the gradient
                    dE_dzeta += 2.0 * D[mu, nu] * (dt_di + dv_di)
                    dE_dzeta -= 2.0 * W[mu, nu] * ds_di

                    # Two electron contribution
                    for nu in range(len(basis)):
                        bj = basis[nu]
                        for la in range(len(basis)):
                            bk = basis[la]
                            for si in range(len(basis)):
                                bl = basis[si]
                                
                                # Call your ERI derivative logic
                                # Note: your electron_repulsion_integral needs to support deriv=True
                                # This returns d(mu,nu|la,si)/d_zeta_mu
                                _, deri_dmu = electron_repulsion_integral(bi, bj, bk, bl, deriv=True)
                                
                                # Combine J and K terms
                                # The factor of 2.0 accounts for the mu-index symmetry
                                val = D[mu, nu] * D[la, si] * (deri_dmu)
                                exc = D[mu, la] * D[nu, si] * (deri_dmu)
                                
                                dE_dzeta += 2.0 * (val - 0.5 * exc)

            grad[p_idx] = dE_dzeta * np.exp(x[p_idx])

        return E_tot, grad


    except np.linalg.LinAlgError:
        return 100.0, grad

def optimize_hf_orbitals(Z, N_elec, initial_zetas):

    counts = [len(z) for z in initial_zetas]
    x0 = pack_params(*initial_zetas)
    
    res = minimize(
        objective_function,
        x0,
        args=(counts, Z, N_elec),
        method='BFGS',
        jac=True,
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