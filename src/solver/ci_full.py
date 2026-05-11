import numpy as np
from scipy.linalg import eigh
from itertools import combinations
from src.utils import ao_to_mo_transform, build_spin_orbital_integrals, get_excitation_level

class FullCISolver:
    def __init__(self, scf_result):
        self.scf_result = scf_result
        self.C = scf_result["coefficients"]
        self.h_ao = scf_result["H_core"]
        self.eri_ao = scf_result["eri"]
        self.n_elec = scf_result.get("n_elec", 2) # Default to 2 if not found
        
        self.h_mo = self.C.T @ self.h_ao @ self.C
        self.eri_mo = ao_to_mo_transform(self.eri_ao, self.C)
        
        self.h_spin, self.g_spin = build_spin_orbital_integrals(self.h_mo, self.eri_mo)
        self.n_spin = self.h_spin.shape[0]

    def _make_determinants(self):
    
        dets = []
        for occ in combinations(range(self.n_spin), self.n_elec):
            det = 0
            for p in occ:
                det |= (1 << p)
            dets.append(det)
        return dets

    def _apply_one_body(self, det, p, q):
        if not (det & (1 << q)): return None, 0
        phase1 = (-1) ** (det & ((1 << q) - 1)).bit_count()
        det_tmp = det ^ (1 << q)
        
        if det_tmp & (1 << p): return None, 0
        phase2 = (-1) ** (det_tmp & ((1 << p) - 1)).bit_count()
        return det_tmp | (1 << p), phase1 * phase2

    def _apply_two_body(self, det, p, q, r, s):
        d1, s1 = self._annihilate(det, r)
        if d1 is None: return None, 0
        d2, s2 = self._annihilate(d1, s)
        if d2 is None: return None, 0
        d3, s3 = self._create(d2, q)
        if d3 is None: return None, 0
        d4, s4 = self._create(d3, p)
        if d4 is None: return None, 0
        return d4, s1 * s2 * s3 * s4


    def _annihilate(self, det, p):
        if not (det & (1 << p)): return None, 0
        phase = (-1) ** (det & ((1 << p) - 1)).bit_count()
        return det ^ (1 << p), phase

    def _create(self, det, p):
        if det & (1 << p): return None, 0
        phase = (-1) ** (det & ((1 << p) - 1)).bit_count()
        return det | (1 << p), phase

    def solve(self):
        dets = self._make_determinants()
        n_det = len(dets)
        det_map = {det: i for i, det in enumerate(dets)}
        H_ci = np.zeros((n_det, n_det))

        for j, det_j in enumerate(dets):
  
            for p in range(self.n_spin):
                for q in range(self.n_spin):
                    val = self.h_spin[p, q]
                    if abs(val) < 1e-12: continue
                    det_i, phase = self._apply_one_body(det_j, p, q)
                    if det_i in det_map:
                        H_ci[det_map[det_i], j] += phase * val

            for p in range(self.n_spin):
                for q in range(self.n_spin):
                    for r in range(self.n_spin):
                        for s in range(self.n_spin):
                            val = self.g_spin[p, q, r, s]
                            if abs(val) < 1e-12: continue
                            det_i, phase = self._apply_two_body(det_j, p, q, r, s)
                            if det_i in det_map:
                                H_ci[det_map[det_i], j] += 0.5 * phase * val

        H_ci = 0.5 * (H_ci + H_ci.T)
        energies, vectors = eigh(H_ci)

        # Extract the ground state CI vector
        groun_state_vector = vectors[:, 0]

        hf_det = dets[0]

        cisd_coeffs = {
            "C_0": 0.0,
            "singles": [],
            "doubles": [],
            "others": []
        }

        for idx, det in enumerate(dets):
            coeff = groun_state_vector[idx]
            level = get_excitation_level(det, hf_det)

            if level == 0:
                cisd_coeffs["C_0"] = coeff
            elif level == 1:
                cisd_coeffs["singles"].append((det, coeff))
            elif level == 2:
                cisd_coeffs["doubles"].append((det, coeff))
            else:
                cisd_coeffs["others"].append((det, coeff))
        
        return {
            "E_ci": energies[0],
            "E_correlation": energies[0] - self.scf_result["E_total"],
            "vectors": vectors,
            "determinants": dets,
            "cisd_coeffs": cisd_coeffs
        }