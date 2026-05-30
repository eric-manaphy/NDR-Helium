import numpy as np
from scipy.linalg import eigh
from sympy import factorial
from src.solver.ci_full import FullCISolver

def calculate_1rdm_full(solver, ci_vector):

    n_spin = solver.n_spin

    dets = solver._make_determinants()
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

                det_i, phase = solver._apply_one_body(det_j, p, q)
                if det_i is not None and det_i in det_map:
                    i = det_map[det_i]
                    c_i = ci_vector[i]
                    rdm[p, q] += c_i * c_j * phase

    return rdm

def calculate_2rdm(solver, ci_vector):

    n_spin = solver.n_spin
    dets = solver._make_determinants()
    det_map = {det: i for i, det in enumerate(dets)}

    rdm2 = np.zeros((n_spin, n_spin, n_spin, n_spin))

    for j, det_j in enumerate(dets):
        c_j = ci_vector[j]
        if abs(c_j) < 1e-12: 
            continue
        for p in range(n_spin):
            for q in range(n_spin):
                for r in range(n_spin):
                    for s in range(n_spin):

                        det_i, phase = solver._apply_two_body(det_j, p, q, r, s)
                        if det_i is not None and det_i in det_map:
                            i = det_map[det_i]
                            c_i = ci_vector[i]
                            rdm2[p, q, r, s] += c_i * c_j * phase

    N = solver.n_elec
    nomalization = 1.0 / (N * (N - 1))

    return rdm2*nomalization

def partial_trace_2rdm(solver, rdm2):
    N = solver.n_elec

    contracted_rdm = np.einsum('pqrq->pr', rdm2)

    rdm1_from_2rdm = contracted_rdm
    return rdm1_from_2rdm

def get_natural_orbitals(rdm, n_elec):
    occupations, U = eigh(rdm)

    idx = np.argsort(occupations)[::-1]
    occupations = occupations[idx]
    natural_orbitals = U[:, idx]

    ndr_occupations = occupations[:n_elec]
    ndr_orbitals = natural_orbitals[:, :n_elec]

    return occupations, natural_orbitals, ndr_orbitals



