class BasisFunction:
    def __init__(self, zeta, l, m):
        self.zeta = float(zeta)
        self.l = int(l)
        self.m = int(m)

    def __repr__(self):
        return f"BasisFunction(zeta={self.zeta}, l={self.l}, m={self.m})"


def make_basis(*zeta_lists):
    bf = []

    for l, zetas in enumerate(zeta_lists):
        for zeta in zetas:
            for m in range(-l, l + 1):
                bf.append(BasisFunction(zeta, l, m))
    return bf

def group_basis_by_lm(basis):
    lm_indices = {}
    for i, bf in enumerate(basis):
        lm_indices.setdefault((bf.l, bf.m), []).append(i)

    l_values = sorted(set(bf.l for bf in basis))

    radial_indices = {}
    for l in l_values:
        if (l, 0) not in lm_indices:
            raise ValueError(f"No m=0 function for l = {l}")
        radial_indices[l] = lm_indices[(l, 0)]

        n_rad = len(radial_indices[l])
        for m in range(-l, l + 1):
            assert len(lm_indices[(l, m)]) == n_rad

    return l_values, radial_indices, lm_indices
