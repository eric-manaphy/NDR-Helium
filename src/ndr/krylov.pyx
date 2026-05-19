from enum import Enum
import threading
import numpy as np
cimport numpy as cnp
from ctypes cimport int_t, real_t, complex_t, char_t, bool_t
cimport ckrylov as ck

# Relevant Enum strings
class Kind(str, Enum):
    REAL    = 'r'
    COMPLEX = 'c'

class Structure(str, Enum):
    SYMMETRIC = 's'
    HERMITIAN = 'h'

class Equation(str, Enum):
    EIGENVALUE      = 'e'
    LINEAR          = 'l'
    SYLVESTER       = 's'
    SHIFTED_LINEAR  = 'h'

# Not sure how to use assert correctly
# so this is for error handling
_ERRORS = {
    ck.CKRYLOV_OK:    "OK",
    ck.CKRYLOV_NO_OPTIONS:    "NO_OPTIONS",
    ck.CKRYLOV_NO_SUCH_OPTION:    "NO_SUCH_OPTION",
    ck.CKRYLOV_INVALID_OPTION:    "INVALID_OPTION",
    ck.CKRYLOV_INVALID_KIND:    "INVALID_KIND",
    ck.CKRYLOV_INVALID_STRUCTURE:    "INVALID_STRUCTURE",
    ck.CKRYLOV_INVALID_EQUATION:    "INVALID_EQUATION",
    ck.CKRYLOV_INVALID_DIMENSION:    "INVALID_DIMENSION",
    ck.CKRYLOV_INVALID_INPUT:    "INVALID_INPUT",
    ck.CKRYLOV_NO_SPACES:    "NO_SPACES",
    ck.CKRYLOV_NO_SUCH_SPACE:    "NO_SUCH_SPACE",
    ck.CKRYLOV_INCOMPATIBLE_SPACE:    "INCOMPATIBLE_SPACE",
    ck.CKRYLOV_NOTHING_TO_DO:    "NOTHING_TO_DO",
    ck.CKRYLOV_NO_ITERATIONS:    "NO_ITERATIONS",
    ck.CKRYLOV_NO_SUCH_ITERATION:    "NO_SUCH_ITERATION",
    ck.CKRYLOV_NO_SOLUTIONS:    "NO_SOLUTIONS",
    ck.CKRYLOV_NO_SUCH_SOLUTION:    "NO_SUCH_SOLUTION",
    ck.CKRYLOV_INVALID_CONFIGURATION:    "INVALID_CONFIGURATION",
    ck.CKRYLOV_INCOMPLETE_CONFIGURATION:    "INCOMPLETE_CONFIGURATION",
    ck.CKRYLOV_INCOMPATIBLE_CONFIGURATION:    "INCOMPATIBLE_CONFIGURATION",
    ck.CKRYLOV_INCOMPATIBLE_EQUATION:    "INCOMPATIBLE_EQUATION",
    ck.CKRYLOV_INCOMPLETE_EQUATION:    "INCOMPLETE_EQUATION",
    ck.CKRYLOV_INCOMPATIBLE_PRECONDITIONER:    "INCOMPATIBLE_PRECONDITIONER",
    ck.CKRYLOV_INCOMPLETE_PRECONDITIONER:    "INCOMPLETE_PRECONDITIONER",
    ck.CKRYLOV_INCOMPATIBLE_ORTHONORMALIZER:    "INCOMPATIBLE_ORTHONORMALIZER",
    ck.CKRYLOV_LINEAR_ALGEBRA_ERROR:    "LINEAR_ALGEBRA_ERROR",
    ck.CKRYLOV_NOT_CONVERGED:    "NOT_CONVERGED",
    ck.CKRYLOV_MAX_ITERATIONS_REACHED:    "MAX_ITERATIONS_REACHED",
    ck.CKRYLOV_MAX_DIM_REACHED:    "MAX_DIM_REACHED",
    ck.CKRYLOV_ILL_CONDITIONED:    "ILL_CONDITIONED",
    ck.CKRYLOV_NO_BASIS_UPDATE:    "NO_BASIS_UPDATE",
    ck.CKRYLOV_LINEARLY_DEPENDENT_BASIS:    "LINEARLY_DEPENDENT_BASIS",
    ck.CKRYLOV_KEY_NOT_FOUND:    "KEY_NOT_FOUND",
}

class LibkrylovError(RuntimeError):
    def __init__(self, int error):
        self.error = error
        name = _ERRORS.get(error, "NOT_FOUND")
        super().__init__(f"Error {error}: {name}")

cdef inline void _assert(int_t error) except *:
    if error != ck.CKRYLOV_OK:
        raise LibkrylovError(error)

# Initialize libkrylov
def initialize():
    cdef int_t error = ck.ckrylov_initialize()
    _assert(error)

# Finalize libkrylov
def finalize():
    cdef int_t error = ck.ckrylov_finalize()
    _assert(error)

# Adds new Krylov subspace solver
def add_space(kind, structure, equation,
              int_t full_dim, int_t solution_dim, int_t basis_dim):
    # Apparently bytes is the easiest way to convert to char *
    cdef bytes k = kind.value.encode('ascii')
    cdef bytes s = structure.value.encode('ascii')
    cdef bytes e = equation.value.encode('ascii')

    # Negative indices are errors
    cdef int_t index = ck.ckrylov_add_space(
        k, len(k), s, len(s), e, len(e),
        full_dim, solution_dim, basis_dim
    )
    if index < 0: _assert(index)

    return index

# Get space kind. Implemented for QoL.
# I think this might be bugged from the libkrylov side
def get_space_kind(int_t index):
    cdef char_t k_p = 0
    # The header file says len isn't a pointer...
    # Since I don't have kind_len, I just have to hardcode the length
    cdef int_t error = ck.ckrylov_get_space_kind(index, &k_p, 1)
    _assert(error)
    return Kind(chr(k_p))

# Sets the preconditioner
def set_space_preconditioner(int_t index, str preconditioner):
    cdef bytes p = preconditioner.encode('ascii')

    cdef int_t error = ck.ckrylov_set_space_preconditioner(
        index, p, len(p)
    )
    _assert(error)

# Sets the diagonal matrix for preconditioning
def set_space_diagonal(int_t index, cnp.ndarray matrix):
    # TODO: Make the type not hardcoded
    matrix = np.asfortranarray(matrix, dtype=np.float64)

    cdef int_t full_dim = matrix.shape[0]
    cdef const real_t[::1, :] m_p = matrix

    cdef int_t error = ck.ckrylov_set_space_diagonal(
        index, full_dim, &m_p[0,0]
    )
    _assert(error)

# Sets starting vectors from diagonal matrix
def set_real_space_vectors_from_diagonal(int_t index, int_t basis_dim, cnp.ndarray matrix):
    # TODO: Make the type not hardcoded
    matrix = np.asfortranarray(matrix, dtype=np.float64)

    cdef int_t full_dim = matrix.shape[0]
    cdef const real_t[::1, :] m_p = matrix

    cdef int_t error = ck.ckrylov_set_real_space_vectors_from_diagonal(
        index, full_dim, basis_dim, &m_p[0,0]
    )
    _assert(error)

# Set starting subspace vectors
def set_real_space_vectors(int_t index, cnp.ndarray vectors):
    # The C example just uses a 1D array so might as well check for that
    if vectors.ndim == 1: vectors = vectors[:, np.newaxis] # or just reshape(-1,1)

    # Since it seems like it's still Fortran
    # we need to make these Fortran-order arrays
    # I don't think whatever wrapping was done
    # was optimized for C, but I could be wrong
    # TODO: Make the type not hardcoded
    vectors = np.asfortranarray(vectors, dtype=np.float64)

    # It seems easier to just define the dimensions inside
    # since I believe they're cached in Python anyway
    cdef int_t full_dim = vectors.shape[0]
    cdef int_t basis_dim = vectors.shape[1]
    # This stride syntax is specifically for Fortran arrays
    cdef const real_t[::1, :] v_p = vectors

    cdef int_t error = ck.ckrylov_set_real_space_vectors(
        index, full_dim, basis_dim, &v_p[0,0]
    )
    _assert(error)

def set_complex_space_vectors(int_t index, cnp.ndarray vectors):
    if vectors.ndim == 1: vectors = vectors[:, np.newaxis]

    # np.complex128 should be the same as complex double
    # TODO: Make the type not hardcoded
    vectors = np.asfortranarray(vectors, dtype=np.complex128)

    cdef int_t full_dim = vectors.shape[0]
    cdef int_t basis_dim = vectors.shape[1]
    # TODO: Make this also not hardcoded
    cdef const double complex[::1, :] v_p = vectors

    # Cast to complex_t
    cdef int_t error = ck.ckrylov_set_complex_space_vectors(
        index, full_dim, basis_dim, <const complex_t *>&v_p[0,0]
    )
    _assert(error)

# Generic version
def set_space_vectors(int_t index, cnp.ndarray vectors):
    if get_space_kind(index) == Kind.REAL:
        set_real_space_vectors(index, vectors)
    else:
        set_complex_space_vectors(index, vectors)

# TODO: Mulitply functions

# If I want to use Python in the callback
# it has to be thread-safe (and global)
# so use thread-local storage
_tls = threading.local()

# Wrapper functions for the Python callbacks
# Might need to be mindful of the GIL because of how the threading works
# Also, perhaps obviously, C can't handle Python exceptions
cdef int_t _real_wrapper(const int_t *full_dim, const int_t *subset_dim,
                          const real_t *vectors, real_t *products) noexcept nogil:
    # nogil in the function signature just means that it can run without GIL
    # not that it releases it, but we still need to manipulate Python objects
    with gil:
        # Get dimensions from pointers
        full = full_dim[0]
        subset = subset_dim[0]
        size = full * subset

        # Reshape the vectors and products
        # It seems the best way to convert
        # to an array is just casting the pointer like so
        v = np.asarray(<real_t[:size:1]>vectors, order='F')
        p = np.asarray(<real_t[:size:1]>products, order='F')
        error = _tls.multiply(v, p)
        iter = get_space_num_iterations(1)
        print(iter, '\n', get_space_eigenvalues(1, 1), '\n', v, '\n', p)
        return error

def solve_real_equation(int_t index, multiply):
    _tls.multiply = multiply
    try:
        error = ck.ckrylov_solve_real_equation(index, _real_wrapper)
        _assert(error)
    finally:
        del _tls.multiply

cdef int_t _complex_wrapper(const int_t *full_dim, const int_t *subset_dim,
                          const complex_t *vectors, complex_t *products) noexcept nogil:
    with gil:
        full = full_dim[0]
        subset = subset_dim[0]
        size = full * subset

        # We need to cast complex_t * to a proper type first
        # TODO: Make the type not hardcoded
        v = np.asarray(<double complex[:size:1]>(<double complex *>vectors), order='F')
        p = np.asarray(<double complex[:size:1]>(<double complex *>products), order='F')

        return _tls.multiply(v, p)

def solve_complex_equation(int_t index, multiply):
    _tls.multiply = multiply
    try:
        error = ck.ckrylov_solve_complex_equation(index, _complex_wrapper)
        _assert(error)
    finally:
        del _tls.multiply

def get_space_num_iterations(int_t index):
    return ck.ckrylov_get_space_num_iterations(index)

# def get_space_iteration_residual_norms(int_t index, int_t iter, int_t solution_dim):
#     residuals = np.empty(solution_dim, dtype=np.float64, order='F')
#     cdef real_t[::1] r_p = residuals

#     cdef int_t error = ck.ckrylov_get_space_iteration_residual_norms(
#         index, iter, solution_dim, &r_p[0]
#     )
#     _assert(error)

#     return residuals

# Get eigenvalues. Only works for eigenvalue problems
# TODO: Make the types not hardcoded
def get_space_eigenvalues(int_t index, int_t solution_dim):
    # It's probably more efficient to do empty(?)
    eigenvalues = np.empty(solution_dim, dtype=np.float64)
    # Eigenvalues have to be real (according to docs, anyway)
    cdef real_t[::1] e_p = eigenvalues

    cdef int_t error = ck.ckrylov_get_space_eigenvalues(
        index, solution_dim, &e_p[0]
    )
    _assert(error)
    
    return eigenvalues

# Get solutions
# TODO: Make the types not hardcoded
def get_real_space_solutions(int_t index, int_t full_dim, int_t solution_dim):
    # Fortran ordering still
    solutions = np.empty((full_dim, solution_dim), dtype=np.float64, order='F')
    cdef real_t[::1, :] s_p = solutions

    cdef int_t error = ck.ckrylov_get_real_space_solutions(
        index, full_dim, solution_dim, &s_p[0,0]
    )
    _assert(error)

    return solutions

def get_complex_space_solutions(int_t index, int_t full_dim, int_t solution_dim):
    solutions = np.empty((full_dim, solution_dim), dtype=np.complex128, order='F')
    cdef double complex[::1, :] s_p = solutions

    cdef int_t error = ck.ckrylov_get_complex_space_solutions(
        index, full_dim, solution_dim, <complex_t *>&s_p[0,0]
    )
    _assert(error)

    return solutions

# Generic version
def get_space_solutions(int_t index, int_t full_dim, int_t solution_dim):
    if get_space_kind(index) == Kind.REAL:
        return get_real_space_solutions(index, full_dim, solution_dim)
    else:
        return get_complex_space_solutions(index, full_dim, solution_dim)