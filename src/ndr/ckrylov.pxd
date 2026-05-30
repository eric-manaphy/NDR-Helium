from ctypes cimport int_t, real_t, complex_t, char_t, bool_t

cdef extern from "include/ckrylov.h":
    # Multiplication function interfaces
    # Since they're callbacks they should probably be pointers(?)
    ctypedef int_t (*real_multiply_t)(const int_t *full_dim, const int_t *subset_dim,
                    const real_t *vectors, real_t *products)
    ctypedef int_t (*complex_multiply_t)(const int_t *full_dim, const int_t *subset_tim,
                       const complex_t *vectors, complex_t *products)
    ctypedef int_t (*real_block_multiply_t)(const int_t *num_spaces, const int_t *total_size,
                            const int_t *full_dims, const int_t *subset_dims,
                            const int_t *offsets, const real_t *vectors,
                            real_t *products)
    ctypedef int_t (*complex_block_multiply_t)(const int_t *num_spaces, const int_t *total_size,
                                const int_t *full_dims, const int_t *subset_dims,
                                const int_t *offsets, const complex_t *vectors,
                                complex_t *products)

    # C-stype interface functions
    int_t ckrylov_initialize()
    int_t ckrylov_finalize()
    int_t ckrylov_get_version(char_t *version, int_t version_len)
    int_t ckrylov_get_compiled_datetime(char_t *compiled_dt, int_t compiled_dt_len)
    int_t ckrylov_get_current_datetime(char_t *current_dt, int_t current_dt_len)
    int_t ckrylov_header()
    int_t ckrylov_footer()
    int_t ckrylov_set_defaults()
    int_t ckrylov_get_integer_option(const char_t *key, int_t key_len)
    int_t ckrylov_set_integer_option(const char_t *key, int_t key_len, int_t value)
    int_t ckrylov_get_string_option(const char_t *key, int_t key_len, char_t *value,
                                    int_t value_len)
    int_t ckrylov_set_string_option(const char_t *key, int_t key_len,
                                    const char_t *value, int_t value_len)
    int_t ckrylov_length_string_option(const char_t *key, int_t key_len)
    real_t ckrylov_get_real_option(const char_t *key, int_t key_len)
    int_t ckrylov_set_real_option(const char_t *key, int_t key_len, real_t value)
    bool_t ckrylov_get_logical_option(const char_t *key, int_t key_len)
    int_t ckrylov_set_logical_option(const char_t *key, int_t key_len,
                                    bool_t value)
    int_t ckrylov_get_enum_option(const char_t *key, int_t key_len,
                                const char_t *value, int_t value_len)
    int_t ckrylov_set_enum_option(const char_t *key, int_t key_len,
                                const char_t *value, int_t value_len)
    int_t ckrylov_length_enum_option(const char_t *key, int_t key_len)
    int_t ckrylov_define_enum_option(const char_t *key, int_t key_len,
                                    const char_t *values, int_t values_len)
    int_t ckrylov_validate_enum_option(const char_t *key, int_t key_len,
                                    const char_t *value, int_t value_len)
    int_t ckrylov_add_space(const char_t *kind, int_t kind_len,
                            const char_t *structure, int_t structure_len,
                            const char_t *equation, int_t equation_len,
                            int_t full_dim, int_t solution_dim, int_t basis_dim)
    int_t ckrylov_get_num_spaces()
    int_t ckrylov_get_space_integer_option(int_t index, const char_t *key,
                                        int_t key_len)
    int_t ckrylov_set_space_integer_option(int_t index, const char_t *key,
                                        int_t key_len, int_t value)
    int_t ckrylov_get_space_string_option(int_t index, const char_t *key,
                                        int_t key_len, char_t *value,
                                        int_t value_len)
    int_t ckrylov_set_space_string_option(int_t index, const char_t *key,
                                        int_t key_len, const char_t *value,
                                        int_t value_len)
    int_t ckrylov_length_space_string_option(int_t index, const char_t *key,
                                            int_t key_len)
    real_t ckrylov_get_space_real_option(int_t index, const char_t *key,
                                        int_t key_len)
    int_t ckrylov_set_space_real_option(int_t index, const char_t *key,
                                        int_t key_len, real_t value)
    bool_t ckrylov_get_space_logical_option(int_t index, const char_t *key,
                                            int_t key_len)
    int_t ckrylov_set_space_logical_option(int_t index, const char_t *key,
                                        int_t key_len, bool_t value)
    int_t ckrylov_get_space_enum_option(int_t index, const char_t *key,
                                        int_t key_len, char_t *value,
                                        int_t value_len)
    int_t ckrylov_set_space_enum_option(int_t index, const char_t *key,
                                        int_t key_len, const char_t *value,
                                        int_t value_len)
    int_t ckrylov_length_space_enum_option(int_t index, const char_t *key,
                                        int_t key_len)
    int_t ckrylov_define_space_enum_option(int_t index, const char_t *key,
                                        int_t key_len, const char_t *values,
                                        int_t values_len)
    int_t ckrylov_validate_space_enum_option(int_t index, const char_t *key,
                                            int_t key_len, const char_t *value,
                                            int_t value_len)
    int_t ckrylov_get_space_kind(int_t index, char_t *kind, int_t kind_len)
    int_t ckrylov_get_space_structure(int_t index, char_t *structure,
                                    int_t structure_len)
    int_t ckrylov_get_space_equation(int_t index, char_t *equation,
                                    int_t equation_len)
    int_t ckrylov_get_space_preconditioner(int_t index, char_t *preconditioner,
                                        int_t preconditioner_len)
    int_t ckrylov_set_space_preconditioner(int_t index,
                                        const char_t *preconditioner,
                                        int_t preconditioner_len)
    int_t ckrylov_get_space_orthonormalizer(int_t index, char_t *orthonormalizer,
                                            int_t orthonormalizer_len)
    int_t ckrylov_set_space_orthonormalizer(int_t index,
                                            const char_t *orthonormalizer,
                                            int_t orthonormalizer_len)
    int_t ckrylov_get_space_full_dim(int_t index)
    int_t ckrylov_get_space_solution_dim(int_t index)
    int_t ckrylov_get_space_basis_dim(int_t index)
    int_t ckrylov_get_space_vector_size(int_t index)
    int_t ckrylov_get_space_solution_size(int_t index)
    int_t ckrylov_get_space_eigenvalue_size(int_t index)
    int_t ckrylov_get_space_rhs_size(int_t index)
    int_t ckrylov_get_space_sylvester_b_size(int_t index)
    int_t ckrylov_get_space_num_iterations(int_t index)
    int_t ckrylov_get_space_iteration_basis_dim(int_t index, int_t iter)
    real_t ckrylov_get_space_iteration_gram_rcond(int_t index, int_t iter)
    int_t ckrylov_get_space_iteration_expectation_vals(int_t index, int_t iter,
                                                    int_t solution_dim,
                                                    real_t *expectation_vals)
    real_t ckrylov_get_space_iteration_lagrangian(int_t index, int_t iter)
    int_t ckrylov_get_space_iteration_residual_norms_(int_t index, int_t iter,
                                                    int_t solution_dim,
                                                    real_t *residual_norms)
    real_t ckrylov_get_space_iteration_max_residual_norms(int_t index, int_t iter)
    int_t ckrylov_get_space_last_basis_dim(int_t index)
    int_t ckrylov_get_space_last_expectation_vals(int_t index, int_t solution_dim,
                                                real_t *expectation_vals)
    real_t ckrylov_get_space_last_lagrangian(int_t index)
    int_t ckrylov_get_space_last_residual_norms(int_t index, int_t solution_dim,
                                                real_t *residual_norms)
    real_t ckrylov_get_space_last_max_residual_norm(int_t index)
    real_t ckrylov_get_space_last_gram_rcond(int_t index)
    int_t ckrylov_get_space_convergence(int_t index)
    int_t ckrylov_get_space_eigenvalues(int_t index, int_t solution_dim,
                                        real_t *eigenvalues)
    int_t ckrylov_set_space_diagonal(int_t index, int_t full_dim,
                                    const real_t *diagonal)
    int_t ckrylov_get_space_shifts(int_t index, int_t solution_dim, real_t *shifts)
    int_t ckrylov_set_space_shifts(int_t index, int_t solution_dim,
                                const real_t *shifts)
    int_t ckrylov_get_real_space_vectors(int_t index, int_t full_dim,
                                        int_t basis_dim, real_t *vectors)
    int_t ckrylov_set_real_space_vectors(int_t index, int_t full_dim,
                                        int_t basis_dim, const real_t *vectors)
    int_t ckrylov_set_real_space_vectors_from_diagonal(int_t index, int_t full_dim,
                                                    int_t basis_dim,
                                                    const real_t *diagonal)
    int_t ckrylov_resize_real_space_vectors(int_t index, int_t basis_dim)
    int_t ckrylov_get_real_space_products(int_t index, int_t full_dim,
                                        int_t basis_dim, real_t *products)
    int_t ckrylov_get_real_space_solutions(int_t index, int_t full_dim,
                                        int_t solution_dim,
                                        const real_t *solutions)
    int_t ckrylov_resize_real_space_solutions(int_t index, int_t solution_dim)
    int_t ckrylov_get_real_space_rhs(int_t index, int_t full_dim,
                                    int_t solution_dim, real_t *rhs)
    int_t ckrylov_set_real_space_rhs(int_t index, int_t full_dim,
                                    int_t solution_dim, const real_t *rhs)
    int_t ckrylov_get_real_space_sylvester_b(int_t index, int_t solution_dim,
                                            real_t *sylvester_b)
    int_t ckrylov_set_real_space_sylvester_b(int_t index, int_t solution_dim,
                                            const real_t *sylvester_b)
    int_t ckrylov_get_real_space_subset_vectors(int_t index, int_t full_dim,
                                                int_t skip_dim, int_t subset_dim,
                                                real_t *vectors)
    int_t ckrylov_set_real_space_subset_vectors(int_t index, int_t full_dim,
                                                int_t skip_dim, int_t subset_dim,
                                                const real_t *vectors)
    int_t ckrylov_get_real_space_subset_products(int_t index, int_t full_dim,
                                                int_t skip_dim, int_t subset_dim,
                                                real_t *products)
    int_t ckrylov_set_real_space_subset_products(int_t index, int_t full_dim,
                                                int_t skip_dim, int_t subset_dim,
                                                const real_t *products)
    int_t ckrylov_get_complex_space_vectors(int_t index, int_t full_dim,
                                            int_t basis_dim, complex_t *vectors)
    int_t ckrylov_set_complex_space_vectors(int_t index, int_t full_dim,
                                            int_t basis_dim,
                                            const complex_t *vectors)
    int_t ckrylov_set_complex_space_vectors_from_diagonal(int_t index,
                                                        int_t full_dim,
                                                        int_t basis_dim,
                                                        const real_t *diagonal)
    int_t ckrylov_resize_complex_space_vectors(int_t index, int_t basis_dim)
    int_t ckrylov_get_complex_space_products(int_t index, int_t full_dim,
                                            int_t basis_dim, complex_t *products)
    int_t ckrylov_get_complex_space_solutions(int_t index, int_t full_dim,
                                            int_t solution_dim,
                                            complex_t *solutions)
    int_t ckrylov_resize_complex_space_solutions(int_t index, int_t solution_dim)
    int_t ckrylov_get_complex_space_rhs(int_t index, int_t full_dim,
                                        int_t solution_dim, complex_t *rhs)
    int_t ckrylov_set_complex_space_rhs(int_t index, int_t full_dim,
                                        int_t solution_dim, const complex_t *rhs)
    int_t ckrylov_get_complex_space_sylvester_b(int_t index, int_t solution_dim,
                                                complex_t *sylvester_b)
    int_t ckrylov_set_complex_space_sylvester_b(int_t index, int_t solution_dim,
                                                const complex_t *sylvester_b)
    int_t ckrylov_get_complex_space_subset_vectors(int_t index, int_t full_dim,
                                                int_t skip_dim, int_t subset_dim,
                                                complex_t *vectors)
    int_t ckrylov_set_complex_space_subset_vectors(int_t index, int_t full_dim,
                                                int_t skip_dim, int_t subset_dim,
                                                const complex_t *vectors)
    int_t ckrylov_get_complex_space_subset_products(int_t index, int_t full_dim,
                                                    int_t skip_dim,
                                                    int_t subset_dim,
                                                    complex_t *products)
    int_t ckrylov_set_complex_space_subset_products(int_t index, int_t full_dim,
                                                    int_t skip_dim,
                                                    int_t subset_dim,
                                                    const complex_t *products)
    int_t ckrylov_get_real_block_total_size()
    int_t ckrylov_get_real_block_dims(int_t num_spaces, int_t *full_dims,
                                    int_t *subset_dims, int_t *offsets)
    int_t ckrylov_get_real_block_vectors(int_t num_spaces, int_t total_size,
                                        const int_t *full_dims,
                                        const int_t *subset_dims,
                                        const int_t *offsets, real_t *vectors)
    int_t ckrylov_set_real_block_products(int_t num_spaces, int_t total_size,
                                        const int_t *full_dims,
                                        const int_t *subset_dims,
                                        const int_t *offsets,
                                        const real_t *products)
    int_t ckrylov_get_real_block_convergence()
    int_t ckrylov_get_complex_block_total_size()
    int_t ckrylov_get_complex_block_dims(int_t num_spaces, int_t *full_dims,
                                        int_t *subset_dims, int_t *offsets)
    int_t ckrylov_get_complex_block_vectors(int_t num_spaces, int_t total_size,
                                            const int_t *full_dims,
                                            const int_t *subset_dims,
                                            const int_t *offsets,
                                            complex_t *vectors)
    int_t ckrylov_set_complex_block_products(int_t num_spaces, int_t total_size,
                                            const int_t *full_dims,
                                            const int_t *subset_dims,
                                            const int_t *offsets,
                                            complex_t *products)
    int_t ckrylov_get_complex_block_convergence()
    int_t ckrylov_solve_real_equation(int_t index, real_multiply_t multiply_fn)
    int_t ckrylov_solve_real_block_equation(real_block_multiply_t multiply_fn)
    int_t ckrylov_solve_complex_equation(int_t index, complex_multiply_t multiply_fn)
    int_t ckrylov_solve_complex_block_equation(complex_block_multiply_t multiply_fn)

    # Enum Strings
    char CKRYLOV_REAL_KIND
    char CKRYLOV_COMPLEX_KIND
    char CKRYLOV_SYMMETRIC_STRUCTURE
    char CKRYLOV_HERMITIAN_STRUCTURE
    char CKRYLOV_EIGENVALUE_EQUATION
    char CKRYLOV_LINEAR_EQUATION
    char CKRYLOV_SYLVESTER_EQUATIION
    char CKRYLOV_SHIFTED_LINEAR_EQUATION

    # Return Values
    int CKRYLOV_OK
    int CKRYLOV_NO_OPTIONS
    int CKRYLOV_NO_SUCH_OPTION
    int CKRYLOV_INVALID_OPTION
    int CKRYLOV_INVALID_KIND
    int CKRYLOV_INVALID_STRUCTURE
    int CKRYLOV_INVALID_EQUATION
    int CKRYLOV_INVALID_DIMENSION
    int CKRYLOV_INVALID_INPUT
    int CKRYLOV_NO_SPACES
    int CKRYLOV_NO_SUCH_SPACE
    int CKRYLOV_INCOMPATIBLE_SPACE
    int CKRYLOV_NOTHING_TO_DO
    int CKRYLOV_NO_ITERATIONS
    int CKRYLOV_NO_SUCH_ITERATION
    int CKRYLOV_NO_SOLUTIONS
    int CKRYLOV_NO_SUCH_SOLUTION
    int CKRYLOV_INVALID_CONFIGURATION
    int CKRYLOV_INCOMPLETE_CONFIGURATION
    int CKRYLOV_INCOMPATIBLE_CONFIGURATION
    int CKRYLOV_INCOMPATIBLE_EQUATION
    int CKRYLOV_INCOMPLETE_EQUATION
    int CKRYLOV_INCOMPATIBLE_PRECONDITIONER
    int CKRYLOV_INCOMPLETE_PRECONDITIONER
    int CKRYLOV_INCOMPATIBLE_ORTHONORMALIZER
    int CKRYLOV_LINEAR_ALGEBRA_ERROR
    int CKRYLOV_NOT_CONVERGED
    int CKRYLOV_MAX_ITERATIONS_REACHED
    int CKRYLOV_MAX_DIM_REACHED
    int CKRYLOV_ILL_CONDITIONED
    int CKRYLOV_NO_BASIS_UPDATE
    int CKRYLOV_LINEARLY_DEPENDENT_BASIS
    int CKRYLOV_KEY_NOT_FOUND