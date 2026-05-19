#ifndef CKRYLOV_H_
#define CKRYLOV_H_

#include "ctypes.h"

/* Multiplication function interfaces */
int_t real_multiply(const int_t *full_dim, const int_t *subset_dim,
                    const real_t *vectors, real_t *products);
int_t complex_multiply(const int_t *full_dim, const int_t *subset_tim,
                       const complex_t *vectors, complex_t *products);
int_t real_block_multiply(const int_t *num_spaces, const int_t *total_size,
                          const int_t *full_dims, const int_t *subset_dims,
                          const int_t *offsets, const real_t *vectors,
                          real_t *products);
int_t complex_block_multiply(const int_t *num_spaces, const int_t *total_size,
                             const int_t *full_dims, const int_t *subset_dims,
                             const int_t *offsets, const complex_t *vectors,
                             complex_t *products);

/* C-stype interface functions */
int_t ckrylov_initialize();
int_t ckrylov_finalize();
int_t ckrylov_get_version(char_t *version, int_t version_len);
int_t ckrylov_get_compiled_datetime(char_t *compiled_dt, int_t compiled_dt_len);
int_t ckrylov_get_current_datetime(char_t *current_dt, int_t current_dt_len);
int_t ckrylov_header();
int_t ckrylov_footer();
int_t ckrylov_set_defaults();
int_t ckrylov_get_integer_option(const char_t *key, int_t key_len);
int_t ckrylov_set_integer_option(const char_t *key, int_t key_len, int_t value);
int_t ckrylov_get_string_option(const char_t *key, int_t key_len, char_t *value,
                                int_t value_len);
int_t ckrylov_set_string_option(const char_t *key, int_t key_len,
                                const char_t *value, int_t value_len);
int_t ckrylov_length_string_option(const char_t *key, int_t key_len);
real_t ckrylov_get_real_option(const char_t *key, int_t key_len);
int_t ckrylov_set_real_option(const char_t *key, int_t key_len, real_t value);
bool_t ckrylov_get_logical_option(const char_t *key, int_t key_len);
int_t ckrylov_set_logical_option(const char_t *key, int_t key_len,
                                 bool_t value);
int_t ckrylov_get_enum_option(const char_t *key, int_t key_len,
                              const char_t *value, int_t value_len);
int_t ckrylov_set_enum_option(const char_t *key, int_t key_len,
                              const char_t *value, int_t value_len);
int_t ckrylov_length_enum_option(const char_t *key, int_t key_len);
int_t ckrylov_define_enum_option(const char_t *key, int_t key_len,
                                 const char_t *values, int_t values_len);
int_t ckrylov_validate_enum_option(const char_t *key, int_t key_len,
                                   const char_t *value, int_t value_len);
int_t ckrylov_add_space(const char_t *kind, int_t kind_len,
                        const char_t *structure, int_t structure_len,
                        const char_t *equation, int_t equation_len,
                        int_t full_dim, int_t solution_dim, int_t basis_dim);
int_t ckrylov_get_num_spaces();
int_t ckrylov_get_space_integer_option(int_t index, const char_t *key,
                                       int_t key_len);
int_t ckrylov_set_space_integer_option(int_t index, const char_t *key,
                                       int_t key_len, int_t value);
int_t ckrylov_get_space_string_option(int_t index, const char_t *key,
                                      int_t key_len, char_t *value,
                                      int_t value_len);
int_t ckrylov_set_space_string_option(int_t index, const char_t *key,
                                      int_t key_len, const char_t *value,
                                      int_t value_len);
int_t ckrylov_length_space_string_option(int_t index, const char_t *key,
                                         int_t key_len);
real_t ckrylov_get_space_real_option(int_t index, const char_t *key,
                                     int_t key_len);
int_t ckrylov_set_space_real_option(int_t index, const char_t *key,
                                    int_t key_len, real_t value);
bool_t ckrylov_get_space_logical_option(int_t index, const char_t *key,
                                        int_t key_len);
int_t ckrylov_set_space_logical_option(int_t index, const char_t *key,
                                       int_t key_len, bool_t value);
int_t ckrylov_get_space_enum_option(int_t index, const char_t *key,
                                    int_t key_len, char_t *value,
                                    int_t value_len);
int_t ckrylov_set_space_enum_option(int_t index, const char_t *key,
                                    int_t key_len, const char_t *value,
                                    int_t value_len);
int_t ckrylov_length_space_enum_option(int_t index, const char_t *key,
                                       int_t key_len);
int_t ckrylov_define_space_enum_option(int_t index, const char_t *key,
                                       int_t key_len, const char_t *values,
                                       int_t values_len);
int_t ckrylov_validate_space_enum_option(int_t index, const char_t *key,
                                         int_t key_len, const char_t *value,
                                         int_t value_len);
int_t ckrylov_get_space_kind(int_t index, char_t *kind, int_t kind_len);
int_t ckrylov_get_space_structure(int_t index, char_t *structure,
                                  int_t structure_len);
int_t ckrylov_get_space_equation(int_t index, char_t *equation,
                                 int_t equation_len);
int_t ckrylov_get_space_preconditioner(int_t index, char_t *preconditioner,
                                       int_t preconditioner_len);
int_t ckrylov_set_space_preconditioner(int_t index,
                                       const char_t *preconditioner,
                                       int_t preconditioner_len);
int_t ckrylov_get_space_orthonormalizer(int_t index, char_t *orthonormalizer,
                                        int_t orthonormalizer_len);
int_t ckrylov_set_space_orthonormalizer(int_t index,
                                        const char_t *orthonormalizer,
                                        int_t orthonormalizer_len);
int_t ckrylov_get_space_full_dim(int_t index);
int_t ckrylov_get_space_solution_dim(int_t index);
int_t ckrylov_get_space_basis_dim(int_t index);
int_t ckrylov_get_space_vector_size(int_t index);
int_t ckrylov_get_space_solution_size(int_t index);
int_t ckrylov_get_space_eigenvalue_size(int_t index);
int_t ckrylov_get_space_rhs_size(int_t index);
int_t ckrylov_get_space_sylvester_b_size(int_t index);
int_t ckrylov_get_space_num_iterations(int_t index);
int_t ckrylov_get_space_iteration_basis_dim(int_t index, int_t iter);
real_t ckrylov_get_space_iteration_gram_rcond(int_t index, int_t iter);
int_t ckrylov_get_space_iteration_expectation_vals(int_t index, int_t iter,
                                                   int_t solution_dim,
                                                   real_t *expectation_vals);
real_t ckrylov_get_space_iteration_lagrangian(int_t index, int_t iter);
int_t ckrylov_get_space_iteration_residual_norms(int_t index, int_t iter,
                                                 int_t solution_dim,
                                                 real_t *residual_norms);
real_t ckrylov_get_space_iteration_max_residual_norms(int_t index, int_t iter);
int_t ckrylov_get_space_last_basis_dim(int_t index);
int_t ckrylov_get_space_last_expectation_vals(int_t index, int_t solution_dim,
                                              real_t *expectation_vals);
real_t ckrylov_get_space_last_lagrangian(int_t index);
int_t ckrylov_get_space_last_residual_norms(int_t index, int_t solution_dim,
                                            real_t *residual_norms);
real_t ckrylov_get_space_last_max_residual_norm(int_t index);
real_t ckrylov_get_space_last_gram_rcond(int_t index);
int_t ckrylov_get_space_convergence(int_t index);
int_t ckrylov_get_space_eigenvalues(int_t index, int_t solution_dim,
                                    real_t *eigenvalues);
int_t ckrylov_set_space_diagonal(int_t index, int_t full_dim,
                                 const real_t *diagonal);
int_t ckrylov_get_space_shifts(int_t index, int_t solution_dim, real_t *shifts);
int_t ckrylov_set_space_shifts(int_t index, int_t solution_dim,
                               const real_t *shifts);
int_t ckrylov_get_real_space_vectors(int_t index, int_t full_dim,
                                     int_t basis_dim, real_t *vectors);
int_t ckrylov_set_real_space_vectors(int_t index, int_t full_dim,
                                     int_t basis_dim, const real_t *vectors);
int_t ckrylov_set_real_space_vectors_from_diagonal(int_t index, int_t full_dim,
                                                   int_t basis_dim,
                                                   const real_t *diagonal);
int_t ckrylov_resize_real_space_vectors(int_t index, int_t basis_dim);
int_t ckrylov_get_real_space_products(int_t index, int_t full_dim,
                                      int_t basis_dim, real_t *products);
int_t ckrylov_get_real_space_solutions(int_t index, int_t full_dim,
                                       int_t solution_dim,
                                       const real_t *solutions);
int_t ckrylov_resize_real_space_solutions(int_t index, int_t solution_dim);
int_t ckrylov_get_real_space_rhs(int_t index, int_t full_dim,
                                 int_t solution_dim, real_t *rhs);
int_t ckrylov_set_real_space_rhs(int_t index, int_t full_dim,
                                 int_t solution_dim, const real_t *rhs);
int_t ckrylov_get_real_space_sylvester_b(int_t index, int_t solution_dim,
                                         real_t *sylvester_b);
int_t ckrylov_set_real_space_sylvester_b(int_t index, int_t solution_dim,
                                         const real_t *sylvester_b);
int_t ckrylov_get_real_space_subset_vectors(int_t index, int_t full_dim,
                                            int_t skip_dim, int_t subset_dim,
                                            real_t *vectors);
int_t ckrylov_set_real_space_subset_vectors(int_t index, int_t full_dim,
                                            int_t skip_dim, int_t subset_dim,
                                            const real_t *vectors);
int_t ckrylov_get_real_space_subset_products(int_t index, int_t full_dim,
                                             int_t skip_dim, int_t subset_dim,
                                             real_t *products);
int_t ckrylov_set_real_space_subset_products(int_t index, int_t full_dim,
                                             int_t skip_dim, int_t subset_dim,
                                             const real_t *products);
int_t ckrylov_get_complex_space_vectors(int_t index, int_t full_dim,
                                        int_t basis_dim, complex_t *vectors);
int_t ckrylov_set_complex_space_vectors(int_t index, int_t full_dim,
                                        int_t basis_dim,
                                        const complex_t *vectors);
int_t ckrylov_set_complex_space_vectors_from_diagonal(int_t index,
                                                      int_t full_dim,
                                                      int_t basis_dim,
                                                      const real_t *diagonal);
int_t ckrylov_resize_complex_space_vectors(int_t index, int_t basis_dim);
int_t ckrylov_get_complex_space_products(int_t index, int_t full_dim,
                                         int_t basis_dim, complex_t *products);
int_t ckrylov_get_complex_space_solutions(int_t index, int_t full_dim,
                                          int_t solution_dim,
                                          complex_t *solutions);
int_t ckrylov_resize_complex_space_solutions(int_t index, int_t solution_dim);
int_t ckrylov_get_complex_space_rhs(int_t index, int_t full_dim,
                                    int_t solution_dim, complex_t *rhs);
int_t ckrylov_set_complex_space_rhs(int_t index, int_t full_dim,
                                    int_t solution_dim, const complex_t *rhs);
int_t ckrylov_get_complex_space_sylvester_b(int_t index, int_t solution_dim,
                                            complex_t *sylvester_b);
int_t ckrylov_set_complex_space_sylvester_b(int_t index, int_t solution_dim,
                                            const complex_t *sylvester_b);
int_t ckrylov_get_complex_space_subset_vectors(int_t index, int_t full_dim,
                                               int_t skip_dim, int_t subset_dim,
                                               complex_t *vectors);
int_t ckrylov_set_complex_space_subset_vectors(int_t index, int_t full_dim,
                                               int_t skip_dim, int_t subset_dim,
                                               const complex_t *vectors);
int_t ckrylov_get_complex_space_subset_products(int_t index, int_t full_dim,
                                                int_t skip_dim,
                                                int_t subset_dim,
                                                complex_t *products);
int_t ckrylov_set_complex_space_subset_products(int_t index, int_t full_dim,
                                                int_t skip_dim,
                                                int_t subset_dim,
                                                const complex_t *products);
int_t ckrylov_get_real_block_total_size();
int_t ckrylov_get_real_block_dims(int_t num_spaces, int_t *full_dims,
                                  int_t *subset_dims, int_t *offsets);
int_t ckrylov_get_real_block_vectors(int_t num_spaces, int_t total_size,
                                     const int_t *full_dims,
                                     const int_t *subset_dims,
                                     const int_t *offsets, real_t *vectors);
int_t ckrylov_set_real_block_products(int_t num_spaces, int_t total_size,
                                      const int_t *full_dims,
                                      const int_t *subset_dims,
                                      const int_t *offsets,
                                      const real_t *products);
int_t ckrylov_get_real_block_convergence();
int_t ckrylov_get_complex_block_total_size();
int_t ckrylov_get_complex_block_dims(int_t num_spaces, int_t *full_dims,
                                     int_t *subset_dims, int_t *offsets);
int_t ckrylov_get_complex_block_vectors(int_t num_spaces, int_t total_size,
                                        const int_t *full_dims,
                                        const int_t *subset_dims,
                                        const int_t *offsets,
                                        complex_t *vectors);
int_t ckrylov_set_complex_block_products(int_t num_spaces, int_t total_size,
                                         const int_t *full_dims,
                                         const int_t *subset_dims,
                                         const int_t *offsets,
                                         complex_t *products);
int_t ckrylov_get_complex_block_convergence();
int_t ckrylov_solve_real_equation(
    int_t index,
    int_t (*real_multiply)(const int_t *full_dim, const int_t *subset_dim,
                           const real_t *vectors, real_t *products));
int_t ckrylov_solve_real_block_equation(int_t (*real_block_multiply)(
    const int_t *num_spaces, const int_t *total_size, const int_t *full_dims,
    const int_t *subset_dims, const int_t *offsets, const real_t *vectors,
    real_t *products));
int_t ckrylov_solve_complex_equation(
    int_t index,
    int_t (*complex_multiply)(const int_t *full_dim, const int_t *subset_tim,
                              const complex_t *vectors, complex_t *products));
int_t ckrylov_solve_complex_block_equation(int_t (*complex_block_multiply)(
    const int_t *num_spaces, const int_t *total_size, const int_t *full_dims,
    const int_t *subset_dims, const int_t *offsets, const complex_t *vectors,
    complex_t *products));

/* Enum strings */
#define CKRYLOV_REAL_KIND "r"
#define CKRYLOV_COMPLEX_KIND "c"
#define CKRYLOV_SYMMETRIC_STRUCTURE "s"
#define CKRYLOV_HERMITIAN_STRUCTURE "h"
#define CKRYLOV_EIGENVALUE_EQUATION "e"
#define CKRYLOV_LINEAR_EQUATION "l"
#define CKRYLOV_SYLVESTER_EQUATIION "s"
#define CKRYLOV_SHIFTED_LINEAR_EQUATION "h"

/* Return values */
#define CKRYLOV_OK 0
#define CKRYLOV_NO_OPTIONS -101
#define CKRYLOV_NO_SUCH_OPTION -102
#define CKRYLOV_INVALID_OPTION -103
#define CKRYLOV_INVALID_KIND -104
#define CKRYLOV_INVALID_STRUCTURE -105
#define CKRYLOV_INVALID_EQUATION -106
#define CKRYLOV_INVALID_DIMENSION -107
#define CKRYLOV_INVALID_INPUT -108
#define CKRYLOV_NO_SPACES -201
#define CKRYLOV_NO_SUCH_SPACE -202
#define CKRYLOV_INCOMPATIBLE_SPACE -203
#define CKRYLOV_NOTHING_TO_DO -204
#define CKRYLOV_NO_ITERATIONS -205
#define CKRYLOV_NO_SUCH_ITERATION -206
#define CKRYLOV_NO_SOLUTIONS -207
#define CKRYLOV_NO_SUCH_SOLUTION -208
#define CKRYLOV_INVALID_CONFIGURATION -209
#define CKRYLOV_INCOMPLETE_CONFIGURATION -210
#define CKRYLOV_INCOMPATIBLE_CONFIGURATION -211
#define CKRYLOV_INCOMPATIBLE_EQUATION -212
#define CKRYLOV_INCOMPLETE_EQUATION -213
#define CKRYLOV_INCOMPATIBLE_PRECONDITIONER -214
#define CKRYLOV_INCOMPLETE_PRECONDITIONER -215
#define CKRYLOV_INCOMPATIBLE_ORTHONORMALIZER -216
#define CKRYLOV_LINEAR_ALGEBRA_ERROR -301
#define CKRYLOV_NOT_CONVERGED -401
#define CKRYLOV_MAX_ITERATIONS_REACHED -402
#define CKRYLOV_MAX_DIM_REACHED -403
#define CKRYLOV_ILL_CONDITIONED -404
#define CKRYLOV_NO_BASIS_UPDATE -405
#define CKRYLOV_LINEARLY_DEPENDENT_BASIS -406
#define CKRYLOV_KEY_NOT_FOUND -501

#endif /* CKRYLOV_H_ */
