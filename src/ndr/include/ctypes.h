#ifndef CTYPES_H_
#define CTYPES_H_

#include <float.h>

/* Variable types */
#ifdef USE_C_INT
#define int_t int
#elif defined USE_C_LONG
#define int_t long
#elif defined USE_C_LONG_LONG
#define int_t long long
#elif defined USE_C_SIZE_T
#define int_t c_size_t
#else
#define int_t long
#endif

#ifdef USE_C_FLOAT
#define real_t float
#define REAL_HUGE FLT_MAX
#define REAL_EPSILON FLT_EPSILON
#elif defined USE_C_DOUBLE
#define real_t double
#define REAL_HUGE DBL_MAX
#define REAL_EPSILON DBL_EPSILON
#elif defined USE_C_LONG_DOUBLE
#define real_t long double
#define REAL_HUGE LDBL_MAX
#define REAL_EPSILSON LDBL_EPSILON
#else
#define real_t double
#define REAL_HUGE DBL_MAX
#define REAL_EPSILON DBL_EPSILON
#endif

#ifdef USE_COMPLEX_C_FLOAT
#define complex_t float _Complex
#elif defined USE_COMPLEX_C_DOUBLE
#define complex_t double _Complex
#elif defined USE_COMPLEX_C_LONG_DOUBLE
#define complex_t long double _Complex
#else
#define complex_t double _Complex
#endif

#define char_t char
#define bool_t _Bool

#endif /* CTYPES_H_ */
