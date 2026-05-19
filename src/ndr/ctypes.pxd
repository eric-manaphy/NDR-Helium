# These are just the default types
# TODO: Make it not hardcoded
cdef extern from "include/ctypes.h":
    ctypedef long long int_t
    ctypedef double real_t
    ctypedef char char_t
    ctypedef bint bool_t

    # Complex external typedefs aren't supported
    # So we'll just have to cast it per function call
    # ...or just use a double and call it a day
    ctypedef struct complex_t:
        pass