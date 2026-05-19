#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "krylov" for configuration ""
set_property(TARGET krylov APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(krylov PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "Fortran"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libkrylov.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS krylov )
list(APPEND _IMPORT_CHECK_FILES_FOR_krylov "${_IMPORT_PREFIX}/lib/libkrylov.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
