add_library(fcdft SHARED nr_numint.c quadrature.c)

set_target_properties(fcdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(fcdft ${OPENMP_C_PROPERTIES})