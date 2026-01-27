add_library(df SHARED nr_contract_k.c)

find_package(LAPACK REQUIRED)

set_target_properties(df PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(df ${OPENMP_C_PROPERTIES} ${LAPACK_LIBRARIES})