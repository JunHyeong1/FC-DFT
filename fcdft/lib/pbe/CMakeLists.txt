add_library(pbe SHARED boundary_condition.c poisson.c)

find_package(GSL REQUIRED)
find_package(LAPACK REQUIRED)

set_target_properties(pbe PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(pbe ${OPENMP_C_PROPERTIES} ${GSL_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(pbe PUBLIC ${GSL_INCLUDE_DIR})