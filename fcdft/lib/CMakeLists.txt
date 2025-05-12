cmake_minimum_required (VERSION 3.5)
project (fcdft)

include(CheckSymbolExists)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
set(CMAKE_C_STANDARD 99)

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()
else()
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse3" COMPILER_SUPPORTS_SSE3)
    if(COMPILER_SUPPORTS_SSE3)
      # Avoids error "‘SIMDD’ undeclared here (not in a function)" in the qcint two-electron integral interface
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    endif()
  endif()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# if (NOT BLAS_LIBRARIES)
# find_package(BLAS)
# endif()
# check_symbol_exists(ffsll "strings.h" HAVE_FFS)

# if (NOT BLAS_LIBRARIES)
#   message(FATAL_ERROR "A required library with BLAS API not found.")
# else()
#   message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
# endif()

option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#
#if(NOT Python3_FOUND)
#    message(FATAL_ERROR "Failed to locate Python3.")
#endif()
#message(STATUS "Python3 interpreter: ${Python3_EXECUTABLE}")
## locate pyscf
#execute_process(
#    COMMAND ${Python3_EXECUTABLE} -c "import pyscf; print(pyscf.__path__[0])"
#    OUTPUT_VARIABLE PYSCF_PATH
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#    RESULT_VARIABLE PYSCF_FOUND
#)
#
#if(NOT PYSCF_FOUND EQUAL 0)
#    message(FATAL_ERROR "Unable to locate pyscf.")
#else()
#    message(STATUS "pyscf path: ${PYSCF_PATH}")
#endif()
#
#set(PYSCF_LIB_PATH "${PYSCF_PATH}/lib")
#message(STATUS "pyscf lib path: ${PYSCF_LIB_PATH}")
#
#include_directories(${PYSCF_LIB_PATH}/vhf)
#include_directories(${PYSCF_LIB_PATH})
#include_directories(${PYSCF_LIB_PATH}/deps/include)
#include_directories(${CMAKE_INSTALL_PREFIX}/include)
#link_directories(${PYSCF_LIB_PATH})
#link_directories(${PYSCF_LIB_PATH}/deps/lib ${PYSCF_LIB_PATH}/deps/lib64)
#link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

#add_subdirectory(vhf)
add_subdirectory(dft)
add_subdirectory(pbe)
#configure_file(
#  "${PYSCF_LIB_PATH}/config.h.in"
#  "${PYSCF_LIB_PATH}/config.h")
# to find config.h
include_directories("${PROJECT_BINARY_DIR}")

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
