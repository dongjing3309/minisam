# project minisam
project(minisam LANGUAGES CXX)
cmake_minimum_required(VERSION 3.4)
enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# for configure_package_config_file
include(CMakePackageConfigHelpers)


# --------------------------------------
# options

# default build type release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# python python wrapper
option(MINISAM_BUILD_PYTHON_PACKAGE "Build wrapper for Python package" OFF)

# shared or static lib
option(MINISAM_BUILD_SHARED_LIB "Build shared lib if ON, static lib if OFF" ON)

# enable multi-threading
option(MINISAM_WITH_MULTI_THREADS "Enable multi-threading if ON, otherwise OFF" OFF)

# number of threads to use if enable multi-threading
set(MINISAM_WITH_MULTI_THREADS_NUM "4" CACHE STRING "Number of threads to use if enable multi-threading")

# enable internal timing
option(MINISAM_WITH_INTERNAL_TIMING "Enable internal profiling if ON, otherwise OFF" OFF)

# enable optional dependencies
# use Sophus
option(MINISAM_WITH_SOPHUS "Enable/Disable Sophus" ON)

# use CUDA cuSOLVER
option(MINISAM_WITH_CUSOLVER "Enable/Disable CUDA cuSOLVER" OFF)

# use SuiteSparse CHOLMOD solver
option(MINISAM_WITH_CHOLMOD "Enable/Disable SuiteSparse CHOLMOD solver" ON)

# use SuiteSparse SPQR solver
option(MINISAM_WITH_SPQR "Enable/Disable SuiteSparse SPQR solver" ON)


# --------------------------------------
# configuration conflict

# python wrapper cannot be compiled with static lib, since static lib is compiled without -fPIC
if(MINISAM_BUILD_PYTHON_PACKAGE AND NOT MINISAM_BUILD_SHARED_LIB)
  message(FATAL_ERROR "Python wrapper cannot be compiled with static lib" )
endif()

# --------------------------------------
# default compiler flags

# GNU gcc
if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
  
# Clang
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

# Visual stdio
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  # use math macro
  add_definitions("-D _USE_MATH_DEFINES")
endif()

# export for DLL
if(WIN32 AND MINISAM_BUILD_SHARED_LIB)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif()

# --------------------------------------
# dependencies

# libraries linked to
set(MINISAM_ADDITIONAL_LIBRARIES)

# mandatory dependency: Eigen3
message(STATUS "Search Dependency: Eigen3")
find_package(Eigen3 REQUIRED 3.3.0)
include_directories(AFTER ${EIGEN3_INCLUDE_DIR})


# --------------------------------------
# optional dependency: Sophus

if(MINISAM_WITH_SOPHUS)
  find_package(Sophus)
  message(STATUS "Search Dependency: Sophus")

  if(Sophus_FOUND)
    set(MINISAM_USE_SOPHUS 1)
    get_target_property(Sophus_INCLUDE_DIR Sophus::Sophus INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(AFTER ${Sophus_INCLUDE_DIR})

  else()
    message(WARNING "Using Sophus is set, but Sophus is NOT found. Sophus will not be used")
    set(MINISAM_USE_SOPHUS 0)

  endif()
else()
  set(MINISAM_USE_SOPHUS 0)
endif()


# --------------------------------------
# optional dependency: SuiteSparse CHOLMOD solver

if(MINISAM_WITH_CHOLMOD)
  message(STATUS "Search Dependency: Cholmod")
  find_package(Cholmod)

  if(CHOLMOD_FOUND)
    set(MINISAM_USE_CHOLMOD 1)  # This will go into config.h
    include_directories(AFTER ${CHOLMOD_INCLUDES})
    list(APPEND MINISAM_ADDITIONAL_LIBRARIES ${CHOLMOD_LIBRARIES})

  else()
    message(WARNING "Using Cholmod is set, but Cholmod is NOT found. Cholmod will not be used")
    set(MINISAM_USE_CHOLMOD 0)  # This will go into config.h

  endif()
else()
  set(MINISAM_USE_CHOLMOD 0)  # This will go into config.h
endif()


# --------------------------------------
# optional dependency: SuiteSparse SPQR solver

if(MINISAM_WITH_SPQR)
  message(STATUS "Search Dependency: SPQR")
  find_package(SPQR)

  if(SPQR_FOUND)
    set(MINISAM_USE_SPQR 1)  # This will go into config.h
    include_directories(AFTER ${SPQR_INCLUDES})
    list(APPEND MINISAM_ADDITIONAL_LIBRARIES ${SPQR_LIBRARIES})
  else()

    message(WARNING "Using SPQR is set, but SPQR is NOT found. SPQR will not be used")
    set(MINISAM_USE_SPQR 0)  # This will go into config.h

  endif()
else()
  set(MINISAM_USE_SPQR 0)  # This will go into config.h
endif()


# --------------------------------------
# optional dependency: cusolver

if(MINISAM_WITH_CUSOLVER)
  # optional dependency: CUDA
  message(STATUS "Search Dependency: CUDA")
  find_package(CUDA)

  if(CUDA_FOUND)
    # optional dependency: cusolver in CUDA >= 8.0 with GCC requires OpenMP
    if((NOT (${CUDA_VERSION_MAJOR} LESS 8)) AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
      message(STATUS "Search Dependency: OpenMP")
      find_package(OpenMP)

      if(OpenMP_FOUND)
        message(STATUS "CUDA and OpenMP found (with GCC): use cuSOLVER")
        set(MINISAM_USE_CUSOLVER 1)   # This will go into config.h
        # CUDA
        include_directories(AFTER ${CUDA_INCLUDE_DIRS})
        list(APPEND MINISAM_ADDITIONAL_LIBRARIES
            ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
        # OpenMP compiler flags
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

      else()
        message(WARNING "Using cuSOLVER is set, but OpenMP is NOT found. cuSOLVER will not be used (with GCC)")
        set(MINISAM_USE_CUSOLVER 0)  # This will go into config.h
      endif()

    else()
      message(STATUS "CUDA found: use cuSOLVER")
      set(MINISAM_USE_CUSOLVER 1)  # This will go into config.h
      # CUDA
      include_directories(AFTER ${CUDA_INCLUDE_DIRS})
      list(APPEND MINISAM_ADDITIONAL_LIBRARIES
          ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
    endif()

  else()
    message(WARNING "Using cuSOLVER is set, but CUDA is NOT found. cuSOLVER will not be used")
    set(MINISAM_USE_CUSOLVER 0)  # This will go into config.h
  endif()
else()
  set(MINISAM_USE_CUSOLVER 0)  # This will go into config.h
endif()

# thread libs
if(MINISAM_WITH_MULTI_THREADS)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  list(APPEND MINISAM_ADDITIONAL_LIBRARIES Threads::Threads)
endif()


# --------------------------------------
# code

# source code list
set(MINISAM_SRCS)

# Generate and install config and dllexport files
configure_file("${PROJECT_NAME}/config.h.in" "${PROJECT_NAME}/config.h")
list(APPEND MINISAM_SRCS "${PROJECT_BINARY_DIR}/${PROJECT_NAME}/config.h")
include_directories(BEFORE ${PROJECT_BINARY_DIR}) # So we can include generated config header files
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}/config.h" DESTINATION "include/${PROJECT_NAME}")

# add current path as include dir
# search local before seach sys path
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# source code
add_subdirectory(minisam)

# unit tests
add_subdirectory(tests)

# C++ examples
add_subdirectory(examples/cpp)


# --------------------------------------
# python module

if(MINISAM_BUILD_PYTHON_PACKAGE)

  # clone git submodule
  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    message(STATUS "git submodule update: pybind11")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()

  # python wrapper compile and install
  add_subdirectory(python)
endif()


# --------------------------------------
# summary

message(STATUS "===============================================================")
message(STATUS "                Configurations for miniSAM                     ")

# lib
message(STATUS " ")
if(MINISAM_BUILD_SHARED_LIB)
  message(STATUS "  Library type                Shared")
else()
  message(STATUS "  Library type                Static")
endif()
message(STATUS "  Install path                ${CMAKE_INSTALL_PREFIX}")

# complier
message(STATUS " ")
message(STATUS "  Compiler type               ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Compiler version            ${CMAKE_CXX_COMPILER_VERSION}")

if(NOT MSVC AND NOT XCODE_VERSION)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" cmake_build_type_upper)
  message(STATUS "  Built type                  ${CMAKE_BUILD_TYPE}")
  message(STATUS "  C++ compilation flags       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${cmake_build_type_upper}}")
endif()

# multi-threading
message(STATUS " ")
if(MINISAM_WITH_MULTI_THREADS)
  message(STATUS "  Multi-threading             Yes (${MINISAM_WITH_MULTI_THREADS_NUM})")
else()
  message(STATUS "  Multi-threading             No")
endif()

# internal timing
if(MINISAM_WITH_INTERNAL_TIMING)
  message(STATUS "  Internal profiling          Yes")
else()
  message(STATUS "  Internal profiling          No")
endif()

# dependency
message(STATUS " ")
message(STATUS "  Eigen                       ${EIGEN3_VERSION} (${EIGEN3_INCLUDE_DIR})")

if(MINISAM_USE_SOPHUS)
  message(STATUS "  Sophus                      Yes (${Sophus_DIR})")
else()
  message(STATUS "  Sophus                      No")
endif()

if(MINISAM_USE_CHOLMOD)
  message(STATUS "  Cholmod                     Yes (${CHOLMOD_INCLUDES})")
else()
  message(STATUS "  Cholmod                     No")
endif()

if(MINISAM_USE_SPQR)
  message(STATUS "  SPQR                        Yes (${SPQR_INCLUDES})")
else()
  message(STATUS "  SPQR                        No")
endif()

if(MINISAM_USE_CUSOLVER)
  message(STATUS "  CUDA cuSOLVER               ${CUDA_VERSION} (${CUDA_INCLUDE_DIRS})")
else()
  message(STATUS "  CUDA cuSOLVER               No")
endif()

# python
message(STATUS " ")
if(MINISAM_BUILD_PYTHON_PACKAGE)
  message(STATUS "  Python package              Yes")
  message(STATUS "  Python version              ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  message(STATUS "  Python executable           ${PYTHON_EXECUTABLE}")
else()
  message(STATUS "  Python package              No")
endif()

message(STATUS " ")
