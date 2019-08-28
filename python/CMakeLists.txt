
# cloned dependency: pybind11
# add path and search for python installed
add_subdirectory(pybind11)

# --------------------------------------
# wrap sophus

if(MINISAM_USE_SOPHUS)
  # add all unit test files
  file(GLOB SOPHUS_WRAPPER_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "sophus_wrapper/*.cpp")

  # wrapper lib
  set(SOPHUS_WRAPPER_TARGET_NAME "_minisam_sophus_py_wrapper")

  # add wrapper source
  add_library(${SOPHUS_WRAPPER_TARGET_NAME} MODULE ${SOPHUS_WRAPPER_SRCS})

  target_link_libraries(${SOPHUS_WRAPPER_TARGET_NAME} pybind11::module)

  # module naming and location
  set_target_properties(${SOPHUS_WRAPPER_TARGET_NAME} PROPERTIES 
      PREFIX "${PYTHON_MODULE_PREFIX}"
      SUFFIX "${PYTHON_MODULE_EXTENSION}" 
      OUTPUT_NAME ${SOPHUS_WRAPPER_TARGET_NAME})

  # for export setup.py
  set(SOPHUS_WRAPPER_FILE_NAME "${PYTHON_MODULE_PREFIX}${SOPHUS_WRAPPER_TARGET_NAME}${PYTHON_MODULE_EXTENSION}")
endif()

# --------------------------------------
# wrap minisam

# list of blocked unit tests
set(EXCLUDED_WRAPPER_SRCS " ")


# add all unit test files
file(GLOB MINISAM_WRAPPER_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "minisam_wrapper/*.cpp")

# remove unused module
if(NOT MINISAM_USE_SOPHUS)
  list(APPEND EXCLUDED_WRAPPER_SRCS "minisam_wrapper/geometry.cpp")
  list(APPEND EXCLUDED_WRAPPER_SRCS "minisam_wrapper/slam.cpp")
endif()

# remove blocked tests
list(REMOVE_ITEM MINISAM_WRAPPER_SRCS ${EXCLUDED_WRAPPER_SRCS})


# wrapper lib
set(MINISAM_WRAPPER_TARGET_NAME "_minisam_py_wrapper")

# add wrapper source
add_library(${MINISAM_WRAPPER_TARGET_NAME} MODULE ${MINISAM_WRAPPER_SRCS})

target_link_libraries(${MINISAM_WRAPPER_TARGET_NAME} ${PROJECT_NAME} 
    ${MINISAM_ADDITIONAL_LIBRARIES} pybind11::module)

# module naming and location
set_target_properties(${MINISAM_WRAPPER_TARGET_NAME} PROPERTIES 
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}" 
    OUTPUT_NAME ${MINISAM_WRAPPER_TARGET_NAME})

# for export setup.py
set(MINISAM_WRAPPER_FILE_NAME "${PYTHON_MODULE_PREFIX}${MINISAM_WRAPPER_TARGET_NAME}${PYTHON_MODULE_EXTENSION}")


# --------------------------------------
# install wrapper and python source

# custom target
add_custom_target(python_package ALL)
# depends on wrappers
add_dependencies(python_package ${MINISAM_WRAPPER_TARGET_NAME} ${SOPHUS_WRAPPER_TARGET_NAME})

# copy built wrapper to package by custom target
# TODO: this is to source location and have to be ignored by gitignore, better solution??
add_custom_command(TARGET python_package POST_BUILD COMMAND ${CMAKE_COMMAND} -E 
    copy $<TARGET_FILE:${MINISAM_WRAPPER_TARGET_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}/minisam")

if(MINISAM_USE_SOPHUS)
  add_custom_command(TARGET python_package POST_BUILD COMMAND ${CMAKE_COMMAND} -E 
      copy $<TARGET_FILE:${SOPHUS_WRAPPER_TARGET_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}/minisam/sophus")
endif()

# copy project lib when shared
if(MINISAM_BUILD_SHARED_LIB)
  add_custom_command(TARGET python_package POST_BUILD COMMAND ${CMAKE_COMMAND} -E 
      copy $<TARGET_FILE:${PROJECT_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}/minisam")
endif()

# path to export setup.py
set(MINISAM_WRAPPER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/minisam")
set(SOPHUS_WRAPPER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/minisam/sophus")

# if windows need to replace slash with backslash
if(WIN32)
  string(REGEX REPLACE "/" "\\\\\\\\" MINISAM_WRAPPER_PATH ${MINISAM_WRAPPER_PATH})
  string(REGEX REPLACE "/" "\\\\\\\\" SOPHUS_WRAPPER_PATH ${SOPHUS_WRAPPER_PATH})
endif()

# generate setup.py for distutils
if(MINISAM_USE_SOPHUS)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in" "${PROJECT_BINARY_DIR}/setup.py")
else()
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup_no_sophus.py.in" "${PROJECT_BINARY_DIR}/setup.py")
endif()

# install by python `python setup.py install`
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py install)")

