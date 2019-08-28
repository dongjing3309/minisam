# add all unit tests

# excluded from make all
# not excluded for VC++, to let check shown in VC++
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  set_directory_properties(${CMAKE_CURRENT_SOURCE_DIR} EXCLUDE_FROM_ALL 1)
endif()

# list of blocked unit tests
set(EXCLUDED_UNIT_TESTS " ")

# add catch test main
# only build this once to save time, since catch main is slow to compile
set(TEST_TARGET_NAME "catch_test_main")
add_library(${TEST_TARGET_NAME} OBJECT "catch_test_main.cpp")

# add all unit test files
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "test*.cpp")

# get test names by remove extension
foreach(TESTFILE ${TEST_SRCS})
  get_filename_component(TEST ${TESTFILE} NAME_WE)
  list(APPEND TEST_LIST ${TEST})
endforeach()

# remove unused test

# sophus
if(NOT MINISAM_USE_SOPHUS)
  list(APPEND EXCLUDED_UNIT_TESTS "testSophus")
  list(APPEND EXCLUDED_UNIT_TESTS "testCalibration")
  list(APPEND EXCLUDED_UNIT_TESTS "testProjection")
endif()

# optional solvers
if(NOT MINISAM_USE_CUSOLVER)
  list(APPEND EXCLUDED_UNIT_TESTS "testCUDASolver")
endif()
if(NOT MINISAM_USE_SPQR)
  list(APPEND EXCLUDED_UNIT_TESTS "testSPQR")
endif()
if(NOT MINISAM_USE_CHOLMOD)
  list(APPEND EXCLUDED_UNIT_TESTS "testCholmod")
endif()

message(STATUS "Add all unit tests, exclude: ${EXCLUDED_UNIT_TESTS}")

# remove blocked tests
list(REMOVE_ITEM TEST_LIST ${EXCLUDED_UNIT_TESTS})


# add all unit test targets
foreach(TEST ${TEST_LIST})
  file(GLOB TEST_SRC "${TEST}.cpp" "*.h")
  add_executable(${TEST} ${TEST_SRC} $<TARGET_OBJECTS:${TEST_TARGET_NAME}>)

  if(WIN32)
    set_target_properties(${TEST} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  endif()

  target_link_libraries(${TEST} ${PROJECT_NAME})
  add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()

# add all tests to target 'check'
# due to CMake dependency bug, cannot use target 'test' here
# call 'make test' before 'make' or 'make all' will give dependency error
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_BUILD_TYPE} 
    DEPENDS ${TEST_LIST})
