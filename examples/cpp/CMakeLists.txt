# add all C++ examples

# list of blocked unit tests
set(EXCLUDED_EXAMPLES " ")

# sophus
if(NOT MINISAM_WITH_SOPHUS)
  list(APPEND EXCLUDED_EXAMPLES "bundle_adjustment_bal")
  list(APPEND EXCLUDED_EXAMPLES "gps_factor_example")
  list(APPEND EXCLUDED_EXAMPLES "pose_graph_example")
  list(APPEND EXCLUDED_EXAMPLES "pose_graph_g2o")
endif()

message(STATUS "Add all C++ examples, exclude: ${EXCLUDED_EXAMPLES}")

# add all example files
file(GLOB EXAMPLE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

# get example names by remove extension
foreach(EXPFILE ${EXAMPLE_SRCS})
  get_filename_component(EXMP ${EXPFILE} NAME_WE)
  list(APPEND EXAMPLE_LIST ${EXMP})
endforeach()

# remove blocked examples
list(REMOVE_ITEM EXAMPLE_LIST ${EXCLUDED_EXAMPLES})

# add all unit example targets
foreach(EXMP ${EXAMPLE_LIST})
  add_executable(${EXMP} "${EXMP}.cpp")

  if(WIN32)
    set_target_properties(${EXMP} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  endif()

  target_link_libraries(${EXMP} ${PROJECT_NAME})
endforeach()

