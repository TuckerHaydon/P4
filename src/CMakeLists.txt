cmake_minimum_required(VERSION 3.1)

set(TARGET lib_p4)

find_package(osqp REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(SOURCE_FILES
  polynomial_solver.cc 
  polynomial_sampler.cc
  common.cc
)

add_library(${TARGET} STATIC ${SOURCE_FILES})

target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${TARGET} PUBLIC
  osqp::osqpstatic
  Eigen3::Eigen
  "dl"
)

set_target_properties(${TARGET} PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

target_compile_options(${TARGET} PRIVATE 
  -Wfatal-errors
)
