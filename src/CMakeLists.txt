cmake_minimum_required(VERSION 2.8.3)

set(TARGET lib_polynomial)

find_package(osqp REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(${TARGET} 
  polynomial_solver.cc 
  polynomial_solver.h
  polynomial_sampler.cc
  polynomial_sampler.h
  polynomial_path.h
  path_constraint.h)
target_link_libraries(${TARGET} PRIVATE 
  osqp::osqpstatic 
  Eigen3::Eigen
  ${CMAKE_DL_LIBS})
