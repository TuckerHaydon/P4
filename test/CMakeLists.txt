cmake_minimum_required(VERSION 2.8.3)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main main.cc)
target_include_directories(main PUBLIC ../src)
target_link_libraries(main 
  lib_polynomial
  Eigen3::Eigen
  )

