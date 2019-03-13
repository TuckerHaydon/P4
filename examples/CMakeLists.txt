cmake_minimum_required(VERSION 2.8.3)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(3D 3D.cc)
target_include_directories(3D PUBLIC ../src)
target_link_libraries(3D 
  lib_polynomial
  Eigen3::Eigen
)

