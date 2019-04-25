cmake_minimum_required(VERSION 3.1)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.40.0 COMPONENTS filesystem system iostreams REQUIRED)
find_package(osqp REQUIRED)



add_executable(3D 3D.cc gnuplot-iostream.h)
target_link_libraries(3D 
  lib_p4
  ${Boost_LIBRARIES} 
  Eigen3::Eigen
  osqp::osqpstatic
)

set_target_properties(3D PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

target_compile_options(3D PRIVATE 
  -Wfatal-errors
)



add_executable(speed speed.cc)
target_include_directories(speed PUBLIC ../src)
target_link_libraries(speed 
  lib_p4
  Eigen3::Eigen
  osqp::osqpstatic
)

set_target_properties(speed PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

target_compile_options(speed PRIVATE 
  -Wfatal-errors
)
