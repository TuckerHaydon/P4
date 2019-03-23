cmake_minimum_required(VERSION 2.8.3)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.40.0 COMPONENTS filesystem system iostreams REQUIRED)

add_executable(3D 3D.cc gnuplot-iostream.h)
target_include_directories(3D PUBLIC ../src)
target_link_libraries(3D 
  lib_p4
  ${Boost_LIBRARIES} 
  Eigen3::Eigen
)

add_executable(speed speed.cc)
target_include_directories(speed PUBLIC ../src)
target_link_libraries(speed 
  lib_p4
  Eigen3::Eigen
)
