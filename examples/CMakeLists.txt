cmake_minimum_required(VERSION 3.1)

find_package(osqp REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

add_executable(speed speed.cc)
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


option(BUILD_3D_EXAMPLE "Build the 3D example. Requires boost." true)
find_package(Boost 1.40.0 COMPONENTS filesystem system iostreams)
IF(Boost_FOUND AND BUILD_3D_EXAMPLE)
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
ELSE()
  MESSAGE("3D example disabled.")
ENDIF()


add_executable(tst tst.cc)
target_link_libraries(tst
  lib_p4
  Eigen3::Eigen
  osqp::osqpstatic
  ${CERES_LIBRARIES}
)

set_target_properties(tst PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

target_compile_options(tst PRIVATE 
  -Wfatal-errors
)
