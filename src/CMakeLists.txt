cmake_minimum_required(VERSION 2.8.3)

set(TARGET min_snap)

find_package(osqp REQUIRED)
find_package(OsqpEigen)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${TARGET} main.cc)
target_link_libraries(${TARGET} PRIVATE osqp::osqpstatic OsqpEigen::OsqpEigen Eigen3::Eigen ${CMAKE_DL_LIBS})
