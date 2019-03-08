cmake_minimum_required(VERSION 3.1)
project(MinSnap)
set(CMAKE_CXX_STANDARD 14)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# add_subdirectory(lib)
add_subdirectory(src)
# add_subdirectory(examples)
# add_subdirectory(test) 
