PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#find_package(opengm2 REQUIRED)
#include(${opengm2_use_file})
SUBDIRS(Common External)

OPTION(TCMALLOC "Use tcmalloc" OFF)



option( BUILD_SRS "Build SRS" ON )

if( ${BUILD_SRS} MATCHES "ON" )
  add_subdirectory( SegmentationRegistration )
endif()

option( BUILD_PROPAGATION "Build Segmentation/Registration Propagation" OFF )
if( ${BUILD_PROPAGATION} MATCHES "ON" )
   add_subdirectory( Propagation )
endif()

option( BUILD_TOOLS "Build General Tools" OFF )
if( ${BUILD_TOOLS} MATCHES "ON" )
   add_subdirectory( Tools )
endif()

#option( BUILD_SHARED_LIBRARIES "Build shared libraries" OFF)

#set(${LIBRARY_TYPE} STATIC)
#if (${BUILD_SHARED_LIBRARIES} MATCHES "ON" )
#  set (${LIBRARY_TYPE} SHARED)
#endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )
#set(MATLAB_ROOT /usr/pack/matlab-7.13r2011b-sd/ )
#set(MATLAB_ROOT /usr/pack/matlab-8.0r2012b-bs/ )

#include(${MATLAB_USE_FILE})



