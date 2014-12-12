PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#find_package(opengm2 REQUIRED)
#include(${opengm2_use_file})


find_package(STATISMO REQUIRED)
OPTION(TCMALLOC "Use tcmalloc" OFF)
set (STATISMO_INCLUDE_DIR ${STATISMO_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )
#set(MATLAB_ROOT /usr/pack/matlab-7.13r2011b-sd/ )
#set(MATLAB_ROOT /usr/pack/matlab-8.0r2012b-bs/ )
set(MATLAB_ROOT /usr/pack/matlab-8.3r2014a-fg/ )
#message(${CMAKE_MODULE_PATH})
find_package(MATLAB REQUIRED)
#include(${MATLAB_USE_FILE})

SUBDIRS(Common External SegmentationRegistration Applications Tools Propagation)


