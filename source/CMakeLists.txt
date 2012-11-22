PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
find_package(STATISMO REQUIRED)

set (STATISMO_INCLUDE_DIR ${STATISMO_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )
set(MATLAB_ROOT /usr/pack/matlab-7.13r2011b-sd/ )
#message(${CMAKE_MODULE_PATH})
find_package(MATLAB REQUIRED)
#include(${MATLAB_USE_FILE})

SUBDIRS(Common External SegmentationRegistration Applications Tools Propagation)


