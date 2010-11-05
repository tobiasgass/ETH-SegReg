PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)

SUBDIRS(Common External Models Applications )

#include(${ITK_USE_FILE})
