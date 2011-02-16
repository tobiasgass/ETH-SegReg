PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
LINK_DIRECTORIES(
    /home/gasst/work/lib
    /usr/lib
)


SUBDIRS(Common External Models Applications )


