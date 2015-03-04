PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

FIND_PACKAGE( Boost 1.40 COMPONENTS system filesystem program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

option(BUILD_SHARED_LIBRARIES "Build (and link to) shared libraries" OFF)
if( ${BUILD_SHARED_LIBRARIES} MATCHES "ON" )
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
SET(CMAKE_EXE_LINKER_FLAGS "-static") 
else()
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
SET(CMAKE_EXE_LINKER_FLAGS "") 
endif()
SUBDIRS(Common)
SUBDIRS(Utils)
include_directories(Common)
include_directories(Utils)

OPTION(TCMALLOC "Use tcmalloc" OFF)



option( BUILD_SRS "Build SRS" ON )

if( ${BUILD_SRS} MATCHES "ON" )
  add_subdirectory( SimultaneousRegistrationSegmentation )
endif()

option( BUILD_CBRR "Build ConsistencyBasedRegistrationRectification" OFF )
if( ${BUILD_CBRR} MATCHES "ON" )
   add_subdirectory( ConsistencyBasedRegistrationRectification )
endif()

option( BUILD_SEMISUPERVISEDSEGMENTATIONPROPAGATION "Build  SemiSupervisedSegmentationPropagation" OFF )
if( ${BUILD_SEMISUPERVISEDSEGMENTATIONPROPAGATION} MATCHES "ON" )
   add_subdirectory( SemiSupervisedSegmentationPropagation )
endif()

option( BUILD_MARKOVRANDOMFIELDREGISTRATIONFUSION "Build  MarkovRandomFieldRegistrationFusion" OFF )
if( ${BUILD_MARKOVRANDOMFIELDREGISTRATIONFUSION} MATCHES "ON" )
   add_subdirectory( MarkovRandomFieldRegistrationFusion )
endif()

option( BUILD_TOOLS "Build General Tools and Utilities" OFF )
if( ${BUILD_TOOLS} MATCHES "ON" )
   add_subdirectory( Tools )
endif()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )


