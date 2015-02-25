PROJECT(SRS-MRF)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

FIND_PACKAGE( Boost 1.40 COMPONENTS system filesystem program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
 

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

#option( BUILD_SHARED_LIBRARIES "Build shared libraries" OFF)

#set(${LIBRARY_TYPE} STATIC)
#if (${BUILD_SHARED_LIBRARIES} MATCHES "ON" )
#  set (${LIBRARY_TYPE} SHARED)
#endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )


