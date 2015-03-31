PROJECT(ETH-SEGREG)
cmake_minimum_required(VERSION 2.6)
INCLUDE_REGULAR_EXPRESSION("^.*$")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

FIND_PACKAGE( Boost 1.49 COMPONENTS filesystem system  program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/source/External/CMakeFindModules/")

if (0)
option(BUILD_SHARED_LIBRARIES "Build (and link to) shared libraries" OFF)
if( ${BUILD_SHARED_LIBRARIES} MATCHES "ON" )
  SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  SET(CMAKE_EXE_LINKER_FLAGS "-static") 
else()
  SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
  SET(CMAKE_EXE_LINKER_FLAGS "") 
endif()
endif(0)

add_subdirectory(source)

# add a target to generate API documentation with Doxygen


option(BUILD_DOC "Build documentation (make doc)" OFF)


if( ${BUILD_DOC} MATCHES "ON" )

find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

endif()
