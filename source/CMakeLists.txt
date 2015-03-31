
##this should include "random" for some tools, but seems to compile&link without it, and CMake can't find boost_random ...
FIND_PACKAGE( Boost 1.40 COMPONENTS  system filesystem program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/Common) 
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/Utils) 
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Utils) 
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/External/InsightJournal) 


####Options
option( BUILD_SRS "Build SRS" ON )

if( ${BUILD_SRS} MATCHES "ON" )
  
  option( USE_TRWS "Use TRWS optimizer" OFF )
  option( USE_GCO "Use GCO optimizer" OFF )
  option( USE_GC "Use GC optimizer" OFF )

endif()

option( BUILD_CBRR "Build ConsistencyBasedRegistrationRectification" OFF )
if( ${BUILD_CBRR} MATCHES "ON" )
   
endif()

option( BUILD_SEMISUPERVISEDSEGMENTATIONPROPAGATION "Build  SemiSupervisedSegmentationPropagation" OFF )
if( ${BUILD_SEMISUPERVISEDSEGMENTATIONPROPAGATION} MATCHES "ON" )
   
   option( USE_GCO "Use GCO optimizer" OFF )

endif()

option( BUILD_MARKOVRANDOMFIELDREGISTRATIONFUSION "Build  MarkovRandomFieldRegistrationFusion" OFF )
if( ${BUILD_MARKOVRANDOMFIELDREGISTRATIONFUSION} MATCHES "ON" )
   
   set(USE_TRWS "ON" CACHE BOOL  "Use TRWS optimizer" FORCE)

endif()


if( ${USE_TRWS} MATCHES "ON" )
  set(DIR_TRWS "/home/gasst/Downloads/TRW_S-v1.3/" CACHE  FILEPATH "Directory for TRWS") 
  if (NOT EXISTS ${DIR_TRWS}/MRFEnergy.h)
    message(SEND_ERROR "TRW-S directory not found or does not appear to contain the TRW-S library")
    set(DIR_TRWS "TRW-S DIR NOTFOUND" CACHE  FILEPATH "Directory for TRWS" FORCE) 
  else()

    #patch it!
    MESSAGE( STATUS "Patching TRWS")
    EXECUTE_PROCESS( COMMAND cmake -E chdir ${DIR_TRWS} patch -r - -N -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/External/Patches/TRWS.patch  RESULT_VARIABLE RESULT_PATCH OUTPUT_QUIET)
    MESSAGE( STATUS "Patch result: ${RESULT_PATCH}, PROF SOURCE DIR: ${DIR_TRWS}" )
    if (  ${RESULT_PATCH} MATCHES "2")
      MESSAGE(WARNING "Patching TRWS failed, see log for details, returned ${RESULT_PATCH}")
    endif()
    
    
    add_definitions(-DWITH_TRWS)
    include_directories( ${DIR_TRWS} )
    FILE(GLOB trwsFiles
      ${DIR_TRWS}/minimize.cpp   ${DIR_TRWS}/MRFEnergy.h   ${DIR_TRWS}/treeProbabilities.cpp  ${DIR_TRWS}/instances.inc
      ${DIR_TRWS}/instances.h  ${DIR_TRWS}/MRFEnergy.cpp  ${DIR_TRWS}/ordering.cpp  typeBinaryFast.h       ${DIR_TRWS}/typeGeneral.h  
      )
    #message("${trwsFiles}")
    ADD_LIBRARY(TRWS_LIBRARIES  ${trwsFiles}   ) 
  endif()
endif()


if( ${USE_GCO} MATCHES "ON" )
  add_definitions(-DWITH_GCO -DGCO_ENERGYTYPE=double -DGCO_ENERGYTERMTYPE=float)
  set(DIR_GCO "${SRS-MRF_SOURCE_DIR}/External/GCO" CACHE  FILEPATH "Directory for GCO")
  if (NOT EXISTS ${DIR_GCO}/GCoptimization.h)
    message(SEND_ERROR "GCO directory not found or does not appear to contain the GCO library")
    set(DIR_GCO "maxflow DIR NOTFOUND" CACHE  FILEPATH "Directory for GCO" FORCE) 
  else()
    
      #patch it!
    MESSAGE( STATUS "Patching GCO")
    EXECUTE_PROCESS( COMMAND cmake -E chdir ${DIR_GCO} patch -r - -N -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/External/Patches/GCO.patch  RESULT_VARIABLE RESULT_PATCH OUTPUT_QUIET)
    MESSAGE( STATUS "Patch result: ${RESULT_PATCH}, PROF SOURCE DIR: ${DIR_GCO}" )
    if (  ${RESULT_PATCH} MATCHES "2")
      MESSAGE(WARNING "Patching GCO failed, see log for details, returned ${RESULT_PATCH}")
    endif()
  
    
    include_directories( ${DIR_GCO} )
    FILE(GLOB gcoFiles 
      ${DIR_GCO}/block.h   ${DIR_GCO}/energy.h   ${DIR_GCO}/GCoptimization.cpp   ${DIR_GCO}/GCoptimization.h  ${DIR_GCO}/graph.cpp   ${DIR_GCO}/graph.h   ${DIR_GCO}/LinkedBlockList.cpp   ${DIR_GCO}/LinkedBlockList.h     ${DIR_GCO}/maxflow.cpp )
    ADD_LIBRARY(GCO
      ${gcoFiles}
      ) 
    endif()

endif()


if( ${USE_GC} MATCHES "ON" )
  set(DIR_GC "${SRS-MRF_SOURCE_DIR}/External/maxFlow" CACHE  FILEPATH "Directory for GC") 
  if (NOT EXISTS ${DIR_GC}/maxflow.cpp)
    message(SEND_ERROR "maxflow directory not found or does not appear to contain the maxflow library")
    set(DIR_GC "maxflow DIR NOTFOUND" CACHE  FILEPATH "Directory for GC" FORCE) 
  else()
    add_definitions(-DWITH_GC )
    include_directories( ${DIR_GC} )
    
    FILE(GLOB gcFiles ${DIR_GC}/block.h   ${DIR_GC}/energy.h   ${DIR_GC}/graph.cpp   ${DIR_GC}/graph.h   ${DIR_GC}/LinkedBlockList.cpp   ${DIR_GC}/LinkedBlockList.h   ${DIR_GC}/maxflow.cpp )

    
    ADD_LIBRARY(GC
      ${gcFiles}
      ) 
  endif()

endif()

option( BUILD_TOOLS "Build General Tools and Utilities" OFF )
if( ${BUILD_TOOLS} MATCHES "ON" )
   add_subdirectory( Tools )
endif()

####subdirectories

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

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )


