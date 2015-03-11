include_directories (${CMAKE_CURRENT_SOURCE_DIR}/Common) 
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/Utils) 
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Utils) 



option( BUILD_SRS "Build SRS" ON )

if( ${BUILD_SRS} MATCHES "ON" )
  add_subdirectory( SimultaneousRegistrationSegmentation )
  option( USE_TRWS "Use TRWS optimizer" OFF )

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


option( BUILD_TOOLS "Build General Tools and Utilities" OFF )
if( ${BUILD_TOOLS} MATCHES "ON" )
   add_subdirectory( Tools )
endif()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )


