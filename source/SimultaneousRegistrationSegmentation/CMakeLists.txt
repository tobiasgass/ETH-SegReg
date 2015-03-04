include(${ITK_USE_FILE})



option( USE_GCO "Use GCO optimizer" ON )

if( ${USE_GCO} MATCHES "ON" )
  add_definitions(-DWITH_GCO -DGCO_ENERGYTYPE=double -DGCO_ENERGYTERMTYPE=float)
  set(DIR_GCO "${SRS-MRF_SOURCE_DIR}/External/GCO" CACHE  FILEPATH "Directory for GCO")
  #  option( GCO_DIR "GCO directory" ${SRS-MRF_SOURCE_DIR}/External/GCO )
  include_directories( ${DIR_GCO} )
  FILE(GLOB gcoFiles 
     ${DIR_GCO}/block.h   ${DIR_GCO}/energy.h   ${DIR_GCO}/GCoptimization.cpp   ${DIR_GCO}/GCoptimization.h  ${DIR_GCO}/graph.cpp   ${DIR_GCO}/graph.h   ${DIR_GCO}/LinkedBlockList.cpp   ${DIR_GCO}/LinkedBlockList.h     ${DIR_GCO}/maxflow.cpp )
  ADD_LIBRARY(GCO
    ${gcoFiles}
    ) 

endif()

option( USE_GC "Use GC optimizer" ON )
if( ${USE_GC} MATCHES "ON" )
  add_definitions(-DWITH_GC )

  set(DIR_GC "${SRS-MRF_SOURCE_DIR}/External/maxFlow" CACHE  FILEPATH "Directory for GC")
  include_directories( ${DIR_GC} )

  FILE(GLOB gcFiles ${DIR_GC}/*.h ${DIR_GC}/*.cxx ${DIR_GC}/*.cpp ${DIR_GC}/*.hxx ${DIR_GC}/*.hpp ${DIR_GC}/*.c)

  ADD_LIBRARY(GC
    ${gcFiles}
    ) 

endif()

option( USE_TRWS "Use TRWS optimizer" ON )
if( ${USE_TRWS} MATCHES "ON" )
  add_definitions(-DWITH_TRWS)
  #set(DIR_TRWS "${SRS-MRF_SOURCE_DIR}/External/TRWS" CACHE  FILEPATH "Directory for TRWS")
  set(DIR_TRWS "/home/gasst/Downloads/TRW_S-v1.3" CACHE  FILEPATH "Directory for TRWS")
  include_directories( ${DIR_TRWS} )
  FILE(GLOB trwsFiles
    ${DIR_TRWS}/minimize.cpp   ${DIR_TRWS}/MRFEnergy.h   ${DIR_TRWS}/treeProbabilities.cpp  ${DIR_TRWS}/instances.inc
    ${DIR_TRWS}/instances.h  ${DIR_TRWS}/MRFEnergy.cpp  ${DIR_TRWS}/ordering.cpp  typeBinaryFast.h       ${DIR_TRWS}/typeGeneral.h  
    )
  #message("${trwsFiles}")
  ADD_LIBRARY(TRWS_LIBRARIES  ${trwsFiles}   ) 
endif()

option( USE_OPENGM "Use OPENGM optimizer" ON )
if( ${USE_OPENGM} MATCHES "ON" )
  add_definitions(-DWITH_OPENGM)
endif()

option( USE_CUGMIX "Use c-ugmix gaussian mixture estimator" ON )
if( ${USE_CUGMIX} MATCHES "ON" )
  add_definitions(-DWITH_CUGMIX)
  include_directories(${SRS-MRF_SOURCE_DIR}/External/c-ugmix)
  add_subdirectory(../External/c-ugmix ${CMAKE_BINARY_DIR}/External/c-ugmix)

endif()

option( USE_RF "Use random forest classifier" ON )
if( ${USE_RF} MATCHES "ON" )
  add_definitions(-DWITH_RF)
  include_directories(../External/RF)
  INCLUDE_DIRECTORIES("${DIR_LIBXML}/")
  add_subdirectory(../External/RF ${CMAKE_BINARY_DIR}/External/RF)
endif()


add_subdirectory(Applications)

FILE(GLOB optHeaders "Optimizers/*.h")
FILE(GLOB graphHeaders "Graphs/*.h")
FILE(GLOB potHeaders "Potentials/*.h")
FILE(GLOB multHeaders "MultiResolution/*.h")



