include(${ITK_USE_FILE})

option( USE_GCO "Use GCO optimizer" OFF )

if( ${USE_GCO} MATCHES "ON" )
  add_definitions(-DWITH_GCO -DGCO_ENERGYTYPE=double -DGCO_ENERGYTERMTYPE=float)
  set(DIR_GCO "${SRS-MRF_SOURCE_DIR}/External/GCO" CACHE  FILEPATH "Directory for GCO")
  if (NOT EXISTS ${DIR_GCO}/GCoptimization.h)
    message(SEND_ERROR "GCO directory not found or does not appear to contain the GCO library")
    set(DIR_GCO "maxflow DIR NOTFOUND" CACHE  FILEPATH "Directory for GCO" FORCE) 
  else()

    include_directories( ${DIR_GCO} )
    FILE(GLOB gcoFiles 
      ${DIR_GCO}/block.h   ${DIR_GCO}/energy.h   ${DIR_GCO}/GCoptimization.cpp   ${DIR_GCO}/GCoptimization.h  ${DIR_GCO}/graph.cpp   ${DIR_GCO}/graph.h   ${DIR_GCO}/LinkedBlockList.cpp   ${DIR_GCO}/LinkedBlockList.h     ${DIR_GCO}/maxflow.cpp )
    ADD_LIBRARY(GCO
      ${gcoFiles}
      ) 
    endif()

endif()

option( USE_GC "Use GC optimizer" OFF )
if( ${USE_GC} MATCHES "ON" )
  set(DIR_GC "${SRS-MRF_SOURCE_DIR}/External/maxFlow" CACHE  FILEPATH "Directory for GC") 
  if (NOT EXISTS ${DIR_GC}/maxflow.cpp)
    message(SEND_ERROR "maxflow directory not found or does not appear to contain the maxflow library")
    set(DIR_GC "maxflow DIR NOTFOUND" CACHE  FILEPATH "Directory for GC" FORCE) 
  else()
    add_definitions(-DWITH_GC )
    include_directories( ${DIR_GC} )
    
    FILE(GLOB gcFiles ${DIR_GC}/*.h ${DIR_GC}/*.cxx ${DIR_GC}/*.cpp ${DIR_GC}/*.hxx ${DIR_GC}/*.hpp ${DIR_GC}/*.c)
    
    ADD_LIBRARY(GC
      ${gcFiles}
      ) 
  endif()

endif()


option( USE_OPENGM "Use OPENGM optimizer" ON )
if( ${USE_OPENGM} MATCHES "ON" )
  add_definitions(-DWITH_OPENGM)
endif()

option( USE_CUGMIX "Use c-ugmix gaussian mixture estimator" ON )
if( ${USE_CUGMIX} MATCHES "ON" )
  add_definitions(-DWITH_CUGMIX)
  include_directories(../External/c-ugmix)
  add_subdirectory(../External/c-ugmix ${CMAKE_BINARY_DIR}/External/c-ugmix)

endif()

option( USE_RF "Use random forest classifier" ON )
if( ${USE_RF} MATCHES "ON" )
  add_definitions(-DWITH_RF)
  include_directories(../External/RF)
  INCLUDE_DIRECTORIES("${DIR_LIBXML}/")
  add_subdirectory(../External/RF ${CMAKE_BINARY_DIR}/External/RF)
endif()

if( NOT ( (${USE_OPENGM} MATCHES "ON") OR ( ${USE_GCO} MATCHES "ON" ) OR ( ${USE_TRWS} MATCHES "ON" )))
  message(SEND_ERROR "No optimizer was chosen which could hande SRS. Please enable at least one.")
endif()


add_subdirectory(Applications)
add_subdirectory(Potentials)

FILE(GLOB optHeaders "Optimizers/*.h")
FILE(GLOB graphHeaders "Graphs/*.h")
FILE(GLOB potHeaders "Potentials/*.h")
FILE(GLOB multHeaders "MultiResolution/*.h")


include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/Potentials/
  ${CMAKE_CURRENT_SOURCE_DIR}/Graphs/
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiResolution/
  ${CMAKE_CURRENT_SOURCE_DIR}/Optimizers/
) 
message(WARNING  ${CMAKE_CURRENT_SOURCE_DIR}/MultiResolution)

