PROJECT(SimultaneousRegistrationAndSegmentation)


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


