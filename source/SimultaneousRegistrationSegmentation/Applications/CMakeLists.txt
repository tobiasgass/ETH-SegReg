
include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/../Potentials/
  ${CMAKE_CURRENT_SOURCE_DIR}/../Graphs/
  ${CMAKE_CURRENT_SOURCE_DIR}/../MultiResolution/
  ${CMAKE_CURRENT_SOURCE_DIR}/../Optimizers/
) 
#SRS
ADD_EXECUTABLE(SRS3D Multiresolution-SRS3D.cxx )
ADD_EXECUTABLE(SRS3DMultilabel Multiresolution-SRS3D-Multilabel.cxx )
if( ${USE_RF} MATCHES "ON" )

ADD_EXECUTABLE(SRS2D Multiresolution-SRS2D.cxx )
ADD_EXECUTABLE(SRS2DMultilabel Multiresolution-SRS2D-Multilabel.cxx )
endif()


#LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/)

TARGET_LINK_LIBRARIES(SRS3D Utils SRSPotentials   ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS3DMultilabel Utils SRSPotentials
${ITK_LIBRARIES}  )
if( ${USE_RF} MATCHES "ON" )
TARGET_LINK_LIBRARIES(SRS2D Utils SRSPotentials   ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS2DMultilabel Utils SRSPotentials     ${ITK_LIBRARIES}  )
endif()
#optional linking
if( ${USE_TRWS} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  TRWS_LIBRARIES     )
  if( ${USE_RF} MATCHES "ON" )

  TARGET_LINK_LIBRARIES(SRS2D  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  TRWS_LIBRARIES     )
  endif()
endif()
#optional linking
if( ${USE_GC} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  GC     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  GC     )
  if( ${USE_RF} MATCHES "ON" )

 TARGET_LINK_LIBRARIES(SRS2D  GC     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  GC     )
endif()
endif()
#optional linking
if( ${USE_GCO} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  GCO     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  GCO     )
  if( ${USE_RF} MATCHES "ON" )

 TARGET_LINK_LIBRARIES(SRS2D  GCO     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  GCO     )
endif()
endif()

#optional linking
if( ${USE_OPENGM} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D    ${Boost_LIBRARIES}   )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel    ${Boost_LIBRARIES}   )
  if( ${USE_RF} MATCHES "ON" )
 TARGET_LINK_LIBRARIES(SRS2D    ${Boost_LIBRARIES}   )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel    ${Boost_LIBRARIES}   )
  endif()
endif()




if (0)


ADD_EXECUTABLE(SRS3DCategorical Multiresolution-SRS3D-Categorical.cxx )


#ARS
###removed from build tree
### might or might not work, stopped to support those explicitly quite some time ago
ADD_EXECUTABLE(ARS2D Multiresolution-ARS2D.cxx )
TARGET_LINK_LIBRARIES(ARS2D Utils ugmix maxflow ${Boost_LIBRARIES}  TRWS     RandomForest     ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ARS3D Multiresolution-ARS3D.cxx )
TARGET_LINK_LIBRARIES(ARS3D Utils ugmix maxflow ${Boost_LIBRARIES}  TRWS     RandomForest     ${ITK_LIBRARIES}  )

endif()
