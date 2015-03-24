
include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/../Potentials/
  ${CMAKE_CURRENT_SOURCE_DIR}/../Graphs/
  ${CMAKE_CURRENT_SOURCE_DIR}/../MultiResolution/
  ${CMAKE_CURRENT_SOURCE_DIR}/../Optimizers/
) 
#SRS
ADD_EXECUTABLE(SRS3D-Bone SRS3D-Bone.cxx )
ADD_EXECUTABLE(SRS3D-Bone-Multilabel SRS3D-Bone-Multilabel.cxx )
ADD_EXECUTABLE(SRS2D-Bone SRS2D-Bone.cxx )


if( ${USE_RF} MATCHES "ON" )

ADD_EXECUTABLE(SRS2D-Classifier SRS2D-Classifier.cxx )
ADD_EXECUTABLE(SRS2D-Classifier-Multilabel
SRS2D-Classifier-Multilabel.cxx )

endif()


#LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/)

TARGET_LINK_LIBRARIES(SRS3D-Bone Utils SRSPotentials   ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS3D-Bone-Multilabel Utils SRSPotentials
${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS2D-Bone Utils SRSPotentials   ${ITK_LIBRARIES}  )

if( ${USE_RF} MATCHES "ON" )
TARGET_LINK_LIBRARIES(SRS2D-Classifier Utils SRSPotentials   ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS2D-Classifier-Multilabel Utils SRSPotentials     ${ITK_LIBRARIES}  )
endif()
#optional linking
if( ${USE_TRWS} MATCHES "ON" )
    TARGET_LINK_LIBRARIES(SRS2D-Bone  TRWS_LIBRARIES     )
TARGET_LINK_LIBRARIES(SRS3D-Bone  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS3D-Bone-Multilabel  TRWS_LIBRARIES     )
  if( ${USE_RF} MATCHES "ON" )

  TARGET_LINK_LIBRARIES(SRS2D-Classifier  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS2D-Classifier-Multilabel  TRWS_LIBRARIES     )
  endif()
endif()
#optional linking
if( ${USE_GC} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D-Bone  GC     )
  TARGET_LINK_LIBRARIES(SRS3D-Bone-Multilabel  GC     )
  if( ${USE_RF} MATCHES "ON" )

 TARGET_LINK_LIBRARIES(SRS2D-Classifier  GC     )
  TARGET_LINK_LIBRARIES(SRS2D-Classifier-Multilabel  GC     )
endif()
endif()
#optional linking
if( ${USE_GCO} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D-Bone  GCO     )
  TARGET_LINK_LIBRARIES(SRS3D-Bone-Multilabel  GCO     )
  if( ${USE_RF} MATCHES "ON" )

  TARGET_LINK_LIBRARIES(SRS2D-Classifier  GCO     )
  TARGET_LINK_LIBRARIES(SRS2D-Classifier-Multilabel  GCO     )


endif()
  TARGET_LINK_LIBRARIES(SRS2D-Bone  GCO     )

endif()

#optional linking
if( ${USE_OPENGM} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D-Bone    ${Boost_LIBRARIES}   )
  TARGET_LINK_LIBRARIES(SRS3D-Bone-Multilabel    ${Boost_LIBRARIES}   )
  if( ${USE_RF} MATCHES "ON" )
 TARGET_LINK_LIBRARIES(SRS2D-Classifier    ${Boost_LIBRARIES}   )
  TARGET_LINK_LIBRARIES(SRS2D-Classifier-Multilabel    ${Boost_LIBRARIES}   )
  endif()
    TARGET_LINK_LIBRARIES(SRS2D-Bone    ${Boost_LIBRARIES}   )

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
