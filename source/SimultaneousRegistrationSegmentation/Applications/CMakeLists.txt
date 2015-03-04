#PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")

include_directories (
  ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/Potentials
  ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/Graphs
  ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/MultiResolution
  ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/Optimizers
) 




#SRS
ADD_EXECUTABLE(SRS3D Multiresolution-SRS3D.cxx )
ADD_EXECUTABLE(SRS2D Multiresolution-SRS2D.cxx )
ADD_EXECUTABLE(SRS3DMultilabel Multiresolution-SRS3D-Multilabel.cxx )
ADD_EXECUTABLE(SRS2DMultilabel Multiresolution-SRS2D-Multilabel.cxx )



LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/)

TARGET_LINK_LIBRARIES(SRS3D Utils    ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS2D Utils    ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS3DMultilabel Utils      ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS2DMultilabel Utils      ${ITK_LIBRARIES}  )

#optional linking
if( ${USE_TRWS} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS2D  TRWS_LIBRARIES     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  TRWS_LIBRARIES     )
endif()
#optional linking
if( ${USE_GC} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  GC     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  GC     )
 TARGET_LINK_LIBRARIES(SRS2D  GC     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  GC     )
endif()
#optional linking
if( ${USE_GCO} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  GCO     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  GCO     )
 TARGET_LINK_LIBRARIES(SRS2D  GCO     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  GCO     )
endif()

#optional linking
if( ${USE_OPENGM} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D    ${Boost_LIBRARIES}   )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel    ${Boost_LIBRARIES}   )
 TARGET_LINK_LIBRARIES(SRS2D    ${Boost_LIBRARIES}   )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel    ${Boost_LIBRARIES}   )
endif()

#optional linking
if( ${USE_RF} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  RandomForest     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  RandomForest     )
  TARGET_LINK_LIBRARIES(SRS2D  RandomForest     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  RandomForest     )
endif()
#optional linking
if( ${USE_CUGMIX} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  ugmix     )
  TARGET_LINK_LIBRARIES(SRS3DMultilabel  ugmix     )
  TARGET_LINK_LIBRARIES(SRS2D  ugmix     )
  TARGET_LINK_LIBRARIES(SRS2DMultilabel  ugmix     )
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