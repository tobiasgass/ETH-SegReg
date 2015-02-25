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
ADD_EXECUTABLE(SRS3DCategorical Multiresolution-SRS3D-Categorical.cxx )
ADD_EXECUTABLE(SRS2DMultilabel Multiresolution-SRS2D-Multilabel.cxx )



LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${SRS-MRF_SOURCE_DIR}/SimultaneousRegistrationSegmentation/)

#TARGET_LINK_LIBRARIES(SRS3D ugmix maxflow GCO  TRWS     RandomForest    ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS3D Utils ${Boost_LIBRARIES}   ${ITK_LIBRARIES}  )



#optional linking
if( ${USE_RF} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRS3D  RandomForest     )
endif()


TARGET_LINK_LIBRARIES(SRS2D Utils ${Boost_LIBRARIES} ugmix maxflow  GCO  TRWS     RandomForest     ${ITK_LIBRARIES}  )

TARGET_LINK_LIBRARIES(SRS3DMultilabel Utils ${Boost_LIBRARIES} ugmix maxflow   TRWS     RandomForest     ${ITK_LIBRARIES}  )
TARGET_LINK_LIBRARIES(SRS3DCategorical Utils  ${Boost_LIBRARIES} ugmix maxflow   TRWS     RandomForest     ${ITK_LIBRARIES}  )

TARGET_LINK_LIBRARIES(SRS2DMultilabel Utils ${Boost_LIBRARIES} ugmix maxflow   TRWS     RandomForest     ${ITK_LIBRARIES}  )


ADD_EXECUTABLE(SRS3D-Bone SRS3D-Bone.cxx )
TARGET_LINK_LIBRARIES(SRS3D-Bone maxflow Utils ${Boost_LIBRARIES}  TRWS     RandomForest     ${ITK_LIBRARIES}  )


#ARS

ADD_EXECUTABLE(ARS2D Multiresolution-ARS2D.cxx )
TARGET_LINK_LIBRARIES(ARS2D Utils ugmix maxflow ${Boost_LIBRARIES}  TRWS     RandomForest     ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ARS3D Multiresolution-ARS3D.cxx )
TARGET_LINK_LIBRARIES(ARS3D Utils ugmix maxflow ${Boost_LIBRARIES}  TRWS     RandomForest     ${ITK_LIBRARIES}  )
