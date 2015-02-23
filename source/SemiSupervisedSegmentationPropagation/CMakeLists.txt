PROJECT(SemiSupervisedSegmentationPropagatiob)


include_directories (${SRS-MRF_SOURCE_DIR}/Common) 



#########################
#MAIN CODE USED FOR PAPER

#seg-propagation
ADD_EXECUTABLE(SegProp2D SegmentationPropagation-2D.cxx )
TARGET_LINK_LIBRARIES(SegProp2D   Utils  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegProp3D SegmentationPropagation-3D.cxx )
TARGET_LINK_LIBRARIES(SegProp3D   Utils   ${ITK_LIBRARIES}  )


#########################
#EXPERIMENTAL CODE, NOT TESTED WELL


#seg-network
ADD_EXECUTABLE(SegNetwork2D SegmentationNetwork-2D.cxx )
TARGET_LINK_LIBRARIES(SegNetwork2D Utils    ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegNetwork3D SegmentationNetwork-3D.cxx )
TARGET_LINK_LIBRARIES(SegNetwork3D   Utils TRWS  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegNetwork2D-ConnectivityTest SegNetwork-Connectivity.cxx )
TARGET_LINK_LIBRARIES( SegNetwork2D-ConnectivityTest   Utils TRWS    ${ITK_LIBRARIES}  )


#Fusion
ADD_EXECUTABLE(SegFuse2D SegmentationFusion-2D.cxx )
TARGET_LINK_LIBRARIES(SegFuse2D   Utils  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegFuse3D SegmentationFusion-3D.cxx )
TARGET_LINK_LIBRARIES(SegFuse3D   Utils   ${ITK_LIBRARIES}  )


