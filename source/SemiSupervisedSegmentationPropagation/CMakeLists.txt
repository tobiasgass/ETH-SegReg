PROJECT(SemiSupervisedSegmentationPropagatiob)




#########################
#MAIN CODE USED FOR PAPER

#seg-propagation
ADD_EXECUTABLE(SegProp2D SegmentationPropagation-2D.cxx )
TARGET_LINK_LIBRARIES(SegProp2D   Utils  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegProp3D SegmentationPropagation-3D.cxx )
TARGET_LINK_LIBRARIES(SegProp3D   Utils   ${ITK_LIBRARIES}  )


#########################
#EXPERIMENTAL CODE, NOT TESTED WELL



#Fusion
ADD_EXECUTABLE(SegFuse2D SegmentationFusion-2D.cxx )
TARGET_LINK_LIBRARIES(SegFuse2D   Utils  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegFuse3D SegmentationFusion-3D.cxx )
TARGET_LINK_LIBRARIES(SegFuse3D   Utils   ${ITK_LIBRARIES}  )


if (${USE_GCO} MATCHES "ON")
  TARGET_LINK_LIBRARIES(SegFuse2D   GCO  )
  TARGET_LINK_LIBRARIES(SegFuse3D   GCO  )
  TARGET_LINK_LIBRARIES(SegProp3D   GCO  )
  TARGET_LINK_LIBRARIES(SegProp2D   GCO  )
  
  
endif()

###deprecated
if (0)

ADD_EXECUTABLE(SegNetwork2D-ConnectivityTest SegNetwork-Connectivity.cxx )
TARGET_LINK_LIBRARIES( SegNetwork2D-ConnectivityTest   Utils TRWS    ${ITK_LIBRARIES}  )

#seg-network
  
  ADD_EXECUTABLE(SegNetwork2D SegmentationNetwork-2D.cxx )
  TARGET_LINK_LIBRARIES(SegNetwork2D Utils GCO   ${ITK_LIBRARIES}  )
  ADD_EXECUTABLE(SegNetwork3D SegmentationNetwork-3D.cxx )
  TARGET_LINK_LIBRARIES(SegNetwork3D   Utils GCO  ${ITK_LIBRARIES}  )
endif()
