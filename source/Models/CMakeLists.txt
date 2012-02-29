INCLUDE_DIRECTORIES(
    /usr/include/libxml2
    ../External/RF
    ../External/fastPD
    ../Common
 )


SET(SRSMRFModels_SRC
  Classifier.h
  BaseLabel.h
  MRF-TRW-S.h
  Graph.h
  SubsamplingGraph.h
  FastRegistrationGraph.h
  SRSConfig.h
  HierarchicalSRSImageToImageFilter.h
  Potential-Registration-Pairwise.h
  Potential-Registration-Unary.h
  Potential-Segmentation-Unary.h
  Potential-SegmentationRegistration-Pairwise.h
  SegmentationGraph.h
  SegmentationImageFilter.h
  dummy.cxx  
)

ADD_LIBRARY(SRSMRFModels ${SRSMRFModels_SRC} )
#TARGET_LINK_LIBRARIES (SRSMRFModels RF ITKBasicFilters FastPD )
TARGET_LINK_LIBRARIES (SRSMRFModels RF  ${ITK_LIBRARIES} FastPD tcmalloc)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
