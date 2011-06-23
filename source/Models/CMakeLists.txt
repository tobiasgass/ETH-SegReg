INCLUDE_DIRECTORIES(
    /usr/local/include
    /usr/include/libxml2
    /home/gasst/work/include
    ../External/RF
    ../External/fastPD
    ../Common
 )

LINK_DIRECTORIES(
    /home/gasst/work/lib
 )
SET(SRSMRFModels_SRC
  Classifier.h
  BaseLabel.h
  MRF-TRW-S.h
  Graph.h
  SRSConfig.h
  HierarchicalSRSImageToImageFilter.h
  Potential-Registration-Pairwise.h
  Potential-Registration-Unary.h
  Potential-Segmentation-Unary.h
  Potential-SegmentationRegistration-Pairwise.h
  dummy.cxx  
)

ADD_LIBRARY(SRSMRFModels ${SRSMRFModels_SRC} )
TARGET_LINK_LIBRARIES (SRSMRFModels RF ITKBasicFilters FastPD)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
