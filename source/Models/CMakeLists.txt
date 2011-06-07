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
  BasePotential.h
  BaseMRF.h
  SRSPotential.cxx
  SRSPotential.h
  SRSPotential-Bone.h
  MRF-FAST-PD.h
  MRF-TRW-S.h
  MRF-GC.h
  Graph.h
  Graph-ITKStyle.h
  Potential-SRS-NCC.h
  Potential-SRS-NCC-Bone.h
  Potential-SRS-MultiVariateNCC.h
  Potential-SRS-SAD-SACentered.h
  Potential-SRS-SAD-PAX_SXSAT.h
  SRSConfig.h
  HierarchicalSRSImageToImageFilter.h
)

ADD_LIBRARY(SRSMRFModels ${SRSMRFModels_SRC} )
TARGET_LINK_LIBRARIES (SRSMRFModels RF ITKBasicFilters FastPD)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
