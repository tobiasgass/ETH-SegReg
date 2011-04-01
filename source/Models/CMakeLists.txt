INCLUDE_DIRECTORIES(
    /usr/local/include
    /usr/include/libxml2
    /home/gasst/work/include
    ../External/RF
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
  MRF-FAST-PD.h
  MRF-TRW-S.h
  Graph.h
  Graph-ITKStyle.h
  Potential-Registration-NCC.h
  Potential-SRS-NCC.h
)

ADD_LIBRARY(SRSMRFModels ${SRSMRFModels_SRC} )
TARGET_LINK_LIBRARIES (SRSMRFModels RF ITKBasicFilters)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
