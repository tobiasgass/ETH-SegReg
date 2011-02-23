INCLUDE_DIRECTORIES(
    /usr/local/include
    /usr/include/libxml2
    /home/gasst/work/include
 )

LINK_DIRECTORIES(
    /home/gasst/work/lib
 )
SET(SRSMRFModels_SRC
  Potentials.h
  Classifier.h
  BaseLabel.h
  BasePotential.h
  Label.h
  RegistrationSegmentationLabel.h
  RegistrationSegmentationPotentials-v2.h
  AlternatingSegmentationRegistrationLabel.h
  SegmentationLabel.h
  MRF.h
  Grid.h
  Graph.h
  FAST-PD-mrf-optimisation.h
  TRW-S-Registration.h
  ImageRegistrationFilter.cxx
)

ADD_LIBRARY(SRSMRFModels ${SRSMRFModels_SRC} )
TARGET_LINK_LIBRARIES (SRSMRFModels ITKBasicFilters)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
