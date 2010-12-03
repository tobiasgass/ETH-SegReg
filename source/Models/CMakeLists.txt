
SET(SRSMRFModels_SRC
  Potentials.h
  Label.h
  MRF.h
  ImageRegistrationFilter.cxx
)

ADD_LIBRARY(SRSMRFModels ${SRSMRFModels_SRC} )
TARGET_LINK_LIBRARIES (SRSMRFModels)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
