find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

ADD_LIBRARY(Utils
  bimap.h
  ImageUtils.cxx
  FilterUtils.hpp
  argstream.h
  itkDisplacementFieldCompositionFilter.h
)
TARGET_LINK_LIBRARIES(Utils ITKCommon ITKIO )

