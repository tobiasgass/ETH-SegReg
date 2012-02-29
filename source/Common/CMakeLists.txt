#find_package(ITK REQUIRED)
#include(${ITK_USE_FILE})

ADD_LIBRARY(Utils
  ImageUtils.cxx
  FilterUtils.hpp
  argstream.h
  itkDisplacementFieldCompositionFilter.h
  ChamferDistanceTransform.h
)
TARGET_LINK_LIBRARIES(Utils ${ITK_LIBRARIES}  )

