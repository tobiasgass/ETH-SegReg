#find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

ADD_LIBRARY(Utils
  ImageUtils.cxx
  FilterUtils.hpp
  argstream.h
  itkDisplacementFieldCompositionFilter.h
  ChamferDistanceTransform.h
  Log.h
  SheetnessFilter.h
  Preprocessing.h
  TransformationUtils.h
  SegmentationMapper.hxx
  Metrics.h
  TemporalMedianImageFilter.h
)
TARGET_LINK_LIBRARIES(Utils ${ITK_LIBRARIES}  )

