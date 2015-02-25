# #find_package(ITK REQUIRED)
# include(${ITK_USE_FILE})
# FIND_PACKAGE( Boost 1.40 COMPONENTS system filesystem program_options REQUIRED )
# INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
# 
# ADD_LIBRARY(Utils
#   ImageUtils.cxx
#   FilterUtils.hpp
#   itkDisplacementFieldCompositionFilter.h
#   ChamferDistanceTransform.h
#   Log.h
#   SheetnessFilter.h
#   Preprocessing.h
#   TransformationUtils.h
#   SegmentationMapper.hxx
#   Metrics.h
#   TemporalMedianImageFilter.h
#   itkSubtractAbsImageFilter.h
#   ArgumentParser.h
# )
# TARGET_LINK_LIBRARIES(Utils ${ITK_LIBRARIES} ${Boost_LIBRARIES}  )
# 
