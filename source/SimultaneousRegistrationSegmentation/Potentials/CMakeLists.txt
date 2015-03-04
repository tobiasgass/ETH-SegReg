
add_library(SRSPotentials
  ExplicitInstantiationsPotentials.cpp
  ExplicitInstantiationsPotentials.h
  GMMClassifier.h)

target_link_libraries(SRSPotentials Utils  ${ITK_LIBRARIES})

#optional linking
if( ${USE_CUGMIX} MATCHES "ON" )
  target_link_libraries(SRSPotentials ugmix)
endif()