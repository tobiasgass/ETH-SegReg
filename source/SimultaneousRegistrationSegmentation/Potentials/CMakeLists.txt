
set(POTFILES  ExplicitInstantiationsPotentials.cpp
  ExplicitInstantiationsPotentials.h)

if( ${USE_CUGMIX} MATCHES "ON" )
set(POTFILES ${POTFILES} Classifier-Segmentation-Unary-GMM.h)
endif()
if( ${USE_RF} MATCHES "ON" )
set(POTFILES ${POTFILES} Classifier-Segmentation-Unary-RandomForest.h)
endif()

add_library(SRSPotentials
  ${POTFILES})

target_link_libraries(SRSPotentials Utils  ${ITK_LIBRARIES})

#optional linking
if( ${USE_CUGMIX} MATCHES "ON" )
  target_link_libraries(SRSPotentials ugmix)
endif()

#optional linking
if( ${USE_RF} MATCHES "ON" )
  TARGET_LINK_LIBRARIES(SRSPotentials  RandomForest     )
endif()
