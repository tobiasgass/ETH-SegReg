include(${ITK_USE_FILE})



option( USE_GCO "Use GCO optimizer" ON )

if( ${USE_GCO} MATCHES "ON" )
  add_definitions(-DWITH_GCO)
  set(DIR_GCO "${SRS-MRF_SOURCE_DIR}/External/GCO" CACHE  FILEPATH "Directory for GCO")
  #  option( GCO_DIR "GCO directory" ${SRS-MRF_SOURCE_DIR}/External/GCO )
  include_directories( ${DIR_GCO} )
  ADD_LIBRARY(GCO
    ${DIR_GCO}/block.h  ${DIR_GCO}/energy.h   ${DIR_GCO}/GCoptimization.cpp  ${DIR_GCO}/GCoptimization.h  ${DIR_GCO}/graph.cpp  ${DIR_GCO}/graph.h  ${DIR_GCO}/LinkedBlockList.cpp  ${DIR_GCO}/LinkedBlockList.h  ${DIR_GCO}/maxflow.cpp
    ) 
endif()

#SUBDIRS(Applications)
add_subdirectory(Applications)

FILE(GLOB optHeaders "Optimizers/*.h")
FILE(GLOB graphHeaders "Graphs/*.h")
FILE(GLOB potHeaders "Potentials/*.h")
FILE(GLOB multHeaders "MultiResolution/*.h")


# INCLUDE_DIRECTORIES(
#     /usr/include/libxml2
#     ../External/RF
#     ../External/c-ugmix
#     ../Common
#  )


# SET(SRSMRFModels_SRC
#   NewClassifier.h
#   Classifier.h
#   BaseLabel.h
#   MRF-TRW-S.h
#   MRF-GCO.h
#   Graph.h
#   SubsamplingGraph.h
#   FastRegistrationGraph.h
#   SRSConfig.h
#   HierarchicalSRSImageToImageFilter.h
#   Potential-Registration-Pairwise.h
#   Potential-Registration-Unary.h
#   Potential-Segmentation-Unary.h
#   Potential-Coherence-Pairwise.h
#   SegmentationGraph.h
#   SegmentationImageFilter.h
#   dummy.cxx  
# )
# 

ADD_LIBRARY(SRSMRFModels ${optHeaders} ${graphHeaders} ${potHeaders} ${multHeaders} dummy.cxx)

#TARGET_LINK_LIBRARIES (SRSMRFModels RF ITKBasicFilters FastPD )

#IF (TCMALLOC)
#  TARGET_LINK_LIBRARIES (SRSMRFModels TRWS GCO ugmix RF   tcmalloc)
#ELSE(TCMALLOC)
#  TARGET_LINK_LIBRARIES (SRSMRFModels TRWS GCO ugmix RF   ) #tcmalloc
#ENDIF(TCMALLOC)
#INSTALL(TARGETS SRSMRFModels)
#TARGET_LINK_LIBRARIES(ITKAlgorithms ITKStatistics ITKNumerics)
