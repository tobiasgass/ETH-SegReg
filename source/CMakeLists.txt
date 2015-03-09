option( BUILD_SRS "Build SRS" ON )

if( ${BUILD_SRS} MATCHES "ON" )
  add_subdirectory( SimultaneousRegistrationSegmentation )
endif()

option( BUILD_CBRR "Build ConsistencyBasedRegistrationRectification" OFF )
if( ${BUILD_CBRR} MATCHES "ON" )
   add_subdirectory( ConsistencyBasedRegistrationRectification )
endif()

option( BUILD_SEMISUPERVISEDSEGMENTATIONPROPAGATION "Build  SemiSupervisedSegmentationPropagation" OFF )
if( ${BUILD_SEMISUPERVISEDSEGMENTATIONPROPAGATION} MATCHES "ON" )
   add_subdirectory( SemiSupervisedSegmentationPropagation )
endif()

option( BUILD_MARKOVRANDOMFIELDREGISTRATIONFUSION "Build  MarkovRandomFieldRegistrationFusion" OFF )
if( ${BUILD_MARKOVRANDOMFIELDREGISTRATIONFUSION} MATCHES "ON" )
   add_subdirectory( MarkovRandomFieldRegistrationFusion )
endif()

option( BUILD_TOOLS "Build General Tools and Utilities" OFF )
if( ${BUILD_TOOLS} MATCHES "ON" )
   add_subdirectory( Tools )
endif()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/gasst/work/src/Novel-SRS/source )


