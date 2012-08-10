PROJECT(ImageRegistration)
include_directories (${SRS-MRF_SOURCE_DIR}/Models) 
include_directories (${SRS-MRF_SOURCE_DIR}/Utils)
include_directories (${SRS-MRF_SOURCE_DIR}/External/fastPD)
include_directories (${SRS-MRF_SOURCE_DIR}/External/TRWS) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/RF) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/c-ugmix) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/maxFlow) 

include_directories (${SRS-MRF_SOURCE_DIR}/Common) 

INCLUDE_DIRECTORIES(
   # /home/gasst/work/include
   # /usr/local/include
    /usr/include/libxml2
 ) 

#Tools
ADD_EXECUTABLE(DeformImage3D DeformImage3D.cxx )
TARGET_LINK_LIBRARIES(DeformImage3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(DeformImage2D DeformImage2D.cxx )
TARGET_LINK_LIBRARIES(DeformImage2D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ComposeDeformations3D ComposeDeformations3D.cxx )
TARGET_LINK_LIBRARIES(ComposeDeformations3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ComposeDeformations2D ComposeDeformations2D.cxx )
TARGET_LINK_LIBRARIES(ComposeDeformations2D  Utils   ${ITK_LIBRARIES}  )

#Compare segmentations
ADD_EXECUTABLE(CompareSegmentations3D CompareSegmentations3D.cxx )
TARGET_LINK_LIBRARIES(CompareSegmentations3D Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(CompareSegmentations2D CompareSegmentations2D.cxx )
TARGET_LINK_LIBRARIES(CompareSegmentations2D Utils   ${ITK_LIBRARIES}  )
 
#Compare registrations 
ADD_EXECUTABLE(ComputeLandmarkRegistrationError ComputeLandmarkRegistrationError.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationError Utils   ${ITK_LIBRARIES}  )