PROJECT(SRS)
INCLUDE_REGULAR_EXPRESSION("^.*$")

include_directories (${SRS-MRF_SOURCE_DIR}/SegmentationRegistration) 

include_directories (${SRS-MRF_SOURCE_DIR}/Utils)
#include_directories (${SRS-MRF_SOURCE_DIR}/External/fastPD)
include_directories (${SRS-MRF_SOURCE_DIR}/External/TRWS) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/GCO) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/RF) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/c-ugmix) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/maxFlow) 

include_directories (${SRS-MRF_SOURCE_DIR}/Common) 
INCLUDE_DIRECTORIES(
   # /home/gasst/work/include
   # /usr/local/include
    /usr/include/libxml2
 ) 

LINK_DIRECTORIES(
    #/home/gasst/work/lib
)
# Make sure the linker can find the Hello library once it is built. 
link_directories (${SRS-MRF_BINARY_DIR}/Models) 



#SRS
ADD_EXECUTABLE(SRS3D Multiresolution-SRS3D.cxx )
TARGET_LINK_LIBRARIES(SRS3D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SRS3DMultilabel Multiresolution-SRS3D-Multilabel.cxx )
TARGET_LINK_LIBRARIES(SRS3DMultilabel ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SRS3DCategorical Multiresolution-SRS3D-Categorical.cxx )
TARGET_LINK_LIBRARIES(SRS3DCategorical ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SRS2DMultilabel Multiresolution-SRS2D-Multilabel.cxx )
TARGET_LINK_LIBRARIES(SRS2DMultilabel ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SRS2D Multiresolution-SRS2D.cxx )
TARGET_LINK_LIBRARIES(SRS2D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SRS3D-Bone SRS3D-Bone.cxx )
TARGET_LINK_LIBRARIES(SRS3D-Bone maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )


#ARS

ADD_EXECUTABLE(ARS2D Multiresolution-ARS2D.cxx )
TARGET_LINK_LIBRARIES(ARS2D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ARS3D Multiresolution-ARS3D.cxx )
TARGET_LINK_LIBRARIES(ARS3D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

