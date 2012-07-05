PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")

include_directories (${SRS-MRF_SOURCE_DIR}/Models) 
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
ADD_EXECUTABLE(SRS2D Multiresolution-SRS2D.cxx )
TARGET_LINK_LIBRARIES(SRS2D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SRS3D-Bone SRS3D-Bone.cxx )
TARGET_LINK_LIBRARIES(SRS3D-Bone maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )


#ARS
#ADD_EXECUTABLE(AlternatingSegReg3D Alternating-JRS3D.cxx )
#TARGET_LINK_LIBRARIES(AlternatingSegReg3D maxflow  Utils TRWS  SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )
#ADD_EXECUTABLE(AlternatingSegReg2D Alternating-JRS.cxx )
#TARGET_LINK_LIBRARIES(AlternatingSegReg2D maxflow  Utils TRWS  SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )


#Segmentation
#ADD_EXECUTABLE(Seg2D Segmentation2D.cxx )
#TARGET_LINK_LIBRARIES(Seg2D maxflow  Utils TRWS  SRSMRFModels  ${ITK_LIBRARIES} RF xml2 /home/gasst/work/lib/libconfig++.a tcmalloc)
#ADD_EXECUTABLE(Seg3D Segmentation3D.cxx )
#TARGET_LINK_LIBRARIES(Seg3D maxflow  Utils TRWS  SRSMRFModels  ${ITK_LIBRARIES} RF xml2 /home/gasst/work/lib/libconfig++.a tcmalloc)

#seg-network
ADD_EXECUTABLE(SegNetwork2D SegmentationNetwork-2D.cxx )
TARGET_LINK_LIBRARIES(SegNetwork2D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegNetwork3D SegmentationNetwork-3D.cxx )
TARGET_LINK_LIBRARIES(SegNetwork3D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegNetwork2D-ConnectivityTest SegNetwork-Connectivity.cxx )
TARGET_LINK_LIBRARIES( SegNetwork2D-ConnectivityTest maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )


#seg-propagation
ADD_EXECUTABLE(SegProp2D SegmentationPropagation-2D.cxx )
TARGET_LINK_LIBRARIES(SegProp2D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegProp3D SegmentationPropagation-3D.cxx )
TARGET_LINK_LIBRARIES(SegProp3D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )