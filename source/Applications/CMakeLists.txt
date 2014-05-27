PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")

include_directories (${SRS-MRF_SOURCE_DIR}/SegmentationRegistration) 
include_directories (${SRS-MRF_SOURCE_DIR}/Propagation) 

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

#Fusion
ADD_EXECUTABLE(SegFuse2D SegmentationFusion-2D.cxx )
TARGET_LINK_LIBRARIES(SegFuse2D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegFuse3D SegmentationFusion-3D.cxx )
TARGET_LINK_LIBRARIES(SegFuse3D ugmix maxflow  Utils TRWS  SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

#reg-propagation
INCLUDE_DIRECTORIES(
    /usr/include/libxml2
    ../External/RF
    ../External/c-ugmix
    ../Common
    /home/gasst/work/src/statismoREADONLY/build/include/statismo_ITK
    /home/gasst/work/src/statismoREADONLY/build/include/Representers/ITK/
    ${STATISMO_INCLUDE_DIR} ${STATISMO_INCLUDE_DIR}/statismo_ITK  ${STATISMO_INCLUDE_DIR}/3rdParty  ${STATISMO_INCLUDE_DIR}  
    ${STATISMO_INCLUDE_DIR}/Representers/ITK
 )

ADD_EXECUTABLE(RegProp2D RegistrationPropagation-2D.cxx )
TARGET_LINK_LIBRARIES(RegProp2D   Utils  SRSMRFModels  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )
ADD_EXECUTABLE(RegProp3D RegistrationPropagation-3D.cxx )
TARGET_LINK_LIBRARIES(RegProp3D   Utils  SRSMRFModels  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  ${STATISMO_LIBRARIES} )

ADD_EXECUTABLE(RegFuse2D RegistrationFusion2D.cxx )
TARGET_LINK_LIBRARIES(RegFuse2D   Utils  SRSMRFModels  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )
ADD_EXECUTABLE(RegFuse3D RegistrationFusion3D.cxx )
TARGET_LINK_LIBRARIES(RegFuse3D   Utils  SRSMRFModels  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )



#Aquirc like stuff
message( ${MATLAB_ENG_LIBRARY}  ${MATLAB_MX_LIBRARY})
include_directories(${MATLAB_INCLUDE_DIR})
ADD_EXECUTABLE(AquircGlobalNorm2D AquircGlobalNorm2D.cxx )
TARGET_LINK_LIBRARIES(AquircGlobalNorm2D   Utils    xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
ADD_EXECUTABLE(AquircLocalErrors2D AquircLocalError2D.cxx )
TARGET_LINK_LIBRARIES(AquircLocalErrors2D Utils    xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )

ADD_EXECUTABLE(CLERC2D CLERC2D.cxx )
TARGET_LINK_LIBRARIES(CLERC2D Utils    xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
ADD_EXECUTABLE(CLERC3D CLERC3D.cxx )
TARGET_LINK_LIBRARIES(CLERC3D Utils    xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )