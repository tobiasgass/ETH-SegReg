PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")

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

LINK_DIRECTORIES(
    #/home/gasst/work/lib
)
# Make sure the linker can find the Hello library once it is built. 
link_directories (${SRS-MRF_BINARY_DIR}/Models) 

ADD_EXECUTABLE(DeformImage3D DeformImage3D.cxx )
TARGET_LINK_LIBRARIES(DeformImage3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(DeformImage2D DeformImage2D.cxx )
TARGET_LINK_LIBRARIES(DeformImage2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SegReg Multiresolution-SegmentationRegistration.cxx )
TARGET_LINK_LIBRARIES(SegReg maxflow  Utils TRWS FastPD SRSMRFModels  ${ITK_LIBRARIES} RF xml2 /home/gasst/work/lib/libconfig++.a )
ADD_EXECUTABLE(SegReg2DClassifier Multiresolution-SegmentationRegistration2DClassifier.cxx )
TARGET_LINK_LIBRARIES(SegReg2DClassifier maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a )


#ADD_EXECUTABLE(SegReg-2D-subsampling Multiresolution-SegmentationRegistration-Subsampling-2D.cxx )
#TARGET_LINK_LIBRARIES(SegReg-2D-subsampling maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a )



ADD_EXECUTABLE(SegReg3D Multiresolution-SegmentationRegistration3D.cxx )
TARGET_LINK_LIBRARIES(SegReg3D maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a  )

ADD_EXECUTABLE(SRS3D Multiresolution-SRS3D.cxx )
TARGET_LINK_LIBRARIES(SRS3D ugmix maxflow  Utils TRWS FastPD SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SRS2D Multiresolution-SRS2D.cxx )
TARGET_LINK_LIBRARIES(SRS2D ugmix maxflow  Utils TRWS FastPD SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SRS3D-Bone SRS3D-Bone.cxx )
TARGET_LINK_LIBRARIES(SRS3D-Bone maxflow  Utils TRWS FastPD SRSMRFModels   RF  xml2 /home/gasst/work/lib/libconfig++.a  ${ITK_LIBRARIES}  )

#ADD_EXECUTABLE(AlternatingSegReg2DBone Alternating-JRS2D-Bone.cxx )
#TARGET_LINK_LIBRARIES(AlternatingSegReg2DBone maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )
#ADD_EXECUTABLE(AlternatingSegReg2DClassifier Alternating-JRS2D-Classifier.cxx )
#TARGET_LINK_LIBRARIES(AlternatingSegReg2DClassifier maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )
#ADD_EXECUTABLE(AlternatingSegReg3DBone Alternating-JRS3D-Bone.cxx )
#TARGET_LINK_LIBRARIES(AlternatingSegReg3DBone maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )

ADD_EXECUTABLE(AlternatingSegReg3D Alternating-JRS3D.cxx )
TARGET_LINK_LIBRARIES(AlternatingSegReg3D maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )
ADD_EXECUTABLE(AlternatingSegReg2D Alternating-JRS.cxx )
TARGET_LINK_LIBRARIES(AlternatingSegReg2D maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc )


#ADD_EXECUTABLE(Seg3D Segmentation3D.cxx )
#TARGET_LINK_LIBRARIES(Seg3D maxflow  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a tcmalloc)
ADD_EXECUTABLE(Seg2D Segmentation2D.cxx )
TARGET_LINK_LIBRARIES(Seg2D maxflow  Utils TRWS FastPD SRSMRFModels  ${ITK_LIBRARIES} RF xml2 /home/gasst/work/lib/libconfig++.a tcmalloc)
ADD_EXECUTABLE(Seg3D Segmentation3D.cxx )
TARGET_LINK_LIBRARIES(Seg3D maxflow  Utils TRWS FastPD SRSMRFModels  ${ITK_LIBRARIES} RF xml2 /home/gasst/work/lib/libconfig++.a tcmalloc)
#ADD_EXECUTABLE(memtest memtest.cxx )
#TARGET_LINK_LIBRARIES(memtest TRWS  /home/gasst/work/lib/libconfig++.a )

#ADD_EXECUTABLE(SegReg-Alt Multiresolution-AlternatingSegmentationRegistration.cxx )
#TARGET_LINK_LIBRARIES(SegReg-Alt  Utils TRWS FastPD SRSMRFModels   RF    xml2 /home/gasst/work/lib/libconfig++.a )

#ADD_EXECUTABLE(SegReg segmentation-registration.cxx )
#TARGET_LINK_LIBRARIES(SegReg  Utils RF TRWS FastPD SRSMRFModels      xml2  /home/gasst/work/lib/libconfig++.a  )

#ADD_EXECUTABLE(Segmentation segmentation.cxx )
#TARGET_LINK_LIBRARIES(Segmentation  Utils TRWS FastPD SRSMRFModels   )

#ADD_EXECUTABLE(fastPD Fast_PD.cpp )
#TARGET_LINK_LIBRARIES(fastPD  Utils TRWS FastPD SRSMRFModels  )
#ADD_EXECUTABLE(test test.cxx )
#TARGET_LINK_LIBRARIES(test  Utils RF TRWS FastPD SRSMRFModels     xml2  /home/gasst/work/lib/libconfig++.a  )
