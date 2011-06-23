PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")

include_directories (${SRS-MRF_SOURCE_DIR}/Models) 
include_directories (${SRS-MRF_SOURCE_DIR}/Utils)
include_directories (${SRS-MRF_SOURCE_DIR}/External/fastPD)
include_directories (${SRS-MRF_SOURCE_DIR}/External/TRWS) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/RF) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/maxFlow) 

include_directories (${SRS-MRF_SOURCE_DIR}/Common) 
INCLUDE_DIRECTORIES(
    /home/gasst/work/include
    /usr/local/include
    /usr/include/libxml2
 ) 

LINK_DIRECTORIES(
    /home/gasst/work/lib
)
# Make sure the linker can find the Hello library once it is built. 
link_directories (${SRS-MRF_BINARY_DIR}/Models) 


#ADD_EXECUTABLE(Registration Multiresolution-Registration.cxx )
#TARGET_LINK_LIBRARIES(Registration ITKIO Utils TRWS FastPD SRSMRFModels  ITKBasicFilters RF ITKStatistics ITKNumerics ITKCommon xml2 /home/gasst/work/lib/libconfig++.so )
ADD_EXECUTABLE(SegReg Multiresolution-SegmentationRegistration.cxx )
TARGET_LINK_LIBRARIES(SegReg maxflow ITKIO Utils TRWS FastPD SRSMRFModels  ITKBasicFilters RF ITKStatistics ITKNumerics ITKCommon xml2 /home/gasst/work/lib/libconfig++.so )

ADD_EXECUTABLE(SegReg3D Multiresolution-SegmentationRegistration3D.cxx )
TARGET_LINK_LIBRARIES(SegReg3D maxflow ITKIO Utils TRWS FastPD SRSMRFModels  ITKBasicFilters RF ITKStatistics ITKNumerics ITKCommon xml2 /home/gasst/work/lib/libconfig++.so )

#ADD_EXECUTABLE(SegReg-Alt Multiresolution-AlternatingSegmentationRegistration.cxx )
#TARGET_LINK_LIBRARIES(SegReg-Alt ITKIO Utils TRWS FastPD SRSMRFModels  ITKBasicFilters RF ITKStatistics ITKNumerics ITKCommon xml2 /home/gasst/work/lib/libconfig++.so )

#ADD_EXECUTABLE(SegReg segmentation-registration.cxx )
#TARGET_LINK_LIBRARIES(SegReg ITKIO Utils RF TRWS FastPD SRSMRFModels  ITKBasicFilters ITKNumerics ITKCommon ITKStatistics xml2  /home/gasst/work/lib/libconfig++.so  )

#ADD_EXECUTABLE(Segmentation segmentation.cxx )
#TARGET_LINK_LIBRARIES(Segmentation ITKIO Utils TRWS FastPD SRSMRFModels  ITKNumerics ITKCommon)

#ADD_EXECUTABLE(fastPD Fast_PD.cpp )
#TARGET_LINK_LIBRARIES(fastPD ITKIO Utils TRWS FastPD SRSMRFModels  ITKNumerics)
#ADD_EXECUTABLE(test test.cxx )
#TARGET_LINK_LIBRARIES(test ITKIO Utils RF TRWS FastPD SRSMRFModels  ITKNumerics ITKCommon ITKStatistics xml2  /home/gasst/work/lib/libconfig++.so  )
