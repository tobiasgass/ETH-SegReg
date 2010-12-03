PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")

include_directories (${SRS-MRF_SOURCE_DIR}/Models) 
include_directories (${SRS-MRF_SOURCE_DIR}/Utils)
include_directories (${SRS-MRF_SOURCE_DIR}/External/fastPD) 
include_directories (${SRS-MRF_SOURCE_DIR}/Common) 

# Make sure the linker can find the Hello library once it is built. 
link_directories (${SRS-MRF_BINARY_DIR}/Models) 


ADD_EXECUTABLE(Registration registration.cxx )
TARGET_LINK_LIBRARIES(Registration ITKIO Utils FastPD SRSMRFModels  ITKNumerics ITKCommon)

ADD_EXECUTABLE(fastPD Fast_PD.cpp )
TARGET_LINK_LIBRARIES(fastPD ITKIO Utils FastPD SRSMRFModels  ITKNumerics)
