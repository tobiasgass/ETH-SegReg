find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

ADD_LIBRARY(Utils
  ImageUtils.cxx
  argstream.h
)
TARGET_LINK_LIBRARIES(Utils ITKCommon ITKIO)

