PROJECT(ImageRegistration)
include_directories (${SRS-MRF_SOURCE_DIR}/SegmentationRegistration) 
include_directories (${SRS-MRF_SOURCE_DIR}/Propagation) 
include_directories (${SRS-MRF_SOURCE_DIR}/Utils)
include_directories (${SRS-MRF_SOURCE_DIR}/External/fastPD)
include_directories (${SRS-MRF_SOURCE_DIR}/External/TRWS) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/RF) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/c-ugmix) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/maxFlow) 
include_directories (${SRS-MRF_SOURCE_DIR}/External/GCO) 

include_directories (${SRS-MRF_SOURCE_DIR}/Common) 

INCLUDE_DIRECTORIES(
   # /home/gasst/work/include
   # /usr/local/include
    /usr/include/libxml2
 ) 

#Tools
ADD_EXECUTABLE(SurfaceExtraction SurfaceExtraction.cxx )
TARGET_LINK_LIBRARIES(SurfaceExtraction  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(DeformImage3D DeformImage3D.cxx )
TARGET_LINK_LIBRARIES(DeformImage3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(DeformImage2D DeformImage2D.cxx )
TARGET_LINK_LIBRARIES(DeformImage2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(DeformImageAffine3D DeformImageAffine3D.cxx )
TARGET_LINK_LIBRARIES(DeformImageAffine3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(DeformLandmarksAffine3D DeformLandmarksAffine3D.cxx )
TARGET_LINK_LIBRARIES(DeformLandmarksAffine3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComposeDeformations3D ComposeDeformations3D.cxx )
TARGET_LINK_LIBRARIES(ComposeDeformations3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(DownsampleDeformations3D DownsampleDeformations3D.cxx )
TARGET_LINK_LIBRARIES(DownsampleDeformations3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComposeDeformations2D ComposeDeformations2D.cxx )
TARGET_LINK_LIBRARIES(ComposeDeformations2D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(AddDeformations2D AddDeformations2D.cxx )
TARGET_LINK_LIBRARIES(AddDeformations2D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SubtractDeformations2D SubtractDeformations2D.cxx )
TARGET_LINK_LIBRARIES(SubtractDeformations2D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(DefSetMetaParametersByReference2D DefSetMetaParametersByReference2D.cxx )
TARGET_LINK_LIBRARIES(DefSetMetaParametersByReference2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(TestDeformationResampling2D TestDeformationResampling2D.cxx )
TARGET_LINK_LIBRARIES(TestDeformationResampling2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(InvertComposeDeformations3D InvertComposeDeformations3d.cxx )
TARGET_LINK_LIBRARIES(InvertComposeDeformations3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(InvertDeformation3D InvertDeformation3D.cxx )
TARGET_LINK_LIBRARIES(InvertDeformation3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(DistanceTransform3D DistanceTransform3D.cxx )
TARGET_LINK_LIBRARIES(DistanceTransform3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(DiluteImage3D DiluteImage3D.cxx )
TARGET_LINK_LIBRARIES(DiluteImage3D  Utils   ${ITK_LIBRARIES}  )


ADD_EXECUTABLE(CreateSegmentationPriorFromAnnotatedImage3D CreateSegmentationPriorFromAnnotatedImage3D.cxx )
TARGET_LINK_LIBRARIES(CreateSegmentationPriorFromAnnotatedImage3D  Utils   ${ITK_LIBRARIES}  )


ADD_EXECUTABLE(convertPointToIndex3D convertPointToIndex3D.cxx )
TARGET_LINK_LIBRARIES(convertPointToIndex3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(CombineBinarySegmentations3D CombineBinarySegmentations3D.cxx )
TARGET_LINK_LIBRARIES(CombineBinarySegmentations3D  Utils   ${ITK_LIBRARIES} )

ADD_EXECUTABLE(FuseSegmentationMaxVote3D FuseSegmentationMaxVote3D.cxx )
TARGET_LINK_LIBRARIES(FuseSegmentationMaxVote3D  Utils   ${ITK_LIBRARIES} )

ADD_EXECUTABLE(FuseSegmentationWeightedMaxVote3D FuseSegmentationWeightedMaxVote3D.cxx )
TARGET_LINK_LIBRARIES(FuseSegmentationWeightedMaxVote3D  Utils   ${ITK_LIBRARIES} )

ADD_EXECUTABLE(CombineAffine3D CombineAffine3D.cxx )
TARGET_LINK_LIBRARIES(CombineAffine3D  Utils   ${ITK_LIBRARIES}  )


ADD_EXECUTABLE(InvertAffine3D InvertAffine3D.cxx )
TARGET_LINK_LIBRARIES(InvertAffine3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(AutoCropImageWithBorder3D AutoCropImageWithBorder3D.cxx )
TARGET_LINK_LIBRARIES(AutoCropImageWithBorder3D  Utils   ${ITK_LIBRARIES}  )

#Compare segmentations
ADD_EXECUTABLE(CompareSegmentations3D CompareSegmentations3D.cxx )
TARGET_LINK_LIBRARIES(CompareSegmentations3D Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(CompareSegmentations2D CompareSegmentations2D.cxx )
TARGET_LINK_LIBRARIES(CompareSegmentations2D Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SelectLabels3D SelectLabels3D.cxx )
TARGET_LINK_LIBRARIES(SelectLabels3D Utils   ${ITK_LIBRARIES}  )
#segment 
ADD_EXECUTABLE(SegmentProbImage3D SegmentProbImage3D.cxx )
TARGET_LINK_LIBRARIES(SegmentProbImage3D Utils SRSMRFModels    ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(SegmentProbImage2D SegmentProbImage2D.cxx )
TARGET_LINK_LIBRARIES(SegmentProbImage2D Utils SRSMRFModels    ${ITK_LIBRARIES}  )
 
#Compare registrations 
ADD_EXECUTABLE(ComputeLandmarkRegistrationError2D ComputeLandmarkRegistrationError.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationError2D Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ComputeLandmarkRegistrationError3D ComputeLandmarkRegistrationError3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationError3D Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ComputeLandmarkError3D ComputeLandmarkError3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkError3D Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(ComputeLandmarkRegistrationErrorTwoDef3D ComputeLandmarkRegistrationErrorTwoDef3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationErrorTwoDef3D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComputeDeformationError2D ComputeDeformationError2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationError2D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComputeDeformationErrorGroup2D ComputeDeformationErrorGroup2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationErrorGroup2D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComputeDeformationErrorGroup3D ComputeDeformationErrorGroup3D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationErrorGroup3D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(SubtractDeformations3D SubtractDeformations3D.cxx )
TARGET_LINK_LIBRARIES(SubtractDeformations3D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComputeDeformedGridImage2D ComputeDeformedGridImage2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformedGridImage2D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComputeDeformationStatistics3D ComputeDeformationStatistics3D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationStatistics3D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ComputeDeformationStatistics2D ComputeDeformationStatistics2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationStatistics2D Utils   ${ITK_LIBRARIES}  )
#Compare images

ADD_EXECUTABLE(CompareImages2D CompareImages2D.cxx )
TARGET_LINK_LIBRARIES(CompareImages2D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(CompareImagesAutoTransform2D CompareImagesAutoTransform2D.cxx )
TARGET_LINK_LIBRARIES(CompareImagesAutoTransform2D Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(GetImageMinMax3D GetImageMinMax3D.cxx )
TARGET_LINK_LIBRARIES(GetImageMinMax3D Utils   ${ITK_LIBRARIES}  )

#registration
ADD_EXECUTABLE(GradientDescentRegistration2D GradientDescentRegistration2D.cxx )
TARGET_LINK_LIBRARIES(GradientDescentRegistration2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(GradientDescentRegistrationITK GradientDescentITK.cxx )
TARGET_LINK_LIBRARIES(GradientDescentRegistrationITK  Utils   ${ITK_LIBRARIES}  )


#Image manipulations


ADD_EXECUTABLE(AddNoiseToImage3D AddNoiseToImage3D.cxx )
TARGET_LINK_LIBRARIES(AddNoiseToImage3D  Utils   ${ITK_LIBRARIES}  )
ADD_EXECUTABLE(AddNoiseToImage2D AddNoiseToImage2D.cxx )
TARGET_LINK_LIBRARIES(AddNoiseToImage2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(AverageImages3D AverageImages3D.cxx )
TARGET_LINK_LIBRARIES(AverageImages3D  Utils   ${ITK_LIBRARIES} )

ADD_EXECUTABLE(BinaryThresholdImage3D BinaryThresholdImage3D.cxx )
TARGET_LINK_LIBRARIES(BinaryThresholdImage3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ThresholdImage3D ThresholdImage3D.cxx )
TARGET_LINK_LIBRARIES(ThresholdImage3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(FillEmptySlice3D FillEmptySlice3D.cxx )
TARGET_LINK_LIBRARIES(FillEmptySlice3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ConvertRGBImageToITK3D ConvertRGBImageToITK3D.cxx )
TARGET_LINK_LIBRARIES(ConvertRGBImageToITK3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(CreateHumerusSegmentationMask CreateHumerusSegmentationMask.cxx )
TARGET_LINK_LIBRARIES(CreateHumerusSegmentationMask  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ResampleImage3D ResampleImage3D.cxx )
TARGET_LINK_LIBRARIES(ResampleImage3D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(ResampleImage2D ResampleImage2D.cxx )
TARGET_LINK_LIBRARIES(ResampleImage2D  Utils   ${ITK_LIBRARIES}  )

ADD_EXECUTABLE(CastShortToUchar2D CastShortToUchar2D.cxx )
TARGET_LINK_LIBRARIES(CastShortToUchar2D  Utils   ${ITK_LIBRARIES}  )
