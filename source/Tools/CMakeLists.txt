PROJECT(ImageRegistration)



#Tools

ADD_EXECUTABLE(RegistrationAffineMultiRes3D RegistrationAffineMultiRes3D.cxx )
TARGET_LINK_LIBRARIES(RegistrationAffineMultiRes3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(CreateMultiLabelAtlasSegmentation3D CreateMultiLabelAtlasSegmentation3D.cxx )
TARGET_LINK_LIBRARIES(CreateMultiLabelAtlasSegmentation3D     ${ITK_LIBRARIES}   Utils  ) 
ADD_EXECUTABLE(CreateMultiLabelAtlasSegmentation2D CreateMultiLabelAtlasSegmentation2D.cxx )
TARGET_LINK_LIBRARIES(CreateMultiLabelAtlasSegmentation2D     ${ITK_LIBRARIES}   Utils  )


ADD_EXECUTABLE(PrintImageContent2D PrintImageContent2D.cxx )
TARGET_LINK_LIBRARIES(PrintImageContent2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(PrintImageContent3D PrintImageContent3D.cxx )
TARGET_LINK_LIBRARIES(PrintImageContent3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ConvertDeformation2D ConvertDeformation2D.cxx )
TARGET_LINK_LIBRARIES(ConvertDeformation2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(SurfaceExtraction SurfaceExtraction.cxx )
TARGET_LINK_LIBRARIES(SurfaceExtraction     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(DeformImage3D DeformImage3D.cxx )
TARGET_LINK_LIBRARIES(DeformImage3D     ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(DeformImage2D DeformImage2D.cxx )
TARGET_LINK_LIBRARIES(DeformImage2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(DeformImageAffine3D DeformImageAffine3D.cxx )
TARGET_LINK_LIBRARIES(DeformImageAffine3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(DeformLandmarksAffine3D DeformLandmarksAffine3D.cxx )
TARGET_LINK_LIBRARIES(DeformLandmarksAffine3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(DeformLandmarks3D DeformLandmarks3D.cxx )
TARGET_LINK_LIBRARIES(DeformLandmarks3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComposeDeformations3D ComposeDeformations3D.cxx )
TARGET_LINK_LIBRARIES(ComposeDeformations3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(DownsampleDeformations3D DownsampleDeformations3D.cxx )
TARGET_LINK_LIBRARIES(DownsampleDeformations3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComposeDeformations2D ComposeDeformations2D.cxx )
TARGET_LINK_LIBRARIES(ComposeDeformations2D     ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(AddDeformations2D AddDeformations2D.cxx )
TARGET_LINK_LIBRARIES(AddDeformations2D     ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(SubtractDeformations2D SubtractDeformations2D.cxx )
TARGET_LINK_LIBRARIES(SubtractDeformations2D     ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(DefSetMetaParametersByReference2D DefSetMetaParametersByReference2D.cxx )
TARGET_LINK_LIBRARIES(DefSetMetaParametersByReference2D     ${ITK_LIBRARIES}   Utils  )


ADD_EXECUTABLE(InvertComposeDeformations3D InvertComposeDeformations3d.cxx )
TARGET_LINK_LIBRARIES(InvertComposeDeformations3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(InvertDeformation3D InvertDeformation3D.cxx )
TARGET_LINK_LIBRARIES(InvertDeformation3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(InvertDeformation2D InvertDeformation2D.cxx )
TARGET_LINK_LIBRARIES(InvertDeformation2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(DistanceTransform3D DistanceTransform3D.cxx )
TARGET_LINK_LIBRARIES(DistanceTransform3D     ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(DilateImage3D DilateImage3D.cxx )
TARGET_LINK_LIBRARIES(DilateImage3D     ${ITK_LIBRARIES}   Utils  )





ADD_EXECUTABLE(convertPointToIndex3D convertPointToIndex3D.cxx )
TARGET_LINK_LIBRARIES(convertPointToIndex3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(CombineBinarySegmentations3D CombineBinarySegmentations3D.cxx )
TARGET_LINK_LIBRARIES(CombineBinarySegmentations3D     ${ITK_LIBRARIES}   Utils )

ADD_EXECUTABLE(AddSegmentationToImage3D AddSegmentationToImage3D.cxx )
TARGET_LINK_LIBRARIES(AddSegmentationToImage3D     ${ITK_LIBRARIES}   Utils )

ADD_EXECUTABLE(FuseSegmentationMaxVote3D FuseSegmentationMaxVote3D.cxx )
TARGET_LINK_LIBRARIES(FuseSegmentationMaxVote3D     ${ITK_LIBRARIES}   Utils )

ADD_EXECUTABLE(FuseSegmentationWeightedMaxVote3D FuseSegmentationWeightedMaxVote3D.cxx )
TARGET_LINK_LIBRARIES(FuseSegmentationWeightedMaxVote3D     ${ITK_LIBRARIES}   Utils )

ADD_EXECUTABLE(CombineAffine3D CombineAffine3D.cxx )
TARGET_LINK_LIBRARIES(CombineAffine3D     ${ITK_LIBRARIES}   Utils  )


ADD_EXECUTABLE(InvertAffine3D InvertAffine3D.cxx )
TARGET_LINK_LIBRARIES(InvertAffine3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(AutoCropImageWithBorder3D AutoCropImageWithBorder3D.cxx )
TARGET_LINK_LIBRARIES(AutoCropImageWithBorder3D     ${ITK_LIBRARIES}   Utils  )

#Compare segmentations
ADD_EXECUTABLE(CompareSegmentations3D CompareSegmentations3D.cxx )
TARGET_LINK_LIBRARIES(CompareSegmentations3D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(CompareSegmentations2D CompareSegmentations2D.cxx )
TARGET_LINK_LIBRARIES(CompareSegmentations2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(SelectLabels3D SelectLabels3D.cxx )
TARGET_LINK_LIBRARIES(SelectLabels3D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(RelabelSegmentation3D RelabelSegmentation3D.cxx )
TARGET_LINK_LIBRARIES(RelabelSegmentation3D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ExtractSegmentationOutline2D ExtractSegmentationOutline2D.cxx )
TARGET_LINK_LIBRARIES(ExtractSegmentationOutline2D    ${ITK_LIBRARIES}   Utils  )

#Compare registrations 
ADD_EXECUTABLE(ComputeLandmarkRegistrationError2D ComputeLandmarkRegistrationError.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationError2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeLandmarkRegistrationError3D ComputeLandmarkRegistrationError3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationError3D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeLandmarkError3D ComputeLandmarkError3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkError3D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeLandmarkRegistrationErrorTwoDef3D ComputeLandmarkRegistrationErrorTwoDef3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLandmarkRegistrationErrorTwoDef3D    ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComputeDeformationError2D ComputeDeformationError2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationError2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(CorrelateDeformationErrorMagnitude2D CorrelateDeformationErrorMagnitude2D.cxx )
TARGET_LINK_LIBRARIES(CorrelateDeformationErrorMagnitude2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeDeformationImprovement2D ComputeDeformationImprovement2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationImprovement2D    ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComputeDeformationErrorGroup2D ComputeDeformationErrorGroup2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationErrorGroup2D    ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComputeDeformationErrorGroup3D ComputeDeformationErrorGroup3D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationErrorGroup3D    ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComputeRegistrationErrorTriplet3D ComputeRegistrationErrorTriplet3D.cxx )
TARGET_LINK_LIBRARIES(ComputeRegistrationErrorTriplet3D    ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(SubtractDeformations3D SubtractDeformations3D.cxx )
TARGET_LINK_LIBRARIES(SubtractDeformations3D    ${ITK_LIBRARIES}   Utils  )



ADD_EXECUTABLE(ComputeDeformationStatistics3D ComputeDeformationStatistics3D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationStatistics3D    ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ComputeDeformationStatistics2D ComputeDeformationStatistics2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformationStatistics2D    ${ITK_LIBRARIES}   Utils  )
#Compare images

ADD_EXECUTABLE(CompareImages2D CompareImages2D.cxx )
TARGET_LINK_LIBRARIES(CompareImages2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeLocalSimilarity2D ComputeLocalSimilarity2D.cxx )
TARGET_LINK_LIBRARIES(ComputeLocalSimilarity2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeLocalSimilarity3D ComputeLocalSimilarity3D.cxx )
TARGET_LINK_LIBRARIES(ComputeLocalSimilarity3D    ${ITK_LIBRARIES}   Utils  )


ADD_EXECUTABLE(GetImageMinMax3D GetImageMinMax3D.cxx )
TARGET_LINK_LIBRARIES(GetImageMinMax3D    ${ITK_LIBRARIES}   Utils  )

#registration


ADD_EXECUTABLE(GradientDescentRegistrationITK GradientDescentITK.cxx )
TARGET_LINK_LIBRARIES(GradientDescentRegistrationITK     ${ITK_LIBRARIES}   Utils  )


#Image manipulations


ADD_EXECUTABLE(AddNoiseToImage3D AddNoiseToImage3D.cxx )
TARGET_LINK_LIBRARIES(AddNoiseToImage3D     ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(AddNoiseToImage2D AddNoiseToImage2D.cxx )
TARGET_LINK_LIBRARIES(AddNoiseToImage2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(AverageImages3D AverageImages3D.cxx )
TARGET_LINK_LIBRARIES(AverageImages3D     ${ITK_LIBRARIES}   Utils )

ADD_EXECUTABLE(BinaryThresholdImage3D BinaryThresholdImage3D.cxx )
TARGET_LINK_LIBRARIES(BinaryThresholdImage3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ThresholdImage3D ThresholdImage3D.cxx )
TARGET_LINK_LIBRARIES(ThresholdImage3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(FillEmptySlice3D FillEmptySlice3D.cxx )
TARGET_LINK_LIBRARIES(FillEmptySlice3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ConvertRGBImageToITK3D ConvertRGBImageToITK3D.cxx )
TARGET_LINK_LIBRARIES(ConvertRGBImageToITK3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(CreateHumerusSegmentationMask CreateHumerusSegmentationMask.cxx )
TARGET_LINK_LIBRARIES(CreateHumerusSegmentationMask     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(NormalizeImage3D NormalizeImage3D.cxx )
TARGET_LINK_LIBRARIES(NormalizeImage3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ResampleImage3D ResampleImage3D.cxx )
TARGET_LINK_LIBRARIES(ResampleImage3D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(ResampleImage2D ResampleImage2D.cxx )
TARGET_LINK_LIBRARIES(ResampleImage2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(FlipImageFilter FlipImageFilter.cxx )
TARGET_LINK_LIBRARIES(FlipImageFilter     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(CastShortToUchar2D CastShortToUchar2D.cxx )
TARGET_LINK_LIBRARIES(CastShortToUchar2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(AntialiasImage3D AntialiasImage3D.cxx )
TARGET_LINK_LIBRARIES(AntialiasImage3D Utils    ${ITK_LIBRARIES}   Utils    Utils )

ADD_EXECUTABLE(GetBoundingBox3D GetBoundingBox3D.cxx )
TARGET_LINK_LIBRARIES(GetBoundingBox3D     ${ITK_LIBRARIES}   Utils  )





ADD_EXECUTABLE(GenerateDeformation2D GenerateDeformation2D.cxx )
TARGET_LINK_LIBRARIES(GenerateDeformation2D      ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(GenerateDeformation3D GenerateDeformation3D.cxx )
TARGET_LINK_LIBRARIES(GenerateDeformation3D      ${ITK_LIBRARIES}
Utils  )

#tests

#tests



ADD_EXECUTABLE(CropImageToBoundingBox3D CropImageToBoundingBox3D.cxx )
TARGET_LINK_LIBRARIES(CropImageToBoundingBox3D      ${ITK_LIBRARIES}   Utils  )

###currently not compiling or broken
if ( 0 )

ADD_EXECUTABLE(CompareImagesAutoTransform2D CompareImagesAutoTransform2D.cxx )
TARGET_LINK_LIBRARIES(CompareImagesAutoTransform2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(ComputeDeformedGridImage2D ComputeDeformedGridImage2D.cxx )
TARGET_LINK_LIBRARIES(ComputeDeformedGridImage2D    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(CreateSegmentationPriorFromAnnotatedImage3D CreateSegmentationPriorFromAnnotatedImage3D.cxx )
TARGET_LINK_LIBRARIES(CreateSegmentationPriorFromAnnotatedImage3D     ${ITK_LIBRARIES}   Utils  )


ADD_EXECUTABLE(DeformationToBSplineTest2D DeformationToBSplineTest2D.cxx )
TARGET_LINK_LIBRARIES(DeformationToBSplineTest2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(GradientDescentRegistration2D GradientDescentRegistration2D.cxx )
TARGET_LINK_LIBRARIES(GradientDescentRegistration2D     ${ITK_LIBRARIES}   Utils  )

ADD_EXECUTABLE(PrintDeformationNorms2D PrintDeformationNorms2D.cxx )
TARGET_LINK_LIBRARIES(PrintDeformationNorms2D      ${ITK_LIBRARIES}   Utils  )

#segment 
ADD_EXECUTABLE(SegmentProbImage3D SegmentProbImage3D.cxx )
TARGET_LINK_LIBRARIES(SegmentProbImage3D  SRSMRFModels    ${ITK_LIBRARIES}   Utils  )
ADD_EXECUTABLE(SegmentProbImage2D SegmentProbImage2D.cxx )
TARGET_LINK_LIBRARIES(SegmentProbImage2D  SRSMRFModels    ${ITK_LIBRARIES}   Utils  )
 

ADD_EXECUTABLE(TestDeformationResampling2D TestDeformationResampling2D.cxx )
TARGET_LINK_LIBRARIES(TestDeformationResampling2D     ${ITK_LIBRARIES}   Utils  )

endif()
