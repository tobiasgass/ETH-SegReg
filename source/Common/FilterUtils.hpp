/*=========================================================================

  Program: Automatic Segmentation Of Bones In 3D-CT Images

  Author:  Marcel Krcah <marcel.krcah@gmail.com>
  Computer Vision Laboratory
  ETH Zurich
  Switzerland

  Date:    2010-09-01

  Version: 1.0

  =========================================================================*/




#pragma once


#include "itkImage.h"
#include "itkImageDuplicator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkMaskNegatedImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkShiftScaleImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include "itkFastMarchingImageFilter.h"
#include "itkPasteImageFilter.h"
#include <itkResampleImageFilter.h>
#include "itkLinearInterpolateImageFunction.h"
#include "ImageUtils.cxx"
#include <algorithm> //max,min
#include "Log.h"
#include "itkStatisticsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNeighborhoodIterator.h"

using namespace std;

template<class InputImage, class OutputImage = InputImage>
class FilterUtils {

    typedef typename InputImage::Pointer  InputImagePointer;
    typedef typename InputImage::ConstPointer  ConstInputImagePointer;
    
    typedef typename InputImage::PixelType InputImagePixelType;
    typedef typename InputImage::IndexType InputImageIndex;
    typedef typename InputImage::RegionType InputImageRegion;

    typedef typename OutputImage::Pointer OutputImagePointer;
    typedef typename OutputImage::IndexType OutputImageIndex;
    typedef typename OutputImage::RegionType OutputImageRegion;
    typedef typename OutputImage::PixelType OutputImagePixelType;
    typedef typename OutputImage::PointType OutputImagePointType;

    typedef itk::FlatStructuringElement< InputImage::ImageDimension > StructuringElementType;
    typedef typename StructuringElementType::RadiusType StructuringElementTypeRadius;

    typedef itk::BinaryThresholdImageFilter<InputImage,OutputImage> BinaryThresholdFilter;
    typedef itk::BinaryErodeImageFilter<InputImage, OutputImage, StructuringElementType > ErodeFilterType;
    typedef itk::BinaryDilateImageFilter<InputImage, OutputImage, StructuringElementType > DilateFilterType;
    typedef itk::CastImageFilter <InputImage,OutputImage> CastImageFilterType;
    typedef itk::MaskImageFilter<InputImage,InputImage,OutputImage> MaskImageFilterType;
    typedef itk::MaskNegatedImageFilter<InputImage,InputImage,OutputImage> MaskNegatedImageFilterType;
    typedef itk::SubtractImageFilter <InputImage,InputImage,OutputImage> SubtractFilterType;
    typedef itk::AddImageFilter <InputImage,InputImage,OutputImage> AddFilterType;
    //    typedef itk::ConnectedComponentImageFilter<InputImage,OutputImage>  ConnectedComponentImageFilterType;
    typedef itk::RelabelComponentImageFilter<InputImage,OutputImage>  RelabelComponentImageFilterType;
    typedef itk::ShiftScaleImageFilter<InputImage,OutputImage>  ShiftScaleImageFilterType;
    typedef itk::DiscreteGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
    //typedef itk::SmoothingRecursiveGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
    typedef itk::FastMarchingImageFilter<OutputImage>  FastMarchingImageFilterType;
    //REMI:typedef itk::PasteImageFilter<InputImage,OutputImage>  PasteImageFilterType;

    typedef typename BinaryThresholdFilter::Pointer BinaryThresholdFilterPointer;
    typedef typename ErodeFilterType::Pointer ErodeFilterPointer;
    typedef typename DilateFilterType::Pointer DilateFilterPointer;
    typedef typename CastImageFilterType::Pointer CastFilterPointer;
    typedef typename MaskImageFilterType::Pointer MaskImageFilterPointer;
    typedef typename MaskNegatedImageFilterType::Pointer MaskNegatedImageFilterPointer;
    typedef typename SubtractFilterType::Pointer SubtractFilterPointer;
    typedef typename AddFilterType::Pointer AddFilterPointer;
    //    typedef typename ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
    typedef typename RelabelComponentImageFilterType::Pointer RelabelComponentImageFilterPointer;
    typedef typename ShiftScaleImageFilterType::Pointer ShiftScaleImageFilterPointer;
    typedef typename DiscreteGaussianImageFilterType::Pointer DiscreteGaussianImageFilterPointer;
    typedef typename FastMarchingImageFilterType::Pointer FastMarchingImageFilterPointer;
    //REMI:typedef typename PasteImageFilterType::Pointer PasteImageFilterPointer;

    typedef ImageUtils<InputImage> InputImageUtils;
    typedef ImageUtils<OutputImage> OutputImageUtils;

    typedef typename FastMarchingImageFilterType::NodeContainer FastMarchingNodeContainer;
    typedef typename FastMarchingNodeContainer::Pointer FastMarchingNodeContainerPointer;
    typedef typename FastMarchingImageFilterType::NodeType FastMarchingNodeType;

    typedef typename itk::LinearInterpolateImageFunction<InputImage, double> LinearInterpolatorType;
    typedef typename LinearInterpolatorType::Pointer LinearInterpolatorPointerType;
    typedef typename itk::NearestNeighborInterpolateImageFunction<InputImage, double> NNInterpolatorType;
    typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;
    typedef typename itk::ResampleImageFilter< InputImage , OutputImage>	ResampleFilterType;
    typedef typename ResampleFilterType::Pointer ResampleFilterPointerType;
    typedef typename itk::StatisticsImageFilter<InputImage > StatisticsFilterType;
    
    typedef typename itk::RescaleIntensityImageFilter<    InputImage, OutputImage >  RescaleFilterType;




public:
	static OutputImagePointer createEmpty(InputImagePointer refImg) {
		OutputImagePointer img=OutputImage::New();
		img->SetRegions(refImg->GetLargestPossibleRegion());
		img->SetOrigin(refImg->GetOrigin());
		img->SetSpacing(refImg->GetSpacing());
		img->SetDirection(refImg->GetDirection());
		img->Allocate();
		return img;
	};
    //#define  ISOTROPIC_RESAMPLING
    //resample with an uniform scaling factor

     static OutputImagePointer minimumResample( InputImagePointer input, InputImagePointer reference, double radius =-1.0) {
         OutputImagePointer result = createEmpty(reference);
         itk::ImageRegionIteratorWithIndex<OutputImage>  resultIt(result,result->GetLargestPossibleRegion());
         typedef typename itk::NeighborhoodIterator<InputImage> ImageNeighborhoodIteratorType;
         typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
         RadiusType r;
         if (radius>0.0)
             r.Fill(radius);
         else{
             //nyi
             r.Fill(8);
         }
         ImageNeighborhoodIteratorType inputIt(r,input,input->GetLargestPossibleRegion());
         for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt){
             InputImageIndex idx=resultIt.GetIndex();
             OutputImagePointType pt;
             result->TransformIndexToPhysicalPoint(idx,pt);
             input->TransformPhysicalPointToIndex(pt,idx);
             inputIt.SetLocation(idx);
             double minVal=std::numeric_limits<OutputImagePixelType>::max();
             for (int i=0;i<inputIt.Size();++i){
                 bool inside;
                 double value=inputIt.GetPixel(i,inside);
                 if (inside && value<minVal)
                     minVal=value;

             }
             resultIt.Set(minVal);
         }
         return result;
    }


#ifdef ISOTROPIC_RESAMPLING
  
    static OutputImagePointer LinearResample( ConstInputImagePointer input,  double scale, bool nnResample=false) {
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        NNInterpolatorPointerType interpolNN=NNInterpolatorType::New();

        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
          if (nnResample)
            resampler->SetInterpolator(interpolNN);
        else
            resampler->SetInterpolator(interpol);
        typename InputImage::SpacingType spacing,inputSpacing;
        typename InputImage::SizeType size,inputSize;
        typename InputImage::PointType origin,inputOrigin;
        inputOrigin=input->GetOrigin();
        inputSize=input->GetLargestPossibleRegion().GetSize();
        inputSpacing=input->GetSpacing();
        for (uint d=0;d<InputImage::ImageDimension;++d){
            size[d]=int(inputSize[d]*scale);
            spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
            origin[d]=inputOrigin[d];//+0.5*spacing[d]/inputSpacing[d];
        }
        resampler->SetOutputOrigin(origin);
		resampler->SetOutputSpacing ( spacing );
		resampler->SetOutputDirection ( input->GetDirection() );
		resampler->SetSize ( size );
        
        resampler->Update();
        return resampler->GetOutput();
    }


#else
    //downscale to isotropic spacing defined by minspacing/scale
    //never upsample!
    static OutputImagePointer LinearResample( ConstInputImagePointer input,  double scale, bool nnResample=false) {

        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        NNInterpolatorPointerType interpolNN=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
        if (nnResample)
            resampler->SetInterpolator(interpolNN);
        else
            resampler->SetInterpolator(interpol);
        typename InputImage::SpacingType spacing,inputSpacing;
        typename InputImage::SizeType size,inputSize;
        typename InputImage::PointType origin,inputOrigin;
        inputOrigin=input->GetOrigin();
        inputSize=input->GetLargestPossibleRegion().GetSize();
        inputSpacing=input->GetSpacing();
        double minSpacing=std::numeric_limits<double>::max();
        for (uint d=0;d<InputImage::ImageDimension;++d){
            if (inputSpacing[d]<minSpacing) minSpacing=inputSpacing[d];
        }
        double newSpacing=minSpacing/scale;
        LOGV(7)<<"new isotrpoic spacing : "<<newSpacing<<endl;
        for (uint d=0;d<InputImage::ImageDimension;++d){
            //determine new spacing
            //never increase resolution!
            spacing[d]=max(inputSpacing[d],newSpacing);//inputSpacing[d]*(1.0*inputSize[d]/size[d]);
            //calculate new image size
            size[d]=int(inputSpacing[d]/spacing[d] * (inputSize[d]-1))+1;
            //finalize spacing as a function of the new size
            spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
            //size[d]=int(inputSpacing[d]/spacing[d]*(inputSize[d]));

            origin[d]=inputOrigin[d];//+0.5*spacing[d]/inputSpacing[d];
        }
        LOGV(7)<<"full parameters : "<<spacing<<" "<<size<<" "<<origin<<endl;
        LOGV(3)<<"Resampling to isotropic  spacing "<<spacing<<" with resolution "<<size<<endl;
        resampler->SetOutputOrigin(origin);
		resampler->SetOutputSpacing ( spacing );
		resampler->SetOutputDirection ( input->GetDirection() );
		resampler->SetSize ( size );
        resampler->Update();
        return resampler->GetOutput();
    }
  
#endif
    static OutputImagePointer LinearResample( InputImagePointer input,  double scale, bool nnResample=false) {
        return LinearResample(ConstInputImagePointer(input),scale,nnResample);
    }

    static OutputImagePointer NNResample( ConstInputImagePointer input,  double scale) {
        return LinearResample(input,scale,true);
    }
    static OutputImagePointer NNResample( InputImagePointer input,  double scale) {
        return LinearResample((ConstInputImagePointer)input,scale,true);
    }
    static OutputImagePointer NNResample( ConstInputImagePointer input,  ConstInputImagePointer reference) {
        NNInterpolatorPointerType interpol=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
        resampler->SetInterpolator(interpol);
    
        resampler->SetOutputOrigin(reference->GetOrigin());
		resampler->SetOutputSpacing ( reference->GetSpacing() );
		resampler->SetOutputDirection ( reference->GetDirection() );
		resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }
    static OutputImagePointer LinearResample( ConstInputImagePointer input,  ConstInputImagePointer reference) {
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
        resampler->SetInterpolator(interpol);
        resampler->SetOutputOrigin(reference->GetOrigin());
		resampler->SetOutputSpacing ( reference->GetSpacing() );
		resampler->SetOutputDirection ( reference->GetDirection() );
		resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }
    static OutputImagePointer LinearResample( InputImagePointer input,  InputImagePointer reference) {
        return LinearResample((ConstInputImagePointer)input,(ConstInputImagePointer)reference);
    }
    static OutputImagePointer NNResample( InputImagePointer input,  InputImagePointer reference) {
        return NNResample((ConstInputImagePointer)input,(ConstInputImagePointer)reference);
    }

    static OutputImagePointer NNResample( InputImagePointer input,  ConstInputImagePointer reference) {
        NNInterpolatorPointerType interpol=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
        resampler->SetInterpolator(interpol);
    
        resampler->SetOutputOrigin(reference->GetOrigin());
		resampler->SetOutputSpacing ( reference->GetSpacing() );
		resampler->SetOutputDirection ( reference->GetDirection() );
		resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }

  


    /**
       Paste the region from the source image to a given position
       in the destination image
    */
    static OutputImagePointer paste(
                                    InputImagePointer sourceImage, InputImageRegion sourceRegion,
                                    OutputImagePointer destinationImage, OutputImageIndex destinationIndex
                                    ) {
		typedef itk::PasteImageFilter<InputImage,OutputImage>  PasteImageFilterType;//REMI
		typedef typename PasteImageFilterType::Pointer PasteImageFilterPointer;//REMI

        PasteImageFilterPointer filter =
            PasteImageFilterType::New();

        filter->SetSourceImage(sourceImage);
        filter->SetSourceRegion(sourceRegion);
        filter->SetDestinationImage(destinationImage);
        filter->SetDestinationIndex(destinationIndex);

        filter->Update();

        return filter->GetOutput();
    }

	static OutputImagePointer createEmpty(ConstInputImagePointer refImg) {
		OutputImagePointer img=OutputImage::New();
		img->SetRegions(refImg->GetLargestPossibleRegion());
		img->SetOrigin(refImg->GetOrigin());
		img->SetSpacing(refImg->GetSpacing());
		img->SetDirection(refImg->GetDirection());
		img->Allocate();
		return img;
	};
    




    // output_pixel =  scale * input_pixel + shift
    static OutputImagePointer linearTransform(
                                              InputImagePointer image,
                                              OutputImagePixelType scale,
                                              OutputImagePixelType shift = 0
                                              ) {

        ShiftScaleImageFilterPointer filter =
            ShiftScaleImageFilterType::New();

        filter->SetInput(image);
        filter->SetScale(scale);
        filter->SetShift(shift);
        filter->Update();
        return filter->GetOutput();
    }


    // smooth the image with a discrete gaussian filter
    static OutputImagePointer gaussian(
                                       ConstInputImagePointer image, float variance, bool spacing=true
                                       ) {

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();

        filter->SetInput(image);
        //filter->SetSigma(variance);
        filter->SetVariance(variance);
        if (!spacing)
            filter->SetUseImageSpacingOff();
        filter->Update();

        return filter->GetOutput();
    }

    static OutputImagePointer gaussian(InputImagePointer image, float variance, bool spacing=true
                                       ) {
        return gaussian(ConstInputImagePointer(image),variance,spacing);
    }



    // relabel components according to its size.
    // Largest component 1, second largest 2, ...
    static OutputImagePointer relabelComponents(InputImagePointer image) {

        RelabelComponentImageFilterPointer filter =
            RelabelComponentImageFilterType::New();

        filter->SetInput(image);
        filter->Update();

        return filter->GetOutput();
    }

    static OutputImagePointer cast(ConstInputImagePointer image) {
        CastFilterPointer castFilter = CastImageFilterType::New();
        castFilter->SetInput(image);
        castFilter->Update();
        return castFilter->GetOutput();
    }

    // cast the image to the output type
    static OutputImagePointer cast(InputImagePointer image) {
        CastFilterPointer castFilter = CastImageFilterType::New();
        castFilter->SetInput(image);
        castFilter->Update();
        return castFilter->GetOutput();
    }
    
    static OutputImagePointer truncateCast(InputImagePointer image) {
        InputImagePointer trunc=FilterUtils<InputImage,InputImage>::thresholding(image,std::numeric_limits<OutputImagePixelType>::min(), std::numeric_limits<OutputImagePixelType>::max());
        CastFilterPointer castFilter = CastImageFilterType::New();
        castFilter->SetInput(image);
        castFilter->Update();
        return castFilter->GetOutput();
    }




    static OutputImagePointer createEmptyFrom(
                                              InputImagePointer input
                                              ) {

        OutputImagePointer output = OutputImageUtils::createEmpty(
                                                                  input->GetLargestPossibleRegion().GetSize());
        output->SetOrigin(input->GetOrigin());
        output->SetSpacing(input->GetSpacing());
        output->SetDirection(input->GetDirection());
        output->FillBuffer(0);

        return output;

    }



    static OutputImagePointer substract(
                                        InputImagePointer image1, InputImagePointer image2
                                        ) {
        SubtractFilterPointer substractFilter = SubtractFilterType::New();
        substractFilter->SetInput1(image1);
        substractFilter->SetInput2(image2);
        //substractFilter->SetCoordinateTolerance(1e-5);
        substractFilter->Update();
        return substractFilter->GetOutput();
    }





    // pixel-wise addition of two images
    static OutputImagePointer add(
                                  InputImagePointer image1, InputImagePointer image2
                                  ) {
        AddFilterPointer addFilter = AddFilterType::New();
        addFilter->SetInput1(image1);
        addFilter->SetInput2(image2);
        addFilter->InPlaceOff();
        addFilter->Update();
        return addFilter->GetOutput();
    }




    static OutputImagePointer mask(
                                   InputImagePointer image,
                                   InputImagePointer mask
                                   ) {
        MaskImageFilterPointer filter = MaskImageFilterType::New();

        filter->SetInput1(image);
        filter->SetInput2(mask);
        filter->Update();

        return filter->GetOutput();
    }





    static OutputImagePointer negatedMask(
                                          InputImagePointer image,
                                          InputImagePointer mask
                                          ) {
        MaskNegatedImageFilterPointer filter = MaskNegatedImageFilterType::New();

        filter->SetInput1(image);
        filter->SetInput2(mask);
        filter->Update();

        return filter->GetOutput();
    }





    static OutputImagePointer binaryThresholding(
                                                 InputImagePointer inputImage,
                                                 InputImagePixelType lowerThreshold,
                                                 InputImagePixelType upperThreshold,
                                                 OutputImagePixelType insideValue = 1,
                                                 OutputImagePixelType outsideValue = 0
                                                 ) {
        BinaryThresholdFilterPointer thresholder = BinaryThresholdFilter::New();
        thresholder->SetInput(inputImage);

        thresholder->SetLowerThreshold( lowerThreshold );
        thresholder->SetUpperThreshold( upperThreshold );
        thresholder->SetInsideValue(insideValue);
        thresholder->SetOutsideValue(outsideValue);

        thresholder->Update();

        return thresholder->GetOutput();
    }

      static OutputImagePointer binaryThresholdingLow(
                                                 InputImagePointer inputImage,
                                                 InputImagePixelType lowerThreshold,
                                                 OutputImagePixelType insideValue = 1,
                                                 OutputImagePixelType outsideValue = 0
                                                 ) {
        BinaryThresholdFilterPointer thresholder = BinaryThresholdFilter::New();
        thresholder->SetInput(inputImage);

        thresholder->SetLowerThreshold( lowerThreshold );
        thresholder->SetInsideValue(insideValue);
        thresholder->SetOutsideValue(outsideValue);

        thresholder->Update();

        return thresholder->GetOutput();
    }
      static OutputImagePointer binaryThresholdingHigh(
                                                 InputImagePointer inputImage,
                                                 InputImagePixelType upperThreshold,
                                                 OutputImagePixelType insideValue = 1,
                                                 OutputImagePixelType outsideValue = 0
                                                 ) {
        BinaryThresholdFilterPointer thresholder = BinaryThresholdFilter::New();
        thresholder->SetInput(inputImage);
        thresholder->SetUpperThreshold( upperThreshold );
        thresholder->SetInsideValue(insideValue);
        thresholder->SetOutsideValue(outsideValue);

        thresholder->Update();

        return thresholder->GetOutput();
    }
    static OutputImagePointer round(
                                    InputImagePointer inputImage
                                    ) {
        OutputImagePointer outputImage=createEmpty(ConstInputImagePointer(inputImage));
        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        itk::ImageRegionIterator<InputImage> it2(
                                                 outputImage, outputImage->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it,++it2) {
            double val=it.Get();
            it2.Set(floor(val+0.5));
        }
        return outputImage;
    }




    // assign pixels with intensities above upperThreshold
    // value upperThreshold and below lowerThreshold value
    // lowerThreshold
    static OutputImagePointer thresholding(
                                           InputImagePointer inputImage,
                                           InputImagePixelType lowerThreshold,
                                           InputImagePixelType upperThreshold
                                           ) {

        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            it.Set(
                   std::min(
                            upperThreshold,
                            std::max(it.Get(),lowerThreshold)
                            )
                   );
        }

        return cast(inputImage);
    }
 // assign pixels with intensities above upperThreshold
    // value upperThreshold and below lowerThreshold value
    // lowerThreshold
    static OutputImagePointer lowerThresholding(
                                           InputImagePointer inputImage,
                                           InputImagePixelType lowerThreshold
                                           ) {

        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            it.Set(
                   std::max(it.Get(),lowerThreshold)
                   );
        }

        return cast(inputImage);
    }




    // perform erosion (mathematical morphology) with a given label image
    // using a ball with a given radius
    static OutputImagePointer erosion(
                                      InputImagePointer labelImage, unsigned radius,
                                      InputImagePixelType valueToErode = 1
                                      ) {
        StructuringElementTypeRadius rad; rad.Fill(radius);
        StructuringElementType K = StructuringElementType::Ball( rad );

        ErodeFilterPointer erosionFilter = ErodeFilterType::New();
        erosionFilter->SetKernel(K);
        erosionFilter->SetErodeValue(valueToErode);

        erosionFilter->SetInput( labelImage );
        erosionFilter->Update();
        return erosionFilter->GetOutput();

    }




    // perform erosion (mathematical morphology) with a given label image
    // using a ball with a given radius
    static OutputImagePointer dilation(
                                       InputImagePointer labelImage, unsigned radius,
                                       InputImagePixelType valueToDilate = 1
                                       ) {
        StructuringElementTypeRadius rad; rad.Fill(radius);
        StructuringElementType K = StructuringElementType::Ball( rad );

        DilateFilterPointer dilateFilter  = DilateFilterType::New();
        dilateFilter->SetKernel(K);
        dilateFilter->SetDilateValue(valueToDilate);

        dilateFilter->SetInput( labelImage );
        dilateFilter->Update();

        return dilateFilter->GetOutput();
    }


#if 0
    // compute connected components of a (binary image)
    static OutputImagePointer connectedComponents(InputImagePointer image) {

        ConnectedComponentImageFilterPointer filter =
            ConnectedComponentImageFilterType::New();

        filter->SetInput(image);
        filter->Update();

        return filter->GetOutput();
    }
#endif



    /**
       Compute distance from the object using fast marching front.

       All pixels with value objectLabel are considered to belong to an object.
       If positive stopping value is specified, the maximum computed distance
       will be stoppingValue, otherwise the distance will be computed for the
       whole image. The image spacing is ignored, 1 is used for all directions.
    */
    static OutputImagePointer distanceMapByFastMarcher(
                                                       InputImagePointer image,
                                                       InputImagePixelType objectLabel,
                                                       float stoppingValue = 0
                                                       ) {

        // prepare fast marching
        FastMarchingImageFilterPointer fastMarcher =
            FastMarchingImageFilterType::New();
        fastMarcher->SetOutputSize(image->GetLargestPossibleRegion().GetSize());
        fastMarcher->SetOutputOrigin(image->GetOrigin() );
        fastMarcher->SetOutputSpacing(image->GetSpacing() );
        fastMarcher->SetOutputDirection(image->GetDirection() );
        fastMarcher->SetSpeedConstant(1.0);
        if (stoppingValue > 0)
            fastMarcher->SetStoppingValue(stoppingValue);

        // set seeds as pixels in the island @label
        FastMarchingNodeContainerPointer seeds = FastMarchingNodeContainer::New();
        seeds->Initialize();

        itk::ImageRegionIteratorWithIndex<InputImage> it(
                                                         image, image->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            if (it.Get() == objectLabel) {
                FastMarchingNodeType & node = seeds->CreateElementAt(seeds->Size());
                node.SetValue(0);
                node.SetIndex(it.GetIndex());
            }
        }
        fastMarcher->SetTrialPoints(seeds);

        // perform fast marching
        fastMarcher->Update();

        // done :)
        return  fastMarcher->GetOutput();

    }


    static OutputImagePointer computeObjectness(InputImagePointer img){
        return img;
    }

    static OutputImagePointer computeMultilabelSegmentation(InputImagePointer img){
        return img;
    }

    
    static InputImagePixelType getMax(InputImagePointer img){
        return getMax(ConstInputImagePointer(img));
    }
    static InputImagePixelType getMax(ConstInputImagePointer img){
        typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
        filter->SetInput(img);
        filter->Update();
        return filter->GetMaximumOutput()->Get();

    }

    static OutputImagePointer normalize(InputImagePointer img){
        typename RescaleFilterType::Pointer    rescaleFilter    = RescaleFilterType::New();
        rescaleFilter->SetInput(img);
        rescaleFilter->SetOutputMinimum(std::numeric_limits<OutputImagePixelType>::min());
        rescaleFilter->SetOutputMaximum(std::numeric_limits<OutputImagePixelType>::max());
        rescaleFilter->Update();
        return rescaleFilter->GetOutput();
    }

     //hellishly inefficient but "clean" implementation of local normalized cross correlation
    static inline OutputImagePointer LNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
        return LNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp);
    }
    static inline OutputImagePointer LNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        //typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        //filter->SetSigma(sigma);
        filter->SetVariance(sigma*sigma);
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Bar.mhd",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        OutputImagePointer i2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput());  
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Bar.mhd",i2Bar));
        //HACK!
        //compute squares of original images
        OutputImagePointer i1Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        OutputImagePointer i2Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Square.mhd",i2Square));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Square.mhd",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        OutputImagePointer i1SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1SquareBar.mhd",i1SquareBar));
        filter->SetInput(i2Square);
        filter->Update();
        OutputImagePointer i2SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2SquareBar.mhd",i2SquareBar));

        //compute squares of local means
        OutputImagePointer i1BarSquare=ImageUtils<OutputImage>::localSquare(i1Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarSquare.mhd",i1BarSquare));
        OutputImagePointer i2BarSquare=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2BarSquare.mhd",i2BarSquare));

        //multiply i1 and i2 locally
        OutputImagePointer i1i2=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1i2.mhd",i1i2));
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        OutputImagePointer i1Timesi2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Timesi2Bar.mhd",i1Timesi2Bar));

        //multiply local means
        OutputImagePointer i1BarTimesi2Bar=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Bar,i2Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarTimesi2Bar.mhd",i1BarTimesi2Bar));


        
        OutputImagePointer numerator=FilterUtils<OutputImage>::substract(i1Timesi2Bar,i1BarTimesi2Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("numerator.mhd",numerator));

        OutputImagePointer varianceI1=FilterUtils<OutputImage>::substract(i1SquareBar,i1BarSquare); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(varianceI1,std::numeric_limits<OutputImagePixelType>::epsilon());
        LOGI(8,ImageUtils<OutputImage>::writeImage("varianceI1.mhd",varianceI1));
       
        OutputImagePointer varianceI2=FilterUtils<OutputImage>::substract(i2SquareBar,i2BarSquare);
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(varianceI2,std::numeric_limits<OutputImagePixelType>::epsilon());

        LOGI(8,ImageUtils<OutputImage>::writeImage("varianceI2.mhd",varianceI2));
        
        ImageUtils<OutputImage>::sqrtImage(varianceI1);
        ImageUtils<OutputImage>::sqrtImage(varianceI2);

        OutputImagePointer denominator=ImageUtils<OutputImage>::multiplyImageOutOfPlace(varianceI1,varianceI2);

        OutputImagePointer result=ImageUtils<OutputImage>::divideImageOutOfPlace(numerator,denominator);
        //convert to weights
        
        itk::ImageRegionIterator<OutputImage> it(result,result->GetLargestPossibleRegion());
        it.GoToBegin();
        for (;!it.IsAtEnd();++it){
            OutputImagePixelType r = it.Get();
            if (r< -(OutputImagePixelType)1.0 ){
                r = 0;
            }else if ( r > (OutputImagePixelType)1.0){
                r = 0;
            }
            r = pow((r+1.0)/2,exp);
            
            r = max(r,std::numeric_limits<OutputImagePixelType>::epsilon());
            it.Set(r);
            
        }
        // LOGI(8,ImageUtils<OutputImage>::add(result,1.0));
        //        ImageUtils<OutputImage>::multiplyImage(result,0.5);
        //        FilterUtils<OutputImage,OutputImage>::thresholding(result,std::numeric_limits<OutputImagePixelType>::epsilon(),1.0);

        LOGI(10,ImageUtils<OutputImage>::writeImage("result.mhd",result));

        return result;

    }
   
    static inline OutputImagePointer LSSDNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetVariance(sigmaWidth*sigmaWidth);

        OutputImagePointer diff = FilterUtils<OutputImage>::substract(i1Cast,i2Cast);
        OutputImagePointer diffSquare = ImageUtils<OutputImage>::localSquare(diff);

        //compute local means by concolving with gaussian
        filter->SetInput(diffSquare);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
       
        OutputImagePointer result = ImageUtils<OutputImage>::createEmpty(i1Bar);

        typename ImageUtils<OutputImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType diffIt(i1Bar,i1Bar->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), diffIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++diffIt){
            double d = diffIt.Get();
            if (d<0.0){
                LOG<<VAR(d)<<endl;
            }
            resultIt.Set(max(0.001,exp( - 0.5 *  d / sigmaNorm) ));
        }
        
        
        return result;
    }

    static inline OutputImagePointer LSSD(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0){
        return LSSD( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
    static inline OutputImagePointer LSSD(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);

        OutputImagePointer diff = FilterUtils<OutputImage>::substract(i1Cast,i2Cast);
        OutputImagePointer diffSquare = ImageUtils<OutputImage>::localSquare(diff);

        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        OutputImagePointer i1Bar=(filter->GetOutput()); 
        
        
        return i1Bar;
    }

    static inline OutputImagePointer LSAD(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0){
        return LSAD( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
    static inline OutputImagePointer LSAD(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);

        OutputImagePointer diff = FilterUtils<OutputImage>::substract(i1Cast,i2Cast);

        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
       
        OutputImagePointer result = ImageUtils<OutputImage>::createEmpty(i1Bar);

        typename ImageUtils<OutputImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType diffIt(i1Bar,i1Bar->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), diffIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++diffIt){
            double d = diffIt.Get();
            resultIt.Set(fabs(d));
        }
        
        
        return result;
    }
 static inline OutputImagePointer LSADNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);

        OutputImagePointer diff = FilterUtils<OutputImage>::substract(i1Cast,i2Cast);

        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
       
        OutputImagePointer result = ImageUtils<OutputImage>::createEmpty(i1Bar);

        typename ImageUtils<OutputImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType diffIt(i1Bar,i1Bar->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), diffIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++diffIt){
            double d = diffIt.Get();
            resultIt.Set(max(0.5,exp(-0.5 * fabs(d)  / sigmaNorm) ));
        }
        
        
        return result;
    }
   static inline OutputImagePointer efficientLNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
           return efficientLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp);
   }
   static inline OutputImagePointer efficientLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigma);
        //filter->SetVariance(sigma*sigma);
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Bar.mhd",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        OutputImagePointer i2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput());  

        //FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Bar.mhd",i2Bar));
        //HACK!
        //compute squares of original images


        
        OutputImagePointer i1Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        OutputImagePointer i2Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Square.mhd",i2Square));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Square.mhd",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        OutputImagePointer i1SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1SquareBar.mhd",i1SquareBar));
       
        filter->SetInput(i2Square);
        filter->Update();
        OutputImagePointer i2SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2SquareBar.mhd",i2SquareBar));

        //compute squares of local means
        OutputImagePointer i1BarSquare=ImageUtils<OutputImage>::localSquare(i1Bar);
        OutputImagePointer i2BarSquare=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        //multiply i1 and i2 locally
        OutputImagePointer i1i2=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1i2.mhd",i1i2));

        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarSquare.mhd",i1BarSquare));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2BarSquare.mhd",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        OutputImagePointer i1Timesi2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Timesi2Bar.mhd",i1Timesi2Bar));

        //finish
        OutputImagePointer result=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        itk::ImageRegionIterator<OutputImage> resultIt(result,result->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        
        itk::ImageRegionIterator<OutputImage> i1BarIt(i1Bar,i1Bar->GetLargestPossibleRegion());
        i1BarIt.GoToBegin();
        itk::ImageRegionIterator<OutputImage> i2BarIt(i2Bar,i2Bar->GetLargestPossibleRegion());
        i2BarIt.GoToBegin();

        itk::ImageRegionIterator<OutputImage> i1Timesi2BarIt(i1Timesi2Bar,i1Timesi2Bar->GetLargestPossibleRegion());
        i1Timesi2BarIt.GoToBegin();

        itk::ImageRegionIterator<OutputImage> i1SquareBarIt(i1SquareBar,i1SquareBar->GetLargestPossibleRegion());
        i1SquareBarIt.GoToBegin();

        itk::ImageRegionIterator<OutputImage> i2SquareBarIt(i2SquareBar,i2SquareBar->GetLargestPossibleRegion());
        i2SquareBarIt.GoToBegin();

        
        
        for (;!resultIt.IsAtEnd();++resultIt){

            
            double i1BarV=i1BarIt.Get();
            double i2BarV=i2BarIt.Get();
            double i1BarTimesi2BarV=i1BarV*i2BarV;
         
            
            double i1Timesi2BarV=i1Timesi2BarIt.Get();
            double numeratorV=i1Timesi2BarV-i1BarTimesi2BarV;

            double varianceI1V=sqrt(max(0.0,i1SquareBarIt.Get() - i1BarV*i1BarV));
            double varianceI2V=sqrt(max(0.0,i2SquareBarIt.Get() - i2BarV*i2BarV));
            double denominatorV=varianceI1V*varianceI2V;

            double r = numeratorV/(denominatorV+0.00000000001);

            ++i1BarIt;
            ++i2BarIt;
            ++i1Timesi2BarIt;
            ++i1SquareBarIt;
            ++i2SquareBarIt;


            if (r< -(OutputImagePixelType)1.0 ){
                r = 0;
            }else if ( r > (OutputImagePixelType)1.0){
                r = 0;
            }
            r = pow((r+1.0)/2,exp);
            
            r = max(r,std::numeric_limits<double>::epsilon());

            resultIt.Set(r);

        }
        
        LOGI(8,ImageUtils<OutputImage>::writeImage("result.mhd",result));

        return result;

    }

    static inline OutputImagePointer coarseLNCC(InputImagePointer i1,InputImagePointer i2, InputImagePointer coarseImg,double sigma=1.0, double exp = 1.0){
        return coarseLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, coarseImg,sigma,exp);
   }
    static inline OutputImagePointer coarseLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2, InputImagePointer coarseImg,double sigma=1.0, double exp = 1.0){
        OutputImagePointer i1Cast=cast(i1);
        OutputImagePointer i2Cast=cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigma);
        //filter->SetVariance(sigma*sigma);
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Bar.mhd",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        OutputImagePointer i2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput());  

        //FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Bar.mhd",i2Bar));
        //HACK!
        //compute squares of original images


        
        OutputImagePointer i1Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        OutputImagePointer i2Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Square.mhd",i2Square));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Square.mhd",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        OutputImagePointer i1SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1SquareBar.mhd",i1SquareBar));
       
        filter->SetInput(i2Square);
        filter->Update();
        OutputImagePointer i2SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2SquareBar.mhd",i2SquareBar));

        //compute squares of local means
        OutputImagePointer i1BarSquare=ImageUtils<OutputImage>::localSquare(i1Bar);
        OutputImagePointer i2BarSquare=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        //multiply i1 and i2 locally
        OutputImagePointer i1i2=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1i2.mhd",i1i2));

        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarSquare.mhd",i1BarSquare));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2BarSquare.mhd",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        OutputImagePointer i1Timesi2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Timesi2Bar.mhd",i1Timesi2Bar));

        //finish
        OutputImagePointer result=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        itk::ImageRegionIterator<OutputImage> resultIt(result,result->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        
        itk::ImageRegionIterator<OutputImage> i1BarIt(i1Bar,i1Bar->GetLargestPossibleRegion());
        i1BarIt.GoToBegin();
        itk::ImageRegionIterator<OutputImage> i2BarIt(i2Bar,i2Bar->GetLargestPossibleRegion());
        i2BarIt.GoToBegin();

        itk::ImageRegionIterator<OutputImage> i1Timesi2BarIt(i1Timesi2Bar,i1Timesi2Bar->GetLargestPossibleRegion());
        i1Timesi2BarIt.GoToBegin();

        itk::ImageRegionIterator<OutputImage> i1SquareBarIt(i1SquareBar,i1SquareBar->GetLargestPossibleRegion());
        i1SquareBarIt.GoToBegin();

        itk::ImageRegionIterator<OutputImage> i2SquareBarIt(i2SquareBar,i2SquareBar->GetLargestPossibleRegion());
        i2SquareBarIt.GoToBegin();

        
        
        for (;!resultIt.IsAtEnd();++resultIt){

            
            double i1BarV=i1BarIt.Get();
            double i2BarV=i2BarIt.Get();
            double i1BarTimesi2BarV=i1BarV*i2BarV;
         
            
            double i1Timesi2BarV=i1Timesi2BarIt.Get();
            double numeratorV=i1Timesi2BarV-i1BarTimesi2BarV;

            double varianceI1V=sqrt(max(0.0,i1SquareBarIt.Get() - i1BarV*i1BarV));
            double varianceI2V=sqrt(max(0.0,i2SquareBarIt.Get() - i2BarV*i2BarV));
            double denominatorV=varianceI1V*varianceI2V;

            double r = numeratorV/(denominatorV+0.00000000001);

            ++i1BarIt;
            ++i2BarIt;
            ++i1Timesi2BarIt;
            ++i1SquareBarIt;
            ++i2SquareBarIt;


            if (r< -(OutputImagePixelType)1.0 ){
                r = 0;
            }else if ( r > (OutputImagePixelType)1.0){
                r = 0;
            }
            r = pow((r+1.0)/2,exp);
            
            r = max(r,std::numeric_limits<double>::epsilon());

            resultIt.Set(r);

        }
        
        LOGI(8,ImageUtils<OutputImage>::writeImage("result.mhd",result));

        return result;

    }
};
