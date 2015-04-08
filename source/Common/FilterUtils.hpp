

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
#include "itkBSplineInterpolateImageFunction.h"
#include "ImageUtils.h"
#include <algorithm> //max,min
#include "Log.h"
#include "itkStatisticsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNeighborhoodIterator.h"
#include "itkLaplacianRecursiveGaussianImageFilter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkNormalizeImageFilter.h"
#include <itkGrayscaleFillholeImageFilter.h>
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include <itkMinimumMaximumImageCalculator.h>
#include <algorithm>

///\brief This class provides static functions which are mostly wrappers around ITK routines.
///Its main purpose is to alleviate the inclusion of common methods such as resampling, thresholding,...
template<class InputImage, class OutputImage = InputImage>
class FilterUtils {
public:

    typedef typename InputImage::Pointer  InputImagePointer;
    typedef typename InputImage::ConstPointer  ConstInputImagePointer;
    
    typedef typename InputImage::PixelType InputImagePixelType;
    typedef typename InputImage::IndexType InputImageIndex;
    typedef typename InputImage::RegionType InputImageRegion;
    typedef typename InputImage::SpacingType SpacingType;
    typedef typename InputImage::SizeType SizeType;
    typedef typename InputImage::DirectionType DirectionType;
    typedef typename InputImage::PointType OriginType;

    typedef typename OutputImage::Pointer OutputImagePointer;
    typedef typename OutputImage::IndexType OutputImageIndex;
    typedef typename OutputImage::RegionType OutputImageRegion;
    typedef typename OutputImage::PixelType OutputImagePixelType;
    typedef typename OutputImage::PointType OutputImagePointType;

    static const int D=InputImage::ImageDimension;
    typedef itk::FlatStructuringElement< InputImage::ImageDimension > StructuringElementType;
    typedef typename StructuringElementType::RadiusType StructuringElementTypeRadius;

    typedef itk::Image<double,D> FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;

    typedef itk::BinaryThresholdImageFilter<InputImage,OutputImage> BinaryThresholdFilter;
    typedef itk::BinaryErodeImageFilter<InputImage, OutputImage, StructuringElementType > ErodeFilterType;
    typedef itk::BinaryDilateImageFilter<InputImage, OutputImage, StructuringElementType > DilateFilterType;
    typedef itk::CastImageFilter <InputImage,OutputImage> CastImageFilterType;
    typedef itk::MaskImageFilter<InputImage,InputImage,OutputImage> MaskImageFilterType;
    typedef itk::MaskNegatedImageFilter<InputImage,InputImage,OutputImage> MaskNegatedImageFilterType;
    typedef itk::SubtractImageFilter <InputImage,InputImage,OutputImage> SubtractFilterType;
    typedef itk::AddImageFilter <InputImage,InputImage,OutputImage> AddFilterType;
    //typedef itk::ConnectedComponentImageFilter<InputImage,OutputImage>  ConnectedComponentImageFilterType;
    typedef itk::RelabelComponentImageFilter<InputImage,OutputImage>  RelabelComponentImageFilterType;
    typedef itk::ShiftScaleImageFilter<InputImage,OutputImage>  ShiftScaleImageFilterType;

#define RECURSIVEGAUSSIAN
#ifdef RECURSIVEGAUSSIAN
    typedef itk::SmoothingRecursiveGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
    //typedef itk::RecursiveGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
#else
    //typedef itk::DiscreteGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
#endif
    typedef itk::FastMarchingImageFilter<OutputImage>  FastMarchingImageFilterType;
    //REMI:typedef itk::PasteImageFilter<InputImage,OutputImage>  PasteImageFilterType;

  
  
    //typedef typename ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
    typedef typename FastMarchingImageFilterType::Pointer FastMarchingImageFilterPointer;
    //REMI:typedef typename PasteImageFilterType::Pointer PasteImageFilterPointer;

    typedef ImageUtils<InputImage> InputImageUtils;
    typedef ImageUtils<OutputImage> OutputImageUtils;

    typedef typename FastMarchingImageFilterType::NodeContainer FastMarchingNodeContainer;
    typedef typename FastMarchingNodeContainer::Pointer FastMarchingNodeContainerPointer;
    typedef typename FastMarchingImageFilterType::NodeType FastMarchingNodeType;

    typedef typename itk::LinearInterpolateImageFunction<InputImage, double> LinearInterpolatorType;
    typedef typename LinearInterpolatorType::Pointer LinearInterpolatorPointerType;
    typedef typename itk::BSplineInterpolateImageFunction<InputImage, double> BSplineInterpolatorType;
    typedef typename itk::NearestNeighborInterpolateImageFunction<InputImage, double> NNInterpolatorType;
    typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;
    typedef typename itk::ResampleImageFilter< InputImage , OutputImage>	ResampleFilterType;
    typedef typename ResampleFilterType::Pointer ResampleFilterPointerType;
    typedef typename itk::StatisticsImageFilter<InputImage > StatisticsFilterType;
    
    typedef typename itk::RescaleIntensityImageFilter<    InputImage, OutputImage >  RescaleFilterType;

    typedef typename itk::LaplacianRecursiveGaussianImageFilter<InputImage,OutputImage> LaplacianFilterType;
    typedef typename itk::GradientMagnitudeImageFilter<InputImage,OutputImage> GradientFilterType;
    typedef itk::NormalizeImageFilter<InputImage,OutputImage> NormalizeImageFilterType;
    typedef typename NormalizeImageFilterType::Pointer NormalizeImageFilterPointerType;

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
    static OutputImagePointer normalizeImage(InputImagePointer input){
     
        return normalizeImage((ConstInputImagePointer)input);
    }
    static OutputImagePointer normalizeImage(ConstInputImagePointer input){
        typename OutputImage::PixelType px;
        px=1.3;
        double test=1.3;
        if ( (test-px) > std::numeric_limits<float>::epsilon ()){
            LOG<<VAR(px)<<" "<<VAR(test)<<" "<<VAR(test-px)<<std::endl;
            LOG<<"WARNING: normalizing integer image type to zero mean/unit variance will probably not work well!"<<std::endl;
        }
        NormalizeImageFilterPointerType filter=NormalizeImageFilterType::New();
        filter->SetInput(input);
        filter->Update();
        return filter->GetOutput();
    }

    static OutputImagePointer gradient(ConstInputImagePointer input){
        
        typename GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
        gradientFilter->SetInput(input);
        gradientFilter->Update();
        return gradientFilter->GetOutput();

    }
    static OutputImagePointer gradient(InputImagePointer input){
        
        typename GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
        gradientFilter->SetInput(input);
        gradientFilter->Update();
        return gradientFilter->GetOutput();

    }

    static OutputImagePointer laplacian(InputImagePointer input,double sigma){
        
        typename LaplacianFilterType::Pointer laplacianFilter = LaplacianFilterType::New();
        laplacianFilter->SetInput((ConstInputImagePointer)input);
        laplacianFilter->SetSigma(sigma);
        laplacianFilter->Update();
        return laplacianFilter->GetOutput();

    }
    static OutputImagePointer normalizedLaplacianWeighting(InputImagePointer input,double sigma,double expo){
        OutputImagePointer gradient=laplacian(input,sigma);
        typedef typename itk::StatisticsImageFilter<OutputImage > OutStatisticsFilterType;
        typename OutStatisticsFilterType::Pointer statistics=OutStatisticsFilterType::New();
        
        statistics->SetInput(gradient);
        statistics->Update();
        double mean=statistics->GetMean();

        typename ImageUtils<OutputImage>::ImageIteratorType gradientIt(gradient,gradient->GetLargestPossibleRegion());
        for(gradientIt.GoToBegin();!gradientIt.IsAtEnd();++gradientIt){

            double grad=gradientIt.Get();
            double weight=1.0 - exp( -0.5* fabs(grad)/(mean*expo));
            gradientIt.Set(weight);

        }
        return gradient;

    }

    

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



    static OutputImagePointer maximumResample( InputImagePointer input, InputImagePointer reference, double radius =-1.0) {
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
            double maxVal=std::numeric_limits<OutputImagePixelType>::min();
            for (int i=0;i<inputIt.Size();++i){
                bool inside;
                double value=inputIt.GetPixel(i,inside);
                if (inside && value>maxVal)
                    maxVal=value;

            }
            resultIt.Set(maxVal);
        }
        return result;
    }


#ifdef ISOTROPIC_RESAMPLING
  
    static OutputImagePointer LinearResample( ConstInputImagePointer input,  double scale, bool smooth,bool nnResample=false) {
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        NNInterpolatorPointerType interpolNN=NNInterpolatorType::New();

        ResampleFilterPointerType resampler=ResampleFilterType::New();
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
        for (unsigned int d=0;d<InputImage::ImageDimension;++d){
            size[d]=int(inputSize[d]*scale);
            spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
            origin[d]=inputOrigin[d];//+0.5*spacing[d]/inputSpacing[d];
        }
        resampler->SetOutputOrigin(origin);
		resampler->SetOutputSpacing ( spacing );
		resampler->SetOutputDirection ( input->GetDirection() );
		resampler->SetSize ( size );
        
        if (smooth && scale<1.0){
            InputImagePointer smoothedInput = gaussian(input,spacing-inputSpacing);
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }

        resampler->Update();
        return resampler->GetOutput();
    }


#else
    //downscale to isotropic spacing defined by minspacing/scale
    //never upsample! unless scale>1.0
    static OutputImagePointer LinearResample( ConstInputImagePointer input,  double scale, bool smooth, bool nnResample=false) {

        if (scale == 1.0) return cast(ImageUtils<InputImage>::duplicateConst(input));
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        NNInterpolatorPointerType interpolNN=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        LOGV(5)<<VAR(smooth)<<" "<<VAR(nnResample)<<std::endl;
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
        for (unsigned int d=0;d<InputImage::ImageDimension;++d){
            if (inputSpacing[d]<minSpacing) minSpacing=inputSpacing[d];
        }
        double newSpacing=minSpacing/scale;
        LOGV(7)<<"new isotrpoic spacing : "<<newSpacing<<std::endl;
        for (unsigned int d=0;d<InputImage::ImageDimension;++d){
            //determine new spacing
            //never increase resolution!
            if (scale>1.0){
                spacing[d]=newSpacing;//inputSpacing[d]*(1.0*inputSize[d]/size[d]);
            }else{
	      spacing[d]=inputSpacing[d]>newSpacing?inputSpacing[d]:newSpacing;
            }
            //calculate new image size
            size[d]=int(inputSpacing[d]/spacing[d] * (inputSize[d]-1))+1;
            //finalize spacing as a function of the new size
            spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
            //size[d]=int(inputSpacing[d]/spacing[d]*(inputSize[d]));

            origin[d]=inputOrigin[d];//+0.5*spacing[d]/inputSpacing[d];
        }
        LOGV(7)<<"full parameters : "<<spacing<<" "<<size<<" "<<origin<<std::endl;
        LOGV(3)<<"Resampling to isotropic  spacing "<<spacing<<" with resolution "<<size<<std::endl;
        resampler->SetOutputOrigin(origin);
		resampler->SetOutputSpacing ( spacing );
		resampler->SetOutputDirection ( input->GetDirection() );
		resampler->SetSize ( size );
        if (smooth && scale<1.0){
            InputImagePointer smoothedInput = FilterUtils<InputImage,InputImage>::gaussian(input,spacing-inputSpacing);
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }

        resampler->Update();
        return resampler->GetOutput();
    }
  
#endif
    static OutputImagePointer LinearResample( InputImagePointer input,  double scale, bool smooth,bool nnResample=false) {
        return LinearResample(ConstInputImagePointer(input),scale, smooth, nnResample);
    }

    static OutputImagePointer NNResample( ConstInputImagePointer input,  double scale, bool smooth) {
        return LinearResample(input,scale,smooth,true);
    }
    static OutputImagePointer NNResample( InputImagePointer input,  double scale,bool smooth) {
        return LinearResample((ConstInputImagePointer)input,scale,smooth,true);
    }

      static OutputImagePointer ResampleIsotropic( InputImagePointer input,  double spacing, bool smooth, bool nnResample=false) {
          return ResampleIsotropic(ConstInputImagePointer(input),spacing, smooth, nnResample);

      }
      static OutputImagePointer ResampleIsotropic( ConstInputImagePointer input,  double isoSpacing, bool smooth, bool nnResample=false) {

        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        NNInterpolatorPointerType interpolNN=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        LOGV(5)<<VAR(smooth)<<" "<<VAR(nnResample)<<std::endl;
        if (nnResample){
            resampler->SetInterpolator(interpolNN);
            smooth=false;
        }
        else
            resampler->SetInterpolator(interpol);
        typename InputImage::SpacingType spacing,inputSpacing;
        typename InputImage::SizeType size,inputSize;
        typename InputImage::PointType origin,inputOrigin;
        inputOrigin=input->GetOrigin();
        inputSize=input->GetLargestPossibleRegion().GetSize();
        inputSpacing=input->GetSpacing();
      
        double newSpacing=isoSpacing;
        LOGV(7)<<"new isotrpoic spacing : "<<newSpacing<<std::endl;
        for (unsigned int d=0;d<InputImage::ImageDimension;++d){
            spacing[d]=newSpacing;//inputSpacing[d]*(1.0*inputSize[d]/size[d]);
            //calculate new image size
            size[d]=int(inputSpacing[d]/newSpacing * (inputSize[d]-1))+1;
            //finalize spacing as a function of the new size
            //spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
            //size[d]=int(inputSpacing[d]/spacing[d]*(inputSize[d]));
            origin[d]=inputOrigin[d];//+0.5*spacing[d]/inputSpacing[d];
        }
        LOGV(7)<<"full parameters : "<<spacing<<" "<<size<<" "<<origin<<std::endl;
        LOGV(3)<<"Resampling to isotropic  spacing "<<spacing<<" with resolution "<<size<<std::endl;
        resampler->SetOutputOrigin(origin);
		resampler->SetOutputSpacing ( spacing );
		resampler->SetOutputDirection ( input->GetDirection() );
		resampler->SetSize ( size );
        if (smooth ){
            InputImagePointer smoothedInput = FilterUtils<InputImage,InputImage>::gaussian(input,spacing-inputSpacing);
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }

        resampler->Update();
        return resampler->GetOutput();
    }

    //resamp
    static OutputImagePointer EmptyResample( InputImagePointer input,  double scale) {

        typename InputImage::SpacingType spacing,inputSpacing;
        typename InputImage::SizeType size,inputSize;
        typename InputImage::PointType origin,inputOrigin;
        typename InputImage::RegionType region;;
        inputOrigin=input->GetOrigin();
        inputSize=input->GetLargestPossibleRegion().GetSize();
        inputSpacing=input->GetSpacing();
        double minSpacing=std::numeric_limits<double>::max();
        for (unsigned int d=0;d<InputImage::ImageDimension;++d){
            if (inputSpacing[d]<minSpacing) minSpacing=inputSpacing[d];
        }
        double newSpacing=minSpacing/scale;
        LOGV(7)<<"new isotrpoic spacing : "<<newSpacing<<std::endl;
        for (unsigned int d=0;d<InputImage::ImageDimension;++d){
            //determine new spacing
            //never increase resolution!
            if (scale>1.0){
                spacing[d]=newSpacing;
            }else{

	      spacing[d]=inputSpacing[d]>newSpacing?inputSpacing[d]:newSpacing;
            }
            //calculate new image size
            size[d]=int(inputSpacing[d]/spacing[d] * (inputSize[d]-1))+1;
            //finalize spacing as a function of the new size
            spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
             
            origin[d]=inputOrigin[d];
        }

        OutputImagePointer result=OutputImage::New();
        result->SetOrigin(origin);
        result->SetSpacing(spacing);
        result->SetDirection(input->GetDirection());
        region.SetSize(size);
        result->SetRegions(region);
        return result;
         
    }



    static OutputImagePointer NNResample( ConstInputImagePointer input,  ConstInputImagePointer reference, bool smooth) {
        NNInterpolatorPointerType interpol=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();  
        if (smooth){
            LOG<<"NN resampling with gaussian filter, DOES NOT MAKE SENSE, does it? "<<std::endl;
            InputImagePointer smoothedInput = gaussian(input,reference->GetSpacing());
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }
        resampler->SetInterpolator(interpol);
    
        resampler->SetOutputOrigin(reference->GetOrigin());
		resampler->SetOutputSpacing ( reference->GetSpacing() );
		resampler->SetOutputDirection ( reference->GetDirection() );
		resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }
    static OutputImagePointer LinearResample( ConstInputImagePointer input,  ConstInputImagePointer reference, bool smooth) {
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        if (smooth){
            InputImagePointer smoothedInput = gaussian(input,(reference->GetSpacing()-input->GetSpacing())/2);
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }
        resampler->SetInterpolator(interpol);
        resampler->SetOutputOrigin(reference->GetOrigin());
		resampler->SetOutputSpacing ( reference->GetSpacing() );
		resampler->SetOutputDirection ( reference->GetDirection() );
		resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }
    static OutputImagePointer LinearResample( InputImagePointer input,  SizeType size, OriginType origin, SpacingType spacing, DirectionType dir, bool smooth) {
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        if (smooth){
            InputImagePointer smoothedInput = gaussian(input,(spacing-input->GetSpacing())/2);
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }
        resampler->SetInterpolator(interpol);
        resampler->SetOutputOrigin(origin);
		resampler->SetOutputSpacing (spacing );
		resampler->SetOutputDirection ( dir );
		resampler->SetSize ( size );
        resampler->Update();
        return resampler->GetOutput();
    }
  
    static OutputImagePointer LinearResample( InputImagePointer input,  InputImagePointer reference, bool smooth) {
        return LinearResample((ConstInputImagePointer)input,(ConstInputImagePointer)reference, smooth);
    }
    static OutputImagePointer NNResample( InputImagePointer input,  InputImagePointer reference, bool smooth) {
        return NNResample((ConstInputImagePointer)input,(ConstInputImagePointer)reference, smooth);
    }

    static OutputImagePointer NNResample( InputImagePointer input,  ConstInputImagePointer reference, bool smooth) {
        NNInterpolatorPointerType interpol=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        if (smooth){
            InputImagePointer smoothedInput = gaussian(input,reference->GetSpacing()-input->GetSpacing());
            resampler->SetInput(smoothedInput);
        }else{
            resampler->SetInput(input);
        }
        resampler->SetInterpolator(interpol);
    
        resampler->SetOutputOrigin(reference->GetOrigin());
		resampler->SetOutputSpacing ( reference->GetSpacing() );
		resampler->SetOutputDirection ( reference->GetDirection() );
		resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }

    static OutputImagePointer BSplineResampleSegmentation( InputImagePointer input,  ConstInputImagePointer reference) {
        typedef typename BSplineInterpolatorType::Pointer BSplineInterpolatorPointerType;

        int nSegmentations = getMax(input)+1;
        typedef typename itk::Image<float,InputImage::ImageDimension> FloatImageType;
        std::vector<typename FloatImageType::Pointer> segmentationImages(nSegmentations);

        typedef typename itk::ResampleImageFilter< InputImage , FloatImageType>	FloatResampleFilterType;
        typedef typename FloatResampleFilterType::Pointer FloatResampleFilterPointerType;
        for (int s=0;s<nSegmentations;++s){
            BSplineInterpolatorPointerType interpol=BSplineInterpolatorType::New();
            FloatResampleFilterPointerType resampler=FloatResampleFilterType::New();
            resampler->SetInput(select(input,s));
            resampler->SetInterpolator(interpol);
            resampler->SetOutputOrigin(reference->GetOrigin());
            resampler->SetOutputSpacing ( reference->GetSpacing() );
            resampler->SetOutputDirection ( reference->GetDirection() );
            resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
            resampler->Update();
            segmentationImages[s]=resampler->GetOutput();

        }
        OutputImagePointer result = createEmpty(reference);
        itk::ImageRegionIteratorWithIndex<OutputImage>  resultIt(result,result->GetLargestPossibleRegion());
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt){
            OutputImageIndex idx=resultIt.GetIndex();
            int maxLabel=-1;
            double maxVal=-1;
            for (int s=0;s<nSegmentations;++s){
                float value = segmentationImages[s]->GetPixel(idx);
                if (value>maxVal){
                    maxVal=value;
                    maxLabel=s;
                }
            }
            resultIt.Set(maxLabel);
        }

        return result;
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

        typedef typename ShiftScaleImageFilterType::Pointer ShiftScaleImageFilterPointer;

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
                                       ConstInputImagePointer image, float sigma, bool spacing=true
                                       ) {
        
        if (! image.IsNotNull()){
            return NULL;
        }
        typedef typename DiscreteGaussianImageFilterType::Pointer DiscreteGaussianImageFilterPointer;

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();


        filter->SetInput(image);
#ifdef RECURSIVEGAUSSIAN
        filter->SetSigma(sigma);
#else
        filter->SetVariance(sigma*sigma);
#endif
        //if (!spacing)
        //  filter->SetUseImageSpacingOff();
        filter->Update();

        return filter->GetOutput();
    }

    static OutputImagePointer gaussian(InputImagePointer image, float variance, bool spacing=true
                                       ) {
        return gaussian(ConstInputImagePointer(image),variance,spacing);
    }

    static OutputImagePointer gaussian(InputImagePointer image, SpacingType spacing){
        
        return gaussian(ConstInputImagePointer(image),spacing);
        
    }

    static OutputImagePointer gaussian(ConstInputImagePointer image, SpacingType spacing
                                       ) {
        if (! image.IsNotNull()){
            return NULL;
        }

        for (int d=0;d<InputImage::ImageDimension;++d){
            if (spacing[d]==0){
                spacing[d]=1e-5;
            }
            spacing[d]=sqrt(spacing[d]);
        }
        typedef typename DiscreteGaussianImageFilterType::Pointer DiscreteGaussianImageFilterPointer;

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();
        LOGV(4)<<"gaussian smoothing with "<<VAR(spacing)<<std::endl;
        filter->SetInput(image);
#ifdef RECURSIVEGAUSSIAN
        filter->SetSigmaArray(spacing);
#else
        
#endif       
        filter->Update();
        LOGV(4)<<"success"<<std::endl;
        return filter->GetOutput();
    }

    // relabel components according to its size.
    // Largest component 1, second largest 2, ...
    static OutputImagePointer relabelComponents(InputImagePointer image) {
        typedef typename RelabelComponentImageFilterType::Pointer RelabelComponentImageFilterPointer;

        RelabelComponentImageFilterPointer filter =
            RelabelComponentImageFilterType::New();

        filter->SetInput(image);
        filter->Update();

        return filter->GetOutput();
    }

    static OutputImagePointer cast(ConstInputImagePointer image) {
        typedef typename CastImageFilterType::Pointer CastFilterPointer;

        CastFilterPointer castFilter = CastImageFilterType::New();
        castFilter->SetInput(image);
        castFilter->Update();
        return castFilter->GetOutput();
    }

    // cast the image to the output type
    static OutputImagePointer cast(InputImagePointer image) {
        typedef typename CastImageFilterType::Pointer CastFilterPointer;

        CastFilterPointer castFilter = CastImageFilterType::New();
        castFilter->SetInput(image);
        castFilter->Update();
        return castFilter->GetOutput();
    }
    
    static OutputImagePointer truncateCast(InputImagePointer image) {
        typedef typename CastImageFilterType::Pointer CastFilterPointer;

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
        typedef typename SubtractFilterType::Pointer SubtractFilterPointer;
        SubtractFilterPointer substractFilter = SubtractFilterType::New();
        substractFilter->SetInput1(image1);
        substractFilter->SetInput2(image2);
        substractFilter->InPlaceOff();
        //substractFilter->SetCoordinateTolerance(1e-5);
        substractFilter->Update();
        return substractFilter->GetOutput();
    }





    // pixel-wise addition of two images
    static OutputImagePointer add(
                                  InputImagePointer image1, InputImagePointer image2
                                  ) {
        typedef typename AddFilterType::Pointer AddFilterPointer;

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
        typedef typename MaskImageFilterType::Pointer MaskImageFilterPointer;
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
        typedef typename MaskNegatedImageFilterType::Pointer MaskNegatedImageFilterPointer;
        MaskNegatedImageFilterPointer filter = MaskNegatedImageFilterType::New();

        filter->SetInput1(image);
        filter->SetInput2(mask);
        filter->Update();

        return filter->GetOutput();
    }


 static OutputImagePointer binaryThresholding(
                                                 ConstInputImagePointer inputImage,
                                                 InputImagePixelType lowerThreshold,
                                                 InputImagePixelType upperThreshold,
                                                 OutputImagePixelType insideValue = 1,
                                                 OutputImagePixelType outsideValue = 0
                                                 ) {
        typedef typename BinaryThresholdFilter::Pointer BinaryThresholdFilterPointer;
  
        BinaryThresholdFilterPointer thresholder = BinaryThresholdFilter::New();
        thresholder->SetInput(inputImage);

        thresholder->SetLowerThreshold( lowerThreshold );
        thresholder->SetUpperThreshold( upperThreshold );
        thresholder->SetInsideValue(insideValue);
        thresholder->SetOutsideValue(outsideValue);

        thresholder->Update();

        return thresholder->GetOutput();
    }



    static OutputImagePointer binaryThresholding(
                                                 InputImagePointer inputImage,
                                                 InputImagePixelType lowerThreshold,
                                                 InputImagePixelType upperThreshold,
                                                 OutputImagePixelType insideValue = 1,
                                                 OutputImagePixelType outsideValue = 0
                                                 ) {
        typedef typename BinaryThresholdFilter::Pointer BinaryThresholdFilterPointer;
  
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
        typedef typename BinaryThresholdFilter::Pointer BinaryThresholdFilterPointer;
  
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
        typedef typename BinaryThresholdFilter::Pointer BinaryThresholdFilterPointer;
        
        BinaryThresholdFilterPointer thresholder = BinaryThresholdFilter::New();
        thresholder->SetInput(inputImage);
        thresholder->SetUpperThreshold( upperThreshold );
        thresholder->SetInsideValue(insideValue);
        thresholder->SetOutsideValue(outsideValue);
        
        thresholder->Update();
        
        return thresholder->GetOutput();
    }

    static OutputImagePointer myBinaryThresholdingHigh(
                                                       InputImagePointer inputImage,
                                                       InputImagePixelType upperThreshold,
                                                       OutputImagePixelType insideValue = 1,
                                                       OutputImagePixelType outsideValue = 0
                                                       ) {

        OutputImagePointer outputImage=createEmpty(ConstInputImagePointer(inputImage));
        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        itk::ImageRegionIterator<OutputImage> it2(
                                                  outputImage, outputImage->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it,++it2) {
            double val=it.Get();
            it2.Set(val<=upperThreshold?insideValue:outsideValue);
        }
        return outputImage;

    }

    static OutputImagePointer round(
                                    InputImagePointer inputImage
                                    ) {
        OutputImagePointer outputImage=createEmpty(ConstInputImagePointer(inputImage));
        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        itk::ImageRegionIterator<OutputImage> it2(
                                                  outputImage, outputImage->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it,++it2) {
            double val=it.Get();
            it2.Set(floor(val+0.5));
        }
        return outputImage;
    }

       static OutputImagePointer sign(
                                    InputImagePointer inputImage
                                    ) {
        OutputImagePointer outputImage=createEmpty(ConstInputImagePointer(inputImage));
        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        itk::ImageRegionIterator<OutputImage> it2(
                                                  outputImage, outputImage->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it,++it2) {
            double val=it.Get();
            it2.Set(val>0?1.0:-1.0);
        }
        return outputImage;
    }

    static OutputImagePointer invert(
                                     InputImagePointer inputImage
                                     ) {
        OutputImagePointer outputImage=createEmpty(ConstInputImagePointer(inputImage));
        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        itk::ImageRegionIterator<OutputImage> it2(
                                                  outputImage, outputImage->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it,++it2) {
            int val=it.Get();
            it2.Set(val>0.0?0.0:1.0);
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

    static double sum(InputImagePointer inputImage) {

        itk::ImageRegionIterator<InputImage> it(
                                                inputImage, inputImage->GetLargestPossibleRegion());
        double result=0;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            result+=it.Get();
        }

        return result;
    }

    static OutputImagePointer select(
                                     InputImagePointer inputImage,
                                     InputImagePixelType value
                                     ) {

        return binaryThresholding(inputImage,value,value);
    }
    static OutputImagePointer select(
                                     ConstInputImagePointer inputImage,
                                     InputImagePixelType value
                                     ) {

        return binaryThresholding(inputImage,value,value);
    }

    // perform erosion (mathematical morphology) with a given label image
    // using a ball with a given radius
    static OutputImagePointer erosion(
                                      InputImagePointer labelImage, unsigned radius,
                                      InputImagePixelType valueToErode = 1
                                      ) {
        StructuringElementTypeRadius rad; rad.Fill(radius);
        StructuringElementType K = StructuringElementType::Ball( rad );
        typedef typename ErodeFilterType::Pointer ErodeFilterPointer;
        ErodeFilterPointer erosionFilter = ErodeFilterType::New();
        erosionFilter->SetKernel(K);
        erosionFilter->SetErodeValue(valueToErode);

        erosionFilter->SetInput( labelImage );
        erosionFilter->Update();
        return erosionFilter->GetOutput();

    }



    // perform dilation (mathematical morphology) with a given label image
    // using a ball with a given radius
    static OutputImagePointer dilation(
                                       InputImagePointer labelImage, unsigned radius,
                                       InputImagePixelType valueToDilate = 1
                                       ) {
        StructuringElementTypeRadius rad; rad.Fill(radius);
        StructuringElementType K = StructuringElementType::Ball( rad );
        typedef typename DilateFilterType::Pointer DilateFilterPointer;

        DilateFilterPointer dilateFilter  = DilateFilterType::New();
        dilateFilter->SetKernel(K);
        dilateFilter->SetDilateValue(valueToDilate);

        dilateFilter->SetInput( labelImage );
        dilateFilter->Update();

        return dilateFilter->GetOutput();
    }

    // perform dilation (distance based) with a given label image
    static OutputImagePointer myDilation(
                                         InputImagePointer labelImage, unsigned radius,
                                         InputImagePixelType valueToDilate = 1
                                         ) {
     
        FloatImagePointerType dist=FilterUtils<InputImage,FloatImageType>::distanceMapByFastMarcher(labelImage,valueToDilate);
        return FilterUtils<FloatImageType,OutputImage>::binaryThresholdingHigh(dist,1.0*radius);

        //        return dilateFilter->GetOutput();
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
        return distanceMapByFastMarcher((ConstInputImagePointer)image,objectLabel, stoppingValue);
    }
    static OutputImagePointer distanceMapByFastMarcher(
                                                       ConstInputImagePointer image,
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

        itk::ImageRegionConstIteratorWithIndex<InputImage> it(
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

    static OutputImagePointer distanceMapBySignedMaurer(
                                                        InputImagePointer image,
                                                        InputImagePixelType objectLabel){
        return distanceMapBySignedMaurer((ConstInputImagePointer)image,objectLabel);
    }
    static OutputImagePointer distanceMapBySignedMaurer(
                                                        ConstInputImagePointer image,
                                                        InputImagePixelType objectLabel
                                                        ) {

        typedef typename itk::SignedMaurerDistanceMapImageFilter< InputImage, OutputImage > DistanceTransformType;
        typename DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
        //distanceTransform->InsideIsPositiveOn();
        distanceTransform->SetInput(select(image,objectLabel));
        distanceTransform->SquaredDistanceOff ();
        distanceTransform->UseImageSpacingOn();
        distanceTransform->Update();

        return distanceTransform->GetOutput();

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
    static InputImagePixelType getMin(InputImagePointer img){
        return getMin(ConstInputImagePointer(img));
    }    
    static InputImagePixelType getMin(ConstInputImagePointer img){
        typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
        filter->SetInput(img);
        filter->Update();
        return filter->GetMinimumOutput()->Get();

    }
    static InputImagePixelType getMin(InputImagePointer img,InputImageRegion region){
        typedef typename itk::MinimumMaximumImageCalculator<InputImage> FilterType;
        typename FilterType::Pointer  filter =FilterType::New();
        filter->SetImage(img);
        filter->SetRegion(region);
        //filter->Update();
        filter->ComputeMinimum();
        return filter->GetMinimum();

    }
    static InputImagePixelType getMean(InputImagePointer img){
        typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
        filter->SetInput(img);
        filter->Update();
        return filter->GetMean();

    }
  static InputImagePixelType getVariance(InputImagePointer img){
        typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
        filter->SetInput(img);
        filter->Update();
        return filter->GetVariance();

    }


    static OutputImagePointer normalize(InputImagePointer img){
        typename RescaleFilterType::Pointer    rescaleFilter    = RescaleFilterType::New();
        rescaleFilter->SetInput(img);
        rescaleFilter->SetOutputMinimum(std::numeric_limits<OutputImagePixelType>::min());
        rescaleFilter->SetOutputMaximum(std::numeric_limits<OutputImagePixelType>::max());
        rescaleFilter->Update();
        return rescaleFilter->GetOutput();
    }

    static void localMax(InputImagePointer i1,InputImagePointer i2){
        itk::ImageRegionIterator<InputImage> i1It(i1,i1->GetLargestPossibleRegion());
        itk::ImageRegionIterator<InputImage> i2It(i2,i2->GetLargestPossibleRegion());
        for (i1It.GoToBegin(),i2It.GoToBegin();!i1It.IsAtEnd();++i1It,++i2It){
            float v1=i1It.Get();
            float v2=i2It.Get();
            i1It.Set(v1>v2?v1:v2);
        }
    }
    static void localMin(InputImagePointer i1,InputImagePointer i2){
        itk::ImageRegionIterator<InputImage> i1It(i1,i1->GetLargestPossibleRegion());
        itk::ImageRegionIterator<InputImage> i2It(i2,i2->GetLargestPossibleRegion());
        for (i1It.GoToBegin(),i2It.GoToBegin();!i1It.IsAtEnd();++i1It,++i2It){
            float v1=i1It.Get();
            float v2=i2It.Get();
            i1It.Set(v1<v2?v1:v2);
        }
    }

    static OutputImagePointer upsampleSegmentation(InputImagePointer seg,InputImagePointer ref){

        OutputImagePointer result=FilterUtils<InputImage,OutputImage>::NNResample(seg,ref,false);
        result->FillBuffer(0);
        int maxLabel=getMax(seg);
        typedef typename ImageUtils<InputImage>::FloatImageType FloatImageType;
        typedef typename ImageUtils<InputImage>::FloatImagePointerType FloatImagePointerType;
        for (int l=1;l<=maxLabel;++l){

            LOGV(2)<<"Upsampling segmentation for label "<<l<<std::endl;
            LOGV(2)<<"Computing distance map"<<std::endl;
            FloatImagePointerType distanceMap=FilterUtils<InputImage,FloatImageType>::distanceMapByFastMarcher(FilterUtils<InputImage>::select(seg,l),1);
            LOGI(6,ImageUtils<FloatImageType>::writeImage("distnaceMapLow.nii",distanceMap));
            LOGV(2)<<"Resampling and smoothing distance map..."<<std::endl;
            distanceMap=FilterUtils<FloatImageType>::LinearResample(distanceMap,FilterUtils<InputImage,FloatImageType>::cast(ref),false);
            distanceMap=FilterUtils<FloatImageType>::gaussian(distanceMap,(seg->GetSpacing()-ref->GetSpacing())*0.5);
            LOGI(6,ImageUtils<FloatImageType>::writeImage("distnaceMaphigh.nii",distanceMap));
            LOGV(2)<<"Thresholding distance map to get segmentation"<<std::endl;
            OutputImagePointer partResult=FilterUtils<FloatImageType,OutputImage>::binaryThresholdingHigh(distanceMap,0.5);
            LOGV(2)<<"Mapping segmentation labels..."<<std::endl;
            combineSegmentations(result,partResult,l);
        }
        return result;
    }

    //combines two segmentations (or images), assuming zero is background
    //second image always overwrites first
    static void combineSegmentations(InputImagePointer i1, InputImagePointer i2, int label=1){
        itk::ImageRegionIterator<InputImage> i1It(i1,i1->GetLargestPossibleRegion());
        itk::ImageRegionIterator<InputImage> i2It(i2,i2->GetLargestPossibleRegion());
        for (i1It.GoToBegin(),i2It.GoToBegin();!i1It.IsAtEnd();++i1It,++i2It){
            int val=i2It.Get();
            if (val)
                i1It.Set(val*label);
        }
    }

    static OutputImagePointer fillHoles(InputImagePointer img){
        typedef typename itk::GrayscaleFillholeImageFilter<InputImage,OutputImage> FilterType;
        typedef typename FilterType::Pointer FilterPointer;
        FilterPointer filter=FilterType::New();
        filter->SetInput(img);
        filter->Update();
        return filter->GetOutput();

    }

    
};
