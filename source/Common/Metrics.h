/**
 * @file   Metrics.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:26:25 2015
 * 
 * @brief  Collection of image comparison metrics as static functions
 * 
 * 
 */

#pragma once



#include "FilterUtils.hpp"
#include "TransformationUtils.h"
#include "itkRecursiveGaussianImageFilter.h"
#include <itkMeanImageFilter.h>
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
//#include <itkBoxMeanImageFilter.h>
#include "itkSubtractAbsImageFilter.h"
#include <itkAbsoluteValueDifferenceImageFilter.h>
#ifdef WITH_MIND
#include "dataCostSSC.h"
#include "dataCostLCC.h"
#endif


template<typename InputImage, typename OutputImage = InputImage, typename InternalPrecision=double>
class Metrics{

    typedef typename InputImage::Pointer  InputImagePointer;
    typedef typename InputImage::ConstPointer  ConstInputImagePointer;
    
    typedef typename InputImage::PixelType InputImagePixelType;
    typedef typename InputImage::IndexType InputImageIndex;
    typedef typename InputImage::RegionType InputImageRegion;
    typedef typename InputImage::SpacingType SpacingType;

    typedef typename OutputImage::Pointer OutputImagePointer;
    typedef typename OutputImage::IndexType OutputImageIndex;
    typedef typename OutputImage::RegionType OutputImageRegion;
    typedef typename OutputImage::PixelType OutputImagePixelType;
    typedef typename OutputImage::PointType OutputImagePointType;
    static const int D=InputImage::ImageDimension;
    typedef typename TransfUtils<InputImage>::DeformationFieldType DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    
    typedef typename itk::Image<InternalPrecision,D> InternalImage;
    typedef typename InternalImage::Pointer InternalImagePointer;

public:


    //hellishly inefficient but "clean" implementation of local normalized cross correlation
    static inline OutputImagePointer LNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
        return LNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp);
    }
    static inline OutputImagePointer LNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigma);
        //filter->SetVariance(sigma*sigma);
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        filter->Update();
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        InternalImagePointer i2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput());  
        FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images
        InternalImagePointer i1Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        InternalImagePointer i2Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        InternalImagePointer i1SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1SquareBar.mha",i1SquareBar));
        filter->SetInput(i2Square);
        filter->Update();
        InternalImagePointer i2SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2SquareBar.mha",i2SquareBar));

        //compute squares of local means
        InternalImagePointer i1BarSquare=ImageUtils<InternalImage>::localSquare(i1Bar);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        InternalImagePointer i2BarSquare=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2BarSquare.mha",i2BarSquare));

        //multiply i1 and i2 locally
        InternalImagePointer i1i2=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1i2.mha",i1i2));
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        InternalImagePointer i1Timesi2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));

        //multiply local means
        InternalImagePointer i1BarTimesi2Bar=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Bar,i2Bar);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1BarTimesi2Bar.mha",i1BarTimesi2Bar));


        
        InternalImagePointer numerator=FilterUtils<InternalImage>::substract(i1Timesi2Bar,i1BarTimesi2Bar);
        LOGI(8,ImageUtils<InternalImage>::writeImage("numerator.mha",numerator));

        InternalImagePointer varianceI1=FilterUtils<InternalImage>::substract(i1SquareBar,i1BarSquare); 
        FilterUtils<InternalImage,InternalImage>::lowerThresholding(varianceI1,std::numeric_limits<InternalPrecision>::epsilon());
        LOGI(8,ImageUtils<InternalImage>::writeImage("varianceI1.mha",varianceI1));
       
        InternalImagePointer varianceI2=FilterUtils<InternalImage>::substract(i2SquareBar,i2BarSquare);
        FilterUtils<InternalImage,InternalImage>::lowerThresholding(varianceI2,std::numeric_limits<InternalPrecision>::epsilon());

        LOGI(8,ImageUtils<InternalImage>::writeImage("varianceI2.mha",varianceI2));
        
        ImageUtils<InternalImage>::sqrtImage(varianceI1);
        ImageUtils<InternalImage>::sqrtImage(varianceI2);

        InternalImagePointer denominator=ImageUtils<InternalImage>::multiplyImageOutOfPlace(varianceI1,varianceI2);

        InternalImagePointer result=ImageUtils<InternalImage>::divideImageOutOfPlace(numerator,denominator);
        //convert to weights
        
        itk::ImageRegionIterator<InternalImage> it(result,result->GetLargestPossibleRegion());
        it.GoToBegin();
        for (;!it.IsAtEnd();++it){
            InternalPrecision r = it.Get();
            if (r< -(InternalPrecision)1.0 ){
                r = 0;
            }else if ( r > (InternalPrecision)1.0){
                r = 0;
            }
            r = pow((r+1.0)/2,exp);
            
            r = max(r,std::numeric_limits<InternalPrecision>::epsilon());
            it.Set(r);
            
        }
        // LOGI(8,ImageUtils<InternalImage>::add(result,1.0));
        //        ImageUtils<InternalImage>::multiplyImage(result,0.5);
        //        FilterUtils<InternalImage,InternalImage>::thresholding(result,std::numeric_limits<InternalPrecision>::epsilon(),1.0);

        LOGI(10,ImageUtils<InternalImage>::writeImage("result.mha",result));

        return FilterUtils<InternalImage,OutputImage>::cast(result);

    }
   
    static inline OutputImagePointer LSSDNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //filter->SetVariance(sigmaWidth*sigmaWidth);

        InternalImagePointer diff = FilterUtils<InternalImage>::substract(i1Cast,i2Cast);
        InternalImagePointer diffSquare = ImageUtils<InternalImage>::localSquare(diff);

        if (sigmaNorm==0.0){
            itk::ImageRegionIterator<InternalImage> it(diffSquare,diffSquare->GetLargestPossibleRegion());
            itk::ImageRegionIterator<InternalImage> it2(i1Cast,diffSquare->GetLargestPossibleRegion());
            it.GoToBegin();it2.GoToBegin();
            InternalPrecision maxVal=FilterUtils<InternalImage>::getMax(diffSquare);
            sigmaNorm=-0.005*maxVal/log(0.1);
            LOGV(6)<<VAR(sqrt(sigmaNorm))<<endl;
#if 0
            for (;!it.IsAtEnd();++it,++it2){
                if (true || it2.Get()>minVal){
                    sigmaNorm+=it.Get();
                    ++c;
                }
            }
            sigmaNorm/=c;
            LOGV(6)<<VAR(sigmaNorm)<<endl;
#endif
            //sigmaNorm=(FilterUtils<InternalImage>::getMean(diffSquare));
        }else{
            sigmaNorm*=sigmaNorm;
        }

        //compute local means by concolving with gaussian
        filter->SetInput(diffSquare);
        filter->Update();
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("lssd.nii",i1Bar));
        typename ImageUtils<InternalImage>::ImageIteratorType meanIt(i1Bar,i1Cast->GetLargestPossibleRegion());

        InternalImagePointer result = ImageUtils<InternalImage>::createEmpty(i1Bar);

        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), meanIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++meanIt){
            InternalPrecision d = meanIt.Get();
            resultIt.Set(exp(-0.5 * fabs(d)  / sigmaNorm) );
        }
        
        
        return FilterUtils<InternalImage,OutputImage>::cast(result);
    }
  
   
    static inline OutputImagePointer LSSD(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        return LSSD( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
    static inline OutputImagePointer LSSD(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
      

        //InternalImagePointer diff = FilterUtils<InternalImage>::substract(i1Cast,i2Cast);
        //InternalImagePointer diffSquare = ImageUtils<InternalImage>::localSquare(diff);
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());
        for (i2It.GoToBegin(),i1It.GoToBegin(); i1It.IsAtEnd(); ++i1It, ++i2It){
            InternalPrecision d=(i1It.Get()-i2It.Get());
            i1It.Set(d*d);
        }
#if 0
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        filter->Update();
        return (filter->GetOutput()); 
#elif 1
        typedef typename itk::BoxMeanImageFilter< InternalImage, InternalImage > FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        typename FilterType::RadiusType r;
        r.Fill(sigmaWidth);
        filter->SetRadius(r);
        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        return filter->GetOutput();

#endif
    }

    static inline OutputImagePointer integralSSD(InputImagePointer i1,InputImagePointer i2){
        InternalImagePointer result=FilterUtils<InputImage,InternalImage>::cast(i1);

      

        //InternalImagePointer diff = FilterUtils<InternalImage>::substract(i1Cast,i2Cast);
        //InternalImagePointer diffSquare = ImageUtils<InternalImage>::localSquare(diff);
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(result,result->GetLargestPossibleRegion());
        typename ImageUtils<InputImage>::ImageIteratorType i2It(i2,result->GetLargestPossibleRegion());
        InternalPrecision sum=0.0;
        for (i2It.GoToBegin(),i1It.GoToBegin(); i1It.IsAtEnd(); ++i1It, ++i2It){
            InternalPrecision d=(i1It.Get()-i2It.Get());
            sum+=d*d;
            i1It.Set(sum);
        }
        return result;
    }

    static inline OutputImagePointer integralSAD(InputImagePointer i1,InputImagePointer i2){
        InternalImagePointer result=FilterUtils<InputImage,InternalImage>::cast(i1);

      

        //InternalImagePointer diff = FilterUtils<InternalImage>::substract(i1Cast,i2Cast);
        //InternalImagePointer diffSquare = ImageUtils<InternalImage>::localSquare(diff);
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(result,result->GetLargestPossibleRegion());
        typename ImageUtils<InputImage>::ImageIteratorType i2It(i2,result->GetLargestPossibleRegion());
        InternalPrecision sum=0.0;
        for (i2It.GoToBegin(),i1It.GoToBegin(); i1It.IsAtEnd(); ++i1It, ++i2It){
            InternalPrecision d=(i1It.Get()-i2It.Get());
            sum+=fabs(d);
            i1It.Set(sum);
        }
        return result;
    }

  

    
    static inline InternalImagePointer LSAD(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        return LSAD( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
  static inline InternalImagePointer LSAD(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //filter->SetVariance(sigmaWidth*sigmaWidth);

#if 0
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        
        InternalImagePointer diff =  ImageUtils<InternalImage>::createEmpty(i1Cast);
        typename ImageUtils<InternalImage>::ImageIteratorType diffIt(diff,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());

        InternalPrecision mean=0.0;
        int c=0;
        //subtract and take absolute value
        for (i2It.GoToBegin(),i1It.GoToBegin(), diffIt.GoToBegin(); !diffIt.IsAtEnd(); ++i1It, ++i2It, ++diffIt,++c){
            InternalPrecision d=fabs(i1It.Get()-i2It.Get());
            diffIt.Set(d);
            mean+=d;
        }
        LOGV(6)<<VAR(mean/c)<<endl;

#else
        //typedef typename itk::SubtractAbsImageFilter<InputImage,InputImage,InternalImage> SubFilterType;

        
        typedef typename itk::AbsoluteValueDifferenceImageFilter<InputImage,InputImage,InternalImage> SubFilterType;
        typename SubFilterType::Pointer subFilter=SubFilterType::New();
        subFilter->SetNumberOfThreads( 8 );
        subFilter->SetInput1(i1);
        subFilter->SetInput2(i2);
        subFilter->Update();
        InternalImagePointer diff=subFilter->GetOutput();
        return FilterUtils<InternalImage,OutputImage>::cast(diff);
#endif

        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        InternalImagePointer result=filter->GetOutput();
#if 0
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
       
        InternalImagePointer result = ImageUtils<InternalImage>::createEmpty(i1Bar);

        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), diffIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++diffIt){
            InternalPrecision d = diffIt.Get();
            resultIt.Set(fabs(d));
        }
#endif   
        
        return FilterUtils<InternalImage,OutputImage>::cast(result);
  }

    
 

    static inline OutputImagePointer LSADNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //filter->SetVariance(sigmaWidth*sigmaWidth);
        InternalImagePointer diff =  ImageUtils<InternalImage>::createEmpty(i1Cast);
        typename ImageUtils<InternalImage>::ImageIteratorType diffIt(diff,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());

        InternalPrecision mean=0.0;
        int c=0;
        //subtract and take absolute value
        for (i2It.GoToBegin(),i1It.GoToBegin(), diffIt.GoToBegin(); !diffIt.IsAtEnd(); ++i1It, ++i2It, ++diffIt,++c){
            InternalPrecision d=fabs(i1It.Get()-i2It.Get());
            diffIt.Set(d);
            mean+=d;
        }
        mean/=c;
        LOGV(6)<<VAR(mean)<<endl;

        if (sigmaNorm==0.0){
            sigmaNorm=mean;
        }

        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        typename ImageUtils<InternalImage>::ImageIteratorType meanIt(i1Bar,i1Cast->GetLargestPossibleRegion());

        InternalImagePointer result = ImageUtils<InternalImage>::createEmpty(i1Bar);

        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), meanIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++meanIt){
            InternalPrecision d = meanIt.Get();
            //resultIt.Set(max(0.5,exp(-0.5 * fabs(d)  / sigmaNorm) ));
            resultIt.Set(exp(-0.5 * fabs(d)  / sigmaNorm) );
            //resultIt.Set(pow(exp(-0.5 * fabs(d)  / mean ),sigmaNorm ));
        }
        
        
        return FilterUtils<InternalImage,OutputImage>::cast(result);
    }

  
    static inline OutputImagePointer efficientLNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
        return efficientLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp);
    }
    static inline OutputImagePointer efficientLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        //if (exp == 0.0 ) exp = 1.0;
        if (sigma==0.0) sigma=0.001;
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
#define RECURSIVE

#ifdef RECURSIVE
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef typename itk::RecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
#else
        typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        //typedef typename itk::MeanImageFilter< InternalImage, InternalImage > FilterType;

#endif
        typename FilterType::Pointer filter=FilterType::New();
#ifdef RECURSIVE
        filter->SetSigma(sigma);

        LOGV(5)<<VAR(sigma)<<endl;
#else
        filter->SetVariance(sigma*sigma);
        //filter->SetRadius(sigma);
#endif
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        //filter->InPlaceOff();
        filter->Update();
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput());  

        //FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images


        
        InternalImagePointer i1Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        InternalImagePointer i2Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);


        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i1SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1SquareBar.mha",i1SquareBar));
        i1Square=NULL;
       
        filter->SetInput(i2Square);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i2SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        //FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2SquareBar.mha",i2SquareBar));
        i2Square=NULL;
        //compute squares of local means
        InternalImagePointer i1BarSquare=ImageUtils<InternalImage>::localSquare(i1Bar);
        InternalImagePointer i2BarSquare=ImageUtils<InternalImage>::localSquare(i2Bar);
        //multiply i1 and i2 locally
        InternalImagePointer i1i2=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1i2.mha",i1i2));
        i1Cast=NULL;
        i2Cast=NULL;
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2BarSquare.mha",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i1Timesi2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));
        i1i2=NULL;

        //finish
        InternalImagePointer result=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        itk::ImageRegionIterator<InternalImage> resultIt(result,result->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        
        itk::ImageRegionIterator<InternalImage> i1BarIt(i1Bar,i1Bar->GetLargestPossibleRegion());
        i1BarIt.GoToBegin();
        itk::ImageRegionIterator<InternalImage> i2BarIt(i2Bar,i2Bar->GetLargestPossibleRegion());
        i2BarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i1Timesi2BarIt(i1Timesi2Bar,i1Timesi2Bar->GetLargestPossibleRegion());
        i1Timesi2BarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i1SquareBarIt(i1SquareBar,i1SquareBar->GetLargestPossibleRegion());
        i1SquareBarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i2SquareBarIt(i2SquareBar,i2SquareBar->GetLargestPossibleRegion());
        i2SquareBarIt.GoToBegin();

        
        
        for (;!resultIt.IsAtEnd();++resultIt){

            
            InternalPrecision i1BarV=i1BarIt.Get();
            InternalPrecision i2BarV=i2BarIt.Get();
            InternalPrecision i1BarTimesi2BarV=i1BarV*i2BarV;
         
            
            InternalPrecision i1Timesi2BarV=i1Timesi2BarIt.Get();
            InternalPrecision numeratorV=i1Timesi2BarV-i1BarTimesi2BarV;
#if 1 //def RECURSIVE
            //the recursive filtering is not very exact.
            //this is hardly noticeable when only smoothing single images, but during the computation of variances this leads to errors
            //mostly these errors happen when the variance is very low, eg in smooth regions
            //we correct this errors by thresholding, which looks fine when compared to discrete gaussian filtering.
            InternalPrecision tmp=i1SquareBarIt.Get() - i1BarV*i1BarV;

            if (tmp<0.0){
                LOGV(10)<<VAR(i1SquareBarIt.Get() - i1BarV*i1BarV)<<endl;
                tmp=0.0;
            }
            InternalPrecision varianceI1V=sqrt(tmp);
            tmp=i2SquareBarIt.Get() - i2BarV*i2BarV;
            if (tmp<0.0){
                LOGV(10)<<VAR(i2SquareBarIt.Get() - i2BarV*i2BarV)<<endl;
                tmp=0.0;
            }
            InternalPrecision varianceI2V=sqrt(tmp);
            InternalPrecision denominatorV=varianceI1V*varianceI2V;
            InternalPrecision r = (abs(denominatorV)>1000.0*std::numeric_limits<InternalPrecision>::epsilon())?numeratorV/(denominatorV):0.0;
                        
#else
            InternalPrecision varianceI1V=sqrt((i1SquareBarIt.Get() - i1BarV*i1BarV));
            InternalPrecision varianceI2V=sqrt((i2SquareBarIt.Get() - i2BarV*i2BarV));
            InternalPrecision denominatorV=varianceI1V*varianceI2V;
            InternalPrecision r = (abs(denominatorV)>0.0)?numeratorV/(denominatorV):0.0;

#endif

            ++i1BarIt;
            ++i2BarIt;
            ++i1Timesi2BarIt;
            ++i1SquareBarIt;
            ++i2SquareBarIt;

#ifdef RECURSIVE
            if (r< -(InternalPrecision)1.0 ){
                r = -1.0;
            }else if ( r > (InternalPrecision)1.0){
                r = 1.0;
            }
#endif
            r = pow((r+1.0)/2,exp);
        
#ifdef SAFE    
            r = max(r,std::numeric_limits<InternalPrecision>::epsilon());
#endif
            //LOGV(7)<<"LNCC-local: "<<r<<endl;
            resultIt.Set(r);

        }
        
        LOGI(8,ImageUtils<InternalImage>::writeImage("result.mha",result));

        return FilterUtils<InternalImage,OutputImage>::cast(result);

    }
       static inline OutputImagePointer efficientLNCCNewNorm(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
           return efficientLNCCNewNorm(ConstInputImagePointer(i1),ConstInputImagePointer(i2), sigma, exp);
       }

       static inline OutputImagePointer efficientLNCCNewNorm(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        if (exp == 0.0 ) exp = 1.0;
        if (sigma==0.0) sigma=0.001;
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        //#define RECURSIVE

#ifdef RECURSIVE
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef typename itk::RecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
#else
        typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        //typedef typename itk::MeanImageFilter< InternalImage, InternalImage > FilterType;

#endif
        typename FilterType::Pointer filter=FilterType::New();
#ifdef RECURSIVE
        filter->SetSigma(sigma);

        LOGV(5)<<VAR(sigma)<<endl;
#else
        filter->SetVariance(sigma*sigma);
        //filter->SetRadius(sigma);
#endif
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        //filter->InPlaceOff();
        filter->Update();
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput());  

        //FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images


        
        InternalImagePointer i1Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        InternalImagePointer i2Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);


        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i1SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1SquareBar.mha",i1SquareBar));
        i1Square=NULL;
       
        filter->SetInput(i2Square);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i2SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        //FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2SquareBar.mha",i2SquareBar));
        i2Square=NULL;
        //compute squares of local means
        InternalImagePointer i1BarSquare=ImageUtils<InternalImage>::localSquare(i1Bar);
        InternalImagePointer i2BarSquare=ImageUtils<InternalImage>::localSquare(i2Bar);
        //multiply i1 and i2 locally
        InternalImagePointer i1i2=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1i2.mha",i1i2));
        i1Cast=NULL;
        i2Cast=NULL;
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2BarSquare.mha",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->UpdateLargestPossibleRegion();
        InternalImagePointer i1Timesi2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));
        i1i2=NULL;

        //finish
        InternalImagePointer result=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        itk::ImageRegionIterator<InternalImage> resultIt(result,result->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        
        itk::ImageRegionIterator<InternalImage> i1BarIt(i1Bar,i1Bar->GetLargestPossibleRegion());
        i1BarIt.GoToBegin();
        itk::ImageRegionIterator<InternalImage> i2BarIt(i2Bar,i2Bar->GetLargestPossibleRegion());
        i2BarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i1Timesi2BarIt(i1Timesi2Bar,i1Timesi2Bar->GetLargestPossibleRegion());
        i1Timesi2BarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i1SquareBarIt(i1SquareBar,i1SquareBar->GetLargestPossibleRegion());
        i1SquareBarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i2SquareBarIt(i2SquareBar,i2SquareBar->GetLargestPossibleRegion());
        i2SquareBarIt.GoToBegin();

        
        
        for (;!resultIt.IsAtEnd();++resultIt){

            
            InternalPrecision i1BarV=i1BarIt.Get();
            InternalPrecision i2BarV=i2BarIt.Get();
            InternalPrecision i1BarTimesi2BarV=i1BarV*i2BarV;
         
            
            InternalPrecision i1Timesi2BarV=i1Timesi2BarIt.Get();
            InternalPrecision numeratorV=i1Timesi2BarV-i1BarTimesi2BarV;
#if 1 //def RECURSIVE
            //the recursive filtering is not very exact.
            //this is hardly noticeable when only smoothing single images, but during the computation of variances this leads to errors
            //mostly these errors happen when the variance is very low, eg in smooth regions
            //we correct this errors by thresholding, which looks fine when compared to discrete gaussian filtering.
            InternalPrecision tmp=i1SquareBarIt.Get() - i1BarV*i1BarV;

            if (tmp<0){
                LOGV(10)<<VAR(i1SquareBarIt.Get() - i1BarV*i1BarV)<<endl;
                tmp=0;
            }
            InternalPrecision varianceI1V=sqrt(tmp);
            tmp=i2SquareBarIt.Get() - i2BarV*i2BarV;
            if (tmp<0){
                LOGV(10)<<VAR(i2SquareBarIt.Get() - i2BarV*i2BarV)<<endl;
                tmp=0;
            }
            InternalPrecision varianceI2V=sqrt(tmp);
            InternalPrecision denominatorV=varianceI1V*varianceI2V;
            InternalPrecision r = (abs(denominatorV)!=0.0)?numeratorV/(denominatorV):0.0;
                        
#else
            InternalPrecision varianceI1V=sqrt((i1SquareBarIt.Get() - i1BarV*i1BarV));
            InternalPrecision varianceI2V=sqrt((i2SquareBarIt.Get() - i2BarV*i2BarV));
            InternalPrecision denominatorV=varianceI1V*varianceI2V;
            InternalPrecision r = (abs(denominatorV)>0.0)?numeratorV/(denominatorV):0.0;

#endif

            ++i1BarIt;
            ++i2BarIt;
            ++i1Timesi2BarIt;
            ++i1SquareBarIt;
            ++i2SquareBarIt;

#ifdef RECURSIVE
            if (r< -(InternalPrecision)1.0 ){
                r = 0;
            }else if ( r > (InternalPrecision)1.0){
                r = 0;
            }
#endif
            //r = pow(fabs(r),exp);
            //r = max((1.0/(min(1.0-pow(fabs(r),exp),0.00001))),0.0);
            r = (1.0/
                 (1.0
                  -pow(min(0.9999,fabs(r)),exp)
                  )
                 );
        
#ifdef SAFE    
            r = max(r,std::numeric_limits<InternalPrecision>::epsilon());
#endif
            resultIt.Set(r);

        }
        
        LOGI(8,ImageUtils<InternalImage>::writeImage("result.mha",result));

        return FilterUtils<InternalImage,OutputImage>::cast(result);

    }

    static inline OutputImagePointer coarseLNCC(InputImagePointer i1,InputImagePointer i2, InputImagePointer coarseImg,double sigma=1.0, double exp = 1.0){
        return coarseLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, coarseImg,sigma,exp);
    }
    static inline OutputImagePointer coarseLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2, InputImagePointer coarseImg,double sigma=1.0, double exp = 1.0){
        if (exp == 0.0 ) exp = 1.0;
        if (sigma==0.0) sigma=0.001;
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();

        filter->SetSigma(sigma);
        //filter->SetVariance(sigma*sigma);
        //        filter->SetMaximumKernelWidth(sigma);


        //compute local means by concolving with gaussian
        filter->SetInput(i1Cast);
        filter->Update();
        InternalImagePointer i1Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        InternalImagePointer i2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput());  

        //FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images


        
        InternalImagePointer i1Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        InternalImagePointer i2Square=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        InternalImagePointer i1SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1SquareBar.mha",i1SquareBar));
       
        filter->SetInput(i2Square);
        filter->Update();
        InternalImagePointer i2SquareBar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        FilterUtils<InternalImage,InternalImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2SquareBar.mha",i2SquareBar));

        //compute squares of local means
        InternalImagePointer i1BarSquare=ImageUtils<InternalImage>::localSquare(i1Bar);
        InternalImagePointer i2BarSquare=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        //multiply i1 and i2 locally
        InternalImagePointer i1i2=ImageUtils<InternalImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1i2.mha",i1i2));

        LOGI(8,ImageUtils<InternalImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        LOGI(8,ImageUtils<InternalImage>::writeImage("i2BarSquare.mha",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        InternalImagePointer i1Timesi2Bar=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<InternalImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));

        //finish
        InternalImagePointer result=ImageUtils<InternalImage>::duplicate(filter->GetOutput()); 
        itk::ImageRegionIterator<InternalImage> resultIt(result,result->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        
        itk::ImageRegionIterator<InternalImage> i1BarIt(i1Bar,i1Bar->GetLargestPossibleRegion());
        i1BarIt.GoToBegin();
        itk::ImageRegionIterator<InternalImage> i2BarIt(i2Bar,i2Bar->GetLargestPossibleRegion());
        i2BarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i1Timesi2BarIt(i1Timesi2Bar,i1Timesi2Bar->GetLargestPossibleRegion());
        i1Timesi2BarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i1SquareBarIt(i1SquareBar,i1SquareBar->GetLargestPossibleRegion());
        i1SquareBarIt.GoToBegin();

        itk::ImageRegionIterator<InternalImage> i2SquareBarIt(i2SquareBar,i2SquareBar->GetLargestPossibleRegion());
        i2SquareBarIt.GoToBegin();

        
        
        for (;!resultIt.IsAtEnd();++resultIt){

            
            InternalPrecision i1BarV=i1BarIt.Get();
            InternalPrecision i2BarV=i2BarIt.Get();
            InternalPrecision i1BarTimesi2BarV=i1BarV*i2BarV;
         
            
            InternalPrecision i1Timesi2BarV=i1Timesi2BarIt.Get();
            InternalPrecision numeratorV=i1Timesi2BarV-i1BarTimesi2BarV;

            InternalPrecision varianceI1V=sqrt(max(0.0,i1SquareBarIt.Get() - i1BarV*i1BarV));
            InternalPrecision varianceI2V=sqrt(max(0.0,i2SquareBarIt.Get() - i2BarV*i2BarV));
            InternalPrecision denominatorV=varianceI1V*varianceI2V;

            InternalPrecision r = numeratorV/(denominatorV+0.00000000001);

            ++i1BarIt;
            ++i2BarIt;
            ++i1Timesi2BarIt;
            ++i1SquareBarIt;
            ++i2SquareBarIt;


            if (r< -(InternalPrecision)1.0 ){
                r = 0;
            }else if ( r > (InternalPrecision)1.0){
                r = 0;
            }
            r = pow((r+1.0)/2,exp);
            
            r = max(r,std::numeric_limits<InternalPrecision>::epsilon());

            resultIt.Set(r);

        }
        
        LOGI(8,ImageUtils<InternalImage>::writeImage("result.mha",result));

        return FilterUtils<InternalImage,OutputImage>::cast(result);

    }


    static inline OutputImagePointer ITKLMI(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0, InputImagePointer coarseImg=NULL){
        return ITKLMI( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp,coarseImg);
    }
    static inline OutputImagePointer ITKLMI(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0, InputImagePointer coarseImg=NULL){
        
        InternalImagePointer result;
        if (coarseImg.IsNotNull()){
            result=FilterUtils<InputImage,InternalImage>::createEmpty(coarseImg);
            //LOGV(6)<<VAR(result->GetSpacing()[0])<<" "<<VAR(i1->GetSpacing()[0])<<endl;
            //sigma=sigma*result->GetSpacing()[0]/i1->GetSpacing()[0];
        }
        else{
            result=FilterUtils<InputImage,InternalImage>::createEmpty(i1);
        }
        result->FillBuffer(0.0);
        typedef typename InputImage::RegionType RegionType;
        RegionType region;
        //LOGV(4)<<VAR(sigma)<<endl;
        typename InputImage::SizeType regionSize;
        regionSize.Fill(2*(2*sigma+1)); //needs some spacing information!?
        region.SetSize(regionSize);
        itk::ImageRegionIteratorWithIndex<InternalImage> resultIt(result,result->GetLargestPossibleRegion());


        typedef typename itk::MattesMutualInformationImageToImageMetric<InputImage,InputImage> MIType;
        typename MIType::Pointer nccMetric = MIType::New();
        typedef typename itk::LinearInterpolateImageFunction<InputImage> InterpolatorType;
        typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
        interpolator->SetInputImage(i2);
        nccMetric->SetFixedImage(i1);
        nccMetric->SetMovingImage(i2);
        nccMetric->SetInterpolator(interpolator);
        typename itk::IdentityTransform< double, D >::Pointer iTrans= itk::IdentityTransform< double, D >::New();
        nccMetric->SetTransform(iTrans);
        //nccMetric->SetSubtractMean(true);

	resultIt.GoToBegin();
        for (;!resultIt.IsAtEnd();++resultIt){
            
            typename InputImage::IndexType idx = resultIt.GetIndex(), newIndex;
            bool fullInside=true;
            typename InputImage::PointType pt;
            //get coordinate in fine image in case we're simultaneously downsampling
            if (coarseImg.IsNotNull()){
                result->TransformIndexToPhysicalPoint(idx,pt);
                i1->TransformPhysicalPointToIndex(pt,newIndex);
            }else{
                newIndex=idx;
            }
            LOGV(9)<<VAR(idx)<<" "<<VAR(newIndex)<<" "<<VAR(sigma)<<endl;

#if 0            
            //convert newIndex to corner of patch, assumind newIndex is the central pixel
            for (int d=0;d<D;++d){
                if (newIndex[d]>=i1->GetLargestPossibleRegion().GetSize()[d]-sigma){
                    fullInside=false;
                    break;
                }
                newIndex[d]-=sigma;
                if (newIndex[d]<0){
                    fullInside=false;
                    break;
                }
            }
#else

            typename InputImage::SizeType localRegionSize;
             //convert newIndex to corner of patch, assumind newIndex is the central pixel
            for (int d=0;d<D;++d){
                //make region smaller
                int maxSize=2*sigma+1;
                
                //size gets smaller when center pixel is outside maxRange-sigma
                int idxDifference=newIndex[d] - (i1->GetLargestPossibleRegion().GetSize()[d]-sigma -1);
                maxSize-=max(0, idxDifference);
                
                //size gets smaller when newIndex-sigma is smaller than zero
                idxDifference=sigma-newIndex[d];
                maxSize-=max(0, idxDifference);

                localRegionSize[d]=maxSize;
                newIndex[d]=max(0.0,newIndex[d]-sigma);

            }
            region.SetSize(localRegionSize);
#endif

            if (fullInside){
                region.SetIndex(newIndex);
                nccMetric->SetFixedImageRegion(region);
                InternalPrecision val=(1.0-nccMetric->GetValue(iTrans->GetParameters()))/2;
                LOGV(9)<<VAR(newIndex)<<" "<<VAR(pt)<<" " <<VAR(val)<<" "<<endl;
                resultIt.Set(pow(val,exp));
            }
            
        }
        return FilterUtils<InternalImage,OutputImage>::cast(result);

    }

    static inline OutputImagePointer ITKLNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0, InputImagePointer coarseImg=NULL){
        return ITKLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp,coarseImg);
    }
    static inline OutputImagePointer ITKLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0, InputImagePointer coarseImg=NULL){
        
        InternalImagePointer result;
        if (coarseImg.IsNotNull()){
            result=FilterUtils<InputImage,InternalImage>::createEmpty(coarseImg);
            //LOGV(6)<<VAR(result->GetSpacing()[0])<<" "<<VAR(i1->GetSpacing()[0])<<endl;
            //sigma=sigma*result->GetSpacing()[0]/i1->GetSpacing()[0];
        }
        else{
            result=FilterUtils<InputImage,InternalImage>::createEmpty(i1);
        }
        result->FillBuffer(0.0);
        typedef typename InputImage::RegionType RegionType;
        RegionType region;
        //LOGV(4)<<VAR(sigma)<<endl;
        typename InputImage::SizeType regionSize;
        regionSize.Fill(2*(2*sigma+1)); //needs some spacing information!?
        region.SetSize(regionSize);
        itk::ImageRegionIteratorWithIndex<InternalImage> resultIt(result,result->GetLargestPossibleRegion());


        typedef typename itk::NormalizedCorrelationImageToImageMetric<InputImage,InputImage> NCCType;
        typename NCCType::Pointer nccMetric = NCCType::New();
        typedef typename itk::LinearInterpolateImageFunction<InputImage> InterpolatorType;
        typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
        interpolator->SetInputImage(i2);
        nccMetric->SetFixedImage(i1);
        nccMetric->SetMovingImage(i2);
        nccMetric->SetInterpolator(interpolator);
        typename itk::IdentityTransform< double, D >::Pointer iTrans= itk::IdentityTransform< double, D >::New();
        nccMetric->SetTransform(iTrans);
        nccMetric->SetSubtractMean(true);

	resultIt.GoToBegin();
        for (;!resultIt.IsAtEnd();++resultIt){
            
            typename InputImage::IndexType idx = resultIt.GetIndex(), newIndex;
            bool fullInside=true;
            typename InputImage::PointType pt;
            //get coordinate in fine image in case we're simultaneously downsampling
            if (coarseImg.IsNotNull()){
                result->TransformIndexToPhysicalPoint(idx,pt);
                i1->TransformPhysicalPointToIndex(pt,newIndex);
            }else{
                newIndex=idx;
            }
            LOGV(9)<<VAR(idx)<<" "<<VAR(newIndex)<<" "<<VAR(sigma)<<endl;

#if 0            
            //convert newIndex to corner of patch, assumind newIndex is the central pixel
            for (int d=0;d<D;++d){
                if (newIndex[d]>=i1->GetLargestPossibleRegion().GetSize()[d]-sigma){
                    fullInside=false;
                    break;
                }
                newIndex[d]-=sigma;
                if (newIndex[d]<0){
                    fullInside=false;
                    break;
                }
            }
#else

            typename InputImage::SizeType localRegionSize;
             //convert newIndex to corner of patch, assumind newIndex is the central pixel
            for (int d=0;d<D;++d){
                //make region smaller
                int maxSize=2*sigma+1;
                
                //size gets smaller when center pixel is outside maxRange-sigma
                int idxDifference=newIndex[d] - (i1->GetLargestPossibleRegion().GetSize()[d]-sigma -1);
                maxSize-=max(0, idxDifference);
                
                //size gets smaller when newIndex-sigma is smaller than zero
                idxDifference=sigma-newIndex[d];
                maxSize-=max(0, idxDifference);

                localRegionSize[d]=maxSize;
                newIndex[d]=max(0.0,newIndex[d]-sigma);

            }
            region.SetSize(localRegionSize);
#endif

            if (fullInside){
                region.SetIndex(newIndex);
                nccMetric->SetFixedImageRegion(region);
                InternalPrecision val=(1.0-nccMetric->GetValue(iTrans->GetParameters()))/2;
                LOGV(9)<<VAR(newIndex)<<" "<<VAR(pt)<<" " <<VAR(val)<<" "<<endl;
                resultIt.Set(pow(val,exp));
            }
            
        }
        return FilterUtils<InternalImage,OutputImage>::cast(result);

    }


    static OutputImagePointer localMetricAutocorrelation(InputImagePointer img1, InputImagePointer img2,double sigma,int nSamples,string metric, double expo=1.0){
        
        InternalImagePointer lncc=efficientLNCC(img1,img2,sigma);
        InternalImagePointer centerMetric; 
        string acc="max";
        string eval="diff";
        if (metric == "lncc"){
            centerMetric = lncc;//efficientLNCC(img1,img2,sigma);
        }else if (metric == "itklncc"){
            centerMetric = ITKLNCC(img1,img2,sigma);
        }else if (metric == "lsad"){
            centerMetric = LSADNorm(img1,img2,sigma,0.0);
        }else if (metric == "lssd"){
            centerMetric = LSSDNorm(img1,img2,sigma,0.0);
        }else{
            std::cout<<"Unknown metric "<<metric<<std::endl;
            exit(0);
        }

        InternalImagePointer accumulator=ImageUtils<InternalImage>::createEmpty(centerMetric);
        accumulator->FillBuffer(0.0);

        int nTotalSamples=1;
        std::vector<InternalImagePointer> metrics(nTotalSamples,NULL);

        DeformationFieldPointerType def=TransfUtils<InputImage>::createEmpty(img2);

    
        int count=0;
        for (int d=0;d<D;++d){

            for (int sign=-1;sign<2;sign+=2){
                for (int sample=0;sample<nSamples;++sample){
                
                    InternalPrecision defMag=sign*pow(2,sample)*sigma;
                    typename TransfUtils<InputImage>::DisplacementType disp;
                    disp.Fill(0);
                    disp[d]=defMag;
                    def->FillBuffer(disp);
                    InputImagePointer warped=TransfUtils<InputImage>::warpImage(img1,def);
                    if (metric == "lncc"){
                        metrics[count] = efficientLNCC(warped,img2,sigma);
                    }else if (metric == "itklncc"){
                        metrics[count] = ITKLNCC(warped,img2,sigma);
                    }else if (metric == "lsad"){
                        metrics[count] = LSADNorm(warped,img2,sigma,0.0);
                    }else if (metric == "lssd"){
                        metrics[count] = LSSDNorm(warped,img2,sigma,0.0);
                    }else{
                        std::cout<<"Unknown metric "<<metric<<std::endl;
                        exit(0);
                    }

                    if (acc=="mean"){
                        accumulator=FilterUtils<InternalImage>::add(accumulator,metrics[count]);
                    }else if (acc=="max"){
                        FilterUtils<InternalImage>::localMax(accumulator,metrics[count]);
                    }else{
                        LOG<<"unknown accumulation method "<<acc<<endl;
                        exit(0);
                    }

                    //++count;
                                                         
                }
            }
        }
        if (acc == "mean")
            ImageUtils<InternalImage>::multiplyImage(accumulator,1.0/(D*2*nSamples));

        if (eval=="diff"){
            // l= 0.5 + (\hat l - mu)/(2*\hat l)
            //InternalImagePointer diff=FilterUtils<InternalImage>::substract(centerMetric,accumulator);
            //centerMetric=FilterUtils<InternalImage>::lowerThresholding(centerMetric,1e-10);
            //centerMetric=ImageUtils<InternalImage>::divideImageOutOfPlace(diff,centerMetric);
            //ImageUtils<InternalImage>::multiplyImage(centerMetric,0.5);

            //ImageUtils<InternalImage>::add(centerMetric,0.5);
            centerMetric=ImageUtils<InternalImage>::divideImageOutOfPlace(centerMetric,accumulator);
            centerMetric=ImageUtils<InternalImage>::multiplyImageOutOfPlace(lncc,centerMetric);
            //centerMetric=FilterUtils<InternalImage>::thresholding(centerMetric,0,1);
            typename ImageUtils<InternalImage>::ImageIteratorType resultIt(centerMetric,centerMetric->GetLargestPossibleRegion());
#if 1 
            if (expo !=1.0){
                for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt){
                    resultIt.Set(pow(resultIt.Get(),expo));
                }
            }
#endif

        }else if (eval == "exp" ){
        
            
            typename ImageUtils<InternalImage>::ImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
            typename ImageUtils<InternalImage>::ImageIteratorType resultIt(centerMetric,centerMetric->GetLargestPossibleRegion());
        
            accIt.GoToBegin();
            for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt,++accIt){
                InternalPrecision Lprime=resultIt.Get();
                InternalPrecision Lbar=accIt.Get();
            
                InternalPrecision weight;
                if (Lprime != 0.0)
                    weight = exp (- 0.6931 * Lbar/Lprime);
                else
                    weight = 0.0;
                resultIt.Set(weight);

            }

        }else{
            LOG<<"Unknown evaluation method "<<eval<<endl;
            exit(0);
        }
        return centerMetric;
    }


    static double nCC(InputImagePointer img1, InputImagePointer img2, DeformationFieldPointerType def){
        
        InputImagePointer warpedImage2=TransfUtils<InputImage>::warpImage(img2,def);
        typedef typename itk::NormalizedCorrelationImageToImageMetric<InputImage,InputImage> NCCType;
        typename NCCType::Pointer nccMetric = NCCType::New();
        typedef typename itk::LinearInterpolateImageFunction<InputImage> InterpolatorType;
        typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
        interpolator->SetInputImage(warpedImage2);
        nccMetric->SetFixedImage(img1);
        nccMetric->SetMovingImage(warpedImage2);
        nccMetric->SetInterpolator(interpolator);
        typename itk::IdentityTransform< double, D >::Pointer iTrans= itk::IdentityTransform< double, D >::New();
        nccMetric->SetTransform(iTrans);
        nccMetric->SetSubtractMean(true);
        nccMetric->SetFixedImageRegion(img1->GetLargestPossibleRegion());
        return nccMetric->GetValue(iTrans->GetParameters());

    }

  static double nCC(InputImagePointer img1, InputImagePointer warpedImage2){
        
      typedef typename itk::NormalizedCorrelationImageToImageMetric<InputImage,InputImage> NCCType;
        typename NCCType::Pointer nccMetric = NCCType::New();
        typedef typename itk::LinearInterpolateImageFunction<InputImage> InterpolatorType;
        typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
        interpolator->SetInputImage(warpedImage2);
        nccMetric->SetFixedImage(img1);
        nccMetric->SetMovingImage(warpedImage2);
        nccMetric->SetInterpolator(interpolator);
        typename itk::IdentityTransform< double, D >::Pointer iTrans= itk::IdentityTransform< double, D >::New();
        nccMetric->SetTransform(iTrans);
        nccMetric->SetSubtractMean(true);
        nccMetric->SetFixedImageRegion(img1->GetLargestPossibleRegion());
        return nccMetric->GetValue(iTrans->GetParameters());

    }

      static double msd(InputImagePointer img1, InputImagePointer warpedImage2){
        
           itk::ImageRegionIterator<InputImage> it1(img1,img1->GetRequestedRegion());
      itk::ImageRegionIterator<InputImage> it2(warpedImage2,warpedImage2->GetRequestedRegion());
      double result=0.0;
      int c=0;
      it1.GoToBegin();
      it2.GoToBegin();
      for (;!it1.IsAtEnd();++it1,++it2,++c){
          double val=(it1.Get()-it2.Get());
          val*=val;
          result+=val;
      }
      return result/c; 

    }

  static double mad(InputImagePointer img1, InputImagePointer warpedImage2){
      itk::ImageRegionIterator<InputImage> it1(img1,img1->GetRequestedRegion());
      itk::ImageRegionIterator<InputImage> it2(warpedImage2,warpedImage2->GetRequestedRegion());
      double result=0.0;
      int c=0;
      it1.GoToBegin();
      it2.GoToBegin();
      for (;!it1.IsAtEnd();++it1,++it2,++c){
          result+=fabs(it1.Get()-it2.Get());
      }
      return result/c;

    }


       static inline InternalImagePointer CategoricalDiff(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        return CategoricalDiff( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
    static inline InternalImagePointer CategoricalDiff(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        //typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        //filter->SetSigma(sigmaWidth);
        filter->SetVariance(sigmaWidth*sigmaWidth);

        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        
        InternalImagePointer diff =  ImageUtils<InternalImage>::createEmpty(i1Cast);
        typename ImageUtils<InternalImage>::ImageIteratorType diffIt(diff,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());

        InternalPrecision mean=0.0;
        int c=0;
        //subtract and take absolute value
        for (i2It.GoToBegin(),i1It.GoToBegin(), diffIt.GoToBegin(); !diffIt.IsAtEnd(); ++i1It, ++i2It, ++diffIt,++c){
            InternalPrecision d=fabs(i1It.Get()-i2It.Get());
            diffIt.Set(d<std::numeric_limits<InternalPrecision>::epsilon());
            mean+=d;
        }
        LOGV(6)<<VAR(mean/c)<<endl;



        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        InternalImagePointer result=filter->GetOutput();

        
        return FilterUtils<InternalImage,OutputImage>::cast(result);
  }
    static inline InternalImagePointer CategoricalDiffNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        return CategoricalDiffNorm( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth,sigmaNorm);
    }
    static inline InternalImagePointer CategoricalDiffNorm(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0,double sigmaNorm=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        //typedef typename itk::SmoothingRecursiveGaussianImageFilter< InternalImage, InternalImage > FilterType;
        typedef itk::DiscreteGaussianImageFilter<InternalImage,InternalImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        //filter->SetSigma(sigmaWidth);
        filter->SetVariance(sigmaWidth*sigmaWidth);

        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        
        InternalImagePointer diff =  ImageUtils<InternalImage>::createEmpty(i1Cast);
        typename ImageUtils<InternalImage>::ImageIteratorType diffIt(diff,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<InternalImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());

        InternalPrecision mean=0.0;
        int c=0;
        //subtract and take absolute value
        for (i2It.GoToBegin(),i1It.GoToBegin(), diffIt.GoToBegin(); !diffIt.IsAtEnd(); ++i1It, ++i2It, ++diffIt,++c){
            InternalPrecision d=fabs(i1It.Get()-i2It.Get());
            diffIt.Set(d<std::numeric_limits<InternalPrecision>::epsilon());
            mean+=d;
        }
        LOGV(6)<<VAR(mean/c)<<endl;



        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        InternalImagePointer result=filter->GetOutput();
        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt){
            InternalPrecision d = resultIt.Get();
            //resultIt.Set(max(0.5,exp(-0.5 * fabs(d)  / sigmaNorm) ));
            resultIt.Set(exp(-0.5 * fabs(d)  / sigmaNorm) );
            LOGV(7)<<VAR(d)<<" "<<resultIt.Get()<<endl;
            //resultIt.Set(pow(exp(-0.5 * fabs(d)  / mean ),sigmaNorm ));
        }
        
        return FilterUtils<InternalImage,OutputImage>::cast(result);
    }
#ifdef WITH_MIND
    static inline OutputImagePointer deedsMIND(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0,double sigmaNorm=1.0){
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        float * i1Data=i1Cast->GetBufferPointer();
        float * i2Data=i2Cast->GetBufferPointer();
        InternalImagePointer result=ImageUtils<InternalImage>::createEmpty(i1Cast);
        float * resultData=result->GetBufferPointer();
        typename InputImage::SizeType size=i1->GetLargestPossibleRegion().GetSize();
        int hw = 0;
        int sparse=1;
        int r=int(sigmaWidth);
        dataRegSSC(resultData,i1Data,i2Data,hw,sparse,r,0,size[0],size[1],D>2?size[2]:1);
        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());
        for (resultIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt){
            InternalPrecision d = resultIt.Get();
            resultIt.Set(InternalPrecision(1.0)-pow(min(d,InternalPrecision(1.0)),InternalPrecision(sigmaNorm)));
            //resultIt.Set(-d);
            //resultIt.Set(pow(exp(-0.5 * fabs(d)  / mean ),sigmaNorm ));
        }
        return FilterUtils<InternalImage,OutputImage>::cast(result);
    }
    static inline OutputImagePointer deedsLCC(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0,double sigmaNorm=1.0){
        InternalImagePointer i1Cast=FilterUtils<InputImage,InternalImage>::cast(i1);
        InternalImagePointer i2Cast=FilterUtils<InputImage,InternalImage>::cast(i2);
        float * i1Data=i1Cast->GetBufferPointer();
        float * i2Data=i2Cast->GetBufferPointer();
        InternalImagePointer result=ImageUtils<InternalImage>::createEmpty(i1Cast);
        float * resultData=result->GetBufferPointer();
        typename InputImage::SizeType size=i1->GetLargestPossibleRegion().GetSize();
        int hw = 0;
        int sparse=1;
        int r=int(sigmaWidth);
        dataRegLCC(resultData,i1Data,i2Data,hw,sparse,r,0,size[0],size[1],D>2?size[2]:1);
        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());
        for (resultIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt){
            InternalPrecision d = resultIt.Get();
            resultIt.Set(pow(1.0-d,InternalPrecision(sigmaNorm)));
            //resultIt.Set(pow(exp(-0.5 * fabs(d)  / mean ),sigmaNorm ));
        }
        return FilterUtils<InternalImage,OutputImage>::cast(result);
    }
#endif
    static inline OutputImagePointer multiScaleLNCCAbs(InputImagePointer i1,InputImagePointer i2,double sigmaMax=1.0, double exp = 1.0){
        double sigma=1.0;
        int count=0;
        InternalImagePointer result;
        for (;sigma<=sigmaMax;sigma*=2){
            if (!count)
                result=efficientLNCCNewNorm(i1,i2,sigma,1.0);
            else
                result=FilterUtils<InternalImage>::add(result,efficientLNCCNewNorm(i1,i2,sigma,1.0));
            ++count;

        }
        typename ImageUtils<InternalImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());
        for (resultIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt){
            InternalPrecision d = resultIt.Get();
            resultIt.Set(pow(d/count,InternalPrecision(exp)));
        }
        return FilterUtils<InternalImage,OutputImage>::cast(result);
        
    }
};
