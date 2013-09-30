#pragma once



#include "FilterUtils.hpp"
#include "TransformationUtils.h"
#include "itkRecursiveGaussianImageFilter.h"
#include <itkMeanImageFilter.h>


using namespace std;

template<class InputImage, class OutputImage = InputImage>
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

public:


    //hellishly inefficient but "clean" implementation of local normalized cross correlation
    static inline OutputImagePointer LNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
        return LNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp);
    }
    static inline OutputImagePointer LNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
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
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        OutputImagePointer i2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput());  
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images
        OutputImagePointer i1Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        OutputImagePointer i2Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        OutputImagePointer i1SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1SquareBar.mha",i1SquareBar));
        filter->SetInput(i2Square);
        filter->Update();
        OutputImagePointer i2SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2SquareBar.mha",i2SquareBar));

        //compute squares of local means
        OutputImagePointer i1BarSquare=ImageUtils<OutputImage>::localSquare(i1Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        OutputImagePointer i2BarSquare=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2BarSquare.mha",i2BarSquare));

        //multiply i1 and i2 locally
        OutputImagePointer i1i2=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1i2.mha",i1i2));
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        OutputImagePointer i1Timesi2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));

        //multiply local means
        OutputImagePointer i1BarTimesi2Bar=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Bar,i2Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarTimesi2Bar.mha",i1BarTimesi2Bar));


        
        OutputImagePointer numerator=FilterUtils<OutputImage>::substract(i1Timesi2Bar,i1BarTimesi2Bar);
        LOGI(8,ImageUtils<OutputImage>::writeImage("numerator.mha",numerator));

        OutputImagePointer varianceI1=FilterUtils<OutputImage>::substract(i1SquareBar,i1BarSquare); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(varianceI1,std::numeric_limits<OutputImagePixelType>::epsilon());
        LOGI(8,ImageUtils<OutputImage>::writeImage("varianceI1.mha",varianceI1));
       
        OutputImagePointer varianceI2=FilterUtils<OutputImage>::substract(i2SquareBar,i2BarSquare);
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(varianceI2,std::numeric_limits<OutputImagePixelType>::epsilon());

        LOGI(8,ImageUtils<OutputImage>::writeImage("varianceI2.mha",varianceI2));
        
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

        LOGI(10,ImageUtils<OutputImage>::writeImage("result.mha",result));

        return result;

    }
   
    static inline OutputImagePointer LSSDNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //filter->SetVariance(sigmaWidth*sigmaWidth);

        OutputImagePointer diff = FilterUtils<OutputImage>::substract(i1Cast,i2Cast);
        OutputImagePointer diffSquare = ImageUtils<OutputImage>::localSquare(diff);

        if (sigmaNorm==0.0){
            itk::ImageRegionIterator<OutputImage> it(diffSquare,diffSquare->GetLargestPossibleRegion());
            itk::ImageRegionIterator<OutputImage> it2(i1Cast,diffSquare->GetLargestPossibleRegion());
            it.GoToBegin();it2.GoToBegin();
            double minVal=FilterUtils<OutputImage>::getMin(i1Cast);
            int c=0;
            double maxVal=FilterUtils<OutputImage>::getMax(diffSquare);
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
            //sigmaNorm=(FilterUtils<OutputImage>::getMean(diffSquare));
        }else{
            sigmaNorm*=sigmaNorm;
        }

        //compute local means by concolving with gaussian
        filter->SetInput(diffSquare);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("lssd.nii",i1Bar));
        typename ImageUtils<OutputImage>::ImageIteratorType meanIt(i1Bar,i1Cast->GetLargestPossibleRegion());

        OutputImagePointer result = ImageUtils<OutputImage>::createEmpty(i1Bar);

        typename ImageUtils<OutputImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), meanIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++meanIt){
            double d = meanIt.Get();
            resultIt.Set(exp(-0.5 * fabs(d)  / sigmaNorm) );
        }
        
        
        return result;
    }
  
   
    static inline OutputImagePointer LSSD(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        return LSSD( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
    static inline OutputImagePointer LSSD(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
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
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        return LSAD( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigmaWidth);
    }
    static inline OutputImagePointer LSAD(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigmaWidth=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //filter->SetVariance(sigmaWidth*sigmaWidth);

        OutputImagePointer diff =  ImageUtils<OutputImage>::createEmpty(i1Cast);
        typename ImageUtils<OutputImage>::ImageIteratorType diffIt(diff,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());

        double mean=0.0;
        int c=0;
        //subtract and take absolute value
        for (i2It.GoToBegin(),i1It.GoToBegin(), diffIt.GoToBegin(); !diffIt.IsAtEnd(); ++i1It, ++i2It, ++diffIt,++c){
            double d=fabs(i1It.Get()-i2It.Get());
            diffIt.Set(d);
            mean+=d;
        }
        LOGV(6)<<VAR(mean/c)<<endl;

        //compute local means by concolving with gaussian
        filter->SetInput(diff);
        filter->Update();
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
       
        OutputImagePointer result = ImageUtils<OutputImage>::createEmpty(i1Bar);

        typename ImageUtils<OutputImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), diffIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++diffIt){
            double d = diffIt.Get();
            resultIt.Set(fabs(d));
        }
        
        
        return result;
    }
    static inline OutputImagePointer LSADNorm(InputImagePointer i1,InputImagePointer i2,double sigmaWidth=1.0, double sigmaNorm=1.0){
        if (sigmaWidth==0.0) sigmaWidth=0.001;
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        //typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        typename FilterType::Pointer filter=FilterType::New();
        filter->SetSigma(sigmaWidth);
        //filter->SetVariance(sigmaWidth*sigmaWidth);
        OutputImagePointer diff =  ImageUtils<OutputImage>::createEmpty(i1Cast);
        typename ImageUtils<OutputImage>::ImageIteratorType diffIt(diff,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType i1It(i1Cast,i1Cast->GetLargestPossibleRegion());
        typename ImageUtils<OutputImage>::ImageIteratorType i2It(i2Cast,i1Cast->GetLargestPossibleRegion());

        double mean=0.0;
        int c=0;
        //subtract and take absolute value
        for (i2It.GoToBegin(),i1It.GoToBegin(), diffIt.GoToBegin(); !diffIt.IsAtEnd(); ++i1It, ++i2It, ++diffIt,++c){
            double d=fabs(i1It.Get()-i2It.Get());
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
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        typename ImageUtils<OutputImage>::ImageIteratorType meanIt(i1Bar,i1Cast->GetLargestPossibleRegion());

        OutputImagePointer result = ImageUtils<OutputImage>::createEmpty(i1Bar);

        typename ImageUtils<OutputImage>::ImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (resultIt.GoToBegin(), meanIt.GoToBegin(); !resultIt.IsAtEnd(); ++resultIt, ++meanIt){
            double d = meanIt.Get();
            //resultIt.Set(max(0.5,exp(-0.5 * fabs(d)  / sigmaNorm) ));
            resultIt.Set(exp(-0.5 * fabs(d)  / sigmaNorm) );
            //resultIt.Set(pow(exp(-0.5 * fabs(d)  / mean ),sigmaNorm ));
        }
        
        
        return result;
    }

  
    static inline OutputImagePointer efficientLNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0){
        return efficientLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp);
    }
    static inline OutputImagePointer efficientLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0){
        if (exp == 0.0 ) exp == 1.0;
        if (sigma==0.0) sigma=0.001;
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
#define RECURSIVE

#ifdef RECURSIVE
        typedef typename itk::SmoothingRecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
        //typedef typename itk::RecursiveGaussianImageFilter< OutputImage, OutputImage > FilterType;
#else
        typedef itk::DiscreteGaussianImageFilter<OutputImage,OutputImage>  FilterType;
        //typedef typename itk::MeanImageFilter< OutputImage, OutputImage > FilterType;

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
        OutputImagePointer i1Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->UpdateLargestPossibleRegion();
        OutputImagePointer i2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput());  

        //FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images


        
        OutputImagePointer i1Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        OutputImagePointer i2Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);


        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->UpdateLargestPossibleRegion();
        OutputImagePointer i1SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1SquareBar.mha",i1SquareBar));
       
        filter->SetInput(i2Square);
        filter->UpdateLargestPossibleRegion();
        OutputImagePointer i2SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        //FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2SquareBar.mha",i2SquareBar));

        //compute squares of local means
        OutputImagePointer i1BarSquare=ImageUtils<OutputImage>::localSquare(i1Bar);
        OutputImagePointer i2BarSquare=ImageUtils<OutputImage>::localSquare(i2Bar);
        //multiply i1 and i2 locally
        OutputImagePointer i1i2=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1i2.mha",i1i2));

        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2BarSquare.mha",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->UpdateLargestPossibleRegion();
        OutputImagePointer i1Timesi2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));

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
#if 1 //def RECURSIVE
            //the recursive filtering is not very exact.
            //this is hardly noticeable when only smoothing single images, but during the computation of variances this leads to errors
            //mostly these errors happen when the variance is very low, eg in smooth regions
            //we correct this errors by thresholding, which looks fine when compared to discrete gaussian filtering.
            double tmp=i1SquareBarIt.Get() - i1BarV*i1BarV;
            if (tmp<0.0){
                LOGV(10)<<VAR(i1SquareBarIt.Get() - i1BarV*i1BarV)<<endl;
                tmp=0.0;
            }
            double varianceI1V=sqrt(tmp);
            tmp=i2SquareBarIt.Get() - i2BarV*i2BarV;
            if (tmp<0.0){
                LOGV(10)<<VAR(i2SquareBarIt.Get() - i2BarV*i2BarV)<<endl;
                tmp=0.0;
            }
            double varianceI2V=sqrt(tmp);
            double denominatorV=varianceI1V*varianceI2V;
            double r = (abs(denominatorV)>1)?numeratorV/(denominatorV):0.0;
            if (r>0.98){
                LOGV(16)<<VAR(r)<<" " <<VAR(numeratorV)<<" "<<VAR(denominatorV)<<" "<<VAR(varianceI1V)<<" "<<VAR(varianceI2V)<<endl;
            }
#else
            double varianceI1V=sqrt((i1SquareBarIt.Get() - i1BarV*i1BarV));
            double varianceI2V=sqrt((i2SquareBarIt.Get() - i2BarV*i2BarV));
            double denominatorV=varianceI1V*varianceI2V;
            double r = (abs(denominatorV)>0.0)?numeratorV/(denominatorV):0.0;

#endif

            ++i1BarIt;
            ++i2BarIt;
            ++i1Timesi2BarIt;
            ++i1SquareBarIt;
            ++i2SquareBarIt;

#ifdef RECURSIVE
            if (r< -(OutputImagePixelType)1.0 ){
                r = 0;
            }else if ( r > (OutputImagePixelType)1.0){
                r = 0;
            }
#endif
            r = pow((r+1.0)/2,exp);
        
#ifdef SAFE    
            r = max(r,std::numeric_limits<double>::epsilon());
#endif
            resultIt.Set(r);

        }
        
        LOGI(8,ImageUtils<OutputImage>::writeImage("result.mha",result));

        return result;

    }

    static inline OutputImagePointer coarseLNCC(InputImagePointer i1,InputImagePointer i2, InputImagePointer coarseImg,double sigma=1.0, double exp = 1.0){
        return coarseLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, coarseImg,sigma,exp);
    }
    static inline OutputImagePointer coarseLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2, InputImagePointer coarseImg,double sigma=1.0, double exp = 1.0){
        if (exp == 0.0 ) exp == 1.0;
        if (sigma==0.0) sigma=0.001;
        OutputImagePointer i1Cast=FilterUtils<InputImage,OutputImage>::cast(i1);
        OutputImagePointer i2Cast=FilterUtils<InputImage,OutputImage>::cast(i2);
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
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Bar.mha",i1Bar));
        filter->SetInput(i2Cast);
        filter->Update();
        OutputImagePointer i2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput());  

        //FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2Bar,std::numeric_limits<InputImagePixelType>::min());
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Bar.mha",i2Bar));
        //HACK!
        //compute squares of original images


        
        OutputImagePointer i1Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i1Cast);
        OutputImagePointer i2Square=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2Square.mha",i2Square));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Square.mha",i1Square));

        //compute local means of squared images by convolving with gaussian kernel
        filter->SetInput(i1Square);
        filter->Update();
        OutputImagePointer i1SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1SquareBar.mha",i1SquareBar));
       
        filter->SetInput(i2Square);
        filter->Update();
        OutputImagePointer i2SquareBar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        FilterUtils<OutputImage,OutputImage>::lowerThresholding(i2SquareBar,0);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2SquareBar.mha",i2SquareBar));

        //compute squares of local means
        OutputImagePointer i1BarSquare=ImageUtils<OutputImage>::localSquare(i1Bar);
        OutputImagePointer i2BarSquare=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i2Bar,i2Bar);
        //multiply i1 and i2 locally
        OutputImagePointer i1i2=ImageUtils<OutputImage>::multiplyImageOutOfPlace(i1Cast,i2Cast);
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1i2.mha",i1i2));

        LOGI(8,ImageUtils<OutputImage>::writeImage("i1BarSquare.mha",i1BarSquare));
        LOGI(8,ImageUtils<OutputImage>::writeImage("i2BarSquare.mha",i2BarSquare));

      
        //compute local means by convolving...
        filter->SetInput(i1i2);
        filter->Update();
        OutputImagePointer i1Timesi2Bar=ImageUtils<OutputImage>::duplicate(filter->GetOutput()); 
        LOGI(8,ImageUtils<OutputImage>::writeImage("i1Timesi2Bar.mha",i1Timesi2Bar));

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
        
        LOGI(8,ImageUtils<OutputImage>::writeImage("result.mha",result));

        return result;

    }


    static inline OutputImagePointer ITKLNCC(InputImagePointer i1,InputImagePointer i2,double sigma=1.0, double exp = 1.0, InputImagePointer coarseImg=NULL){
        return ITKLNCC( (ConstInputImagePointer)i1, (ConstInputImagePointer)i2, sigma,exp,coarseImg);
    }
    static inline OutputImagePointer ITKLNCC(ConstInputImagePointer i1,ConstInputImagePointer i2,double sigma=1.0, double exp = 1.0, InputImagePointer coarseImg=NULL){
        
        OutputImagePointer result;
        if (coarseImg.IsNotNull()){
            result=FilterUtils<InputImage,OutputImage>::createEmpty(coarseImg);
            //LOGV(6)<<VAR(result->GetSpacing()[0])<<" "<<VAR(i1->GetSpacing()[0])<<endl;
            //sigma=sigma*result->GetSpacing()[0]/i1->GetSpacing()[0];
        }
        else{
            result=FilterUtils<InputImage,OutputImage>::createEmpty(i1);
        }
        result->FillBuffer(0.0);
        typedef typename InputImage::RegionType RegionType;
        RegionType region;
        //LOGV(4)<<VAR(sigma)<<endl;
        typename InputImage::SizeType regionSize;
        regionSize.Fill(2*sigma+1); //needs some spacing information!?
        region.SetSize(regionSize);
        itk::ImageRegionIteratorWithIndex<OutputImage> resultIt(result,result->GetLargestPossibleRegion());


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


        for (resultIt.Begin();!resultIt.IsAtEnd();++resultIt){
            
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
            LOGV(7)<<VAR(idx)<<" "<<VAR(newIndex)<<" "<<VAR(sigma)<<endl;

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
                double val=(1.0-nccMetric->GetValue(iTrans->GetParameters()))/2;
                LOGV(7)<<VAR(newIndex)<<" "<<VAR(pt)<<" " <<VAR(val)<<" "<<endl;
                resultIt.Set(pow(val,exp));
            }
            
        }
        return result;

    }


    static OutputImagePointer localMetricAutocorrelation(InputImagePointer img1, InputImagePointer img2,double sigma,int nSamples,string metric, double expo=1.0){
        
        OutputImagePointer lncc=efficientLNCC(img1,img2,sigma);
        OutputImagePointer centerMetric; 
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

        OutputImagePointer accumulator=ImageUtils<OutputImage>::createEmpty(centerMetric);
        accumulator->FillBuffer(0.0);

        int nTotalSamples=1;
        std::vector<OutputImagePointer> metrics(nTotalSamples,NULL);

        DeformationFieldPointerType def=TransfUtils<InputImage>::createEmpty(img2);

    
        int count=0;
        for (int d=0;d<D;++d){

            for (int sign=-1;sign<2;sign+=2){
                for (int sample=0;sample<nSamples;++sample){
                
                    double defMag=sign*pow(2,sample)*sigma;
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
                        accumulator=FilterUtils<OutputImage>::add(accumulator,metrics[count]);
                    }else if (acc=="max"){
                        FilterUtils<OutputImage>::localMax(accumulator,metrics[count]);
                    }else{
                        LOG<<"unknown accumulation method "<<acc<<endl;
                        exit(0);
                    }

                    //++count;
                                                         
                }
            }
        }
        if (acc == "mean")
            ImageUtils<OutputImage>::multiplyImage(accumulator,1.0/(D*2*nSamples));

        if (eval=="diff"){
            // l= 0.5 + (\hat l - mu)/(2*\hat l)
            //OutputImagePointer diff=FilterUtils<OutputImage>::substract(centerMetric,accumulator);
            //centerMetric=FilterUtils<OutputImage>::lowerThresholding(centerMetric,1e-10);
            //centerMetric=ImageUtils<OutputImage>::divideImageOutOfPlace(diff,centerMetric);
            //ImageUtils<OutputImage>::multiplyImage(centerMetric,0.5);

            //ImageUtils<OutputImage>::add(centerMetric,0.5);
            centerMetric=ImageUtils<OutputImage>::divideImageOutOfPlace(centerMetric,accumulator);
            centerMetric=ImageUtils<OutputImage>::multiplyImageOutOfPlace(lncc,centerMetric);
            //centerMetric=FilterUtils<OutputImage>::thresholding(centerMetric,0,1);
            typename ImageUtils<OutputImage>::ImageIteratorType resultIt(centerMetric,centerMetric->GetLargestPossibleRegion());
#if 1 
            if (expo !=1.0){
                for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt){
                    resultIt.Set(pow(resultIt.Get(),expo));
                }
            }
#endif

        }else if (eval == "exp" ){
        
            
            typename ImageUtils<OutputImage>::ImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
            typename ImageUtils<OutputImage>::ImageIteratorType resultIt(centerMetric,centerMetric->GetLargestPossibleRegion());
        
            accIt.GoToBegin();
            for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt,++accIt){
                double Lprime=resultIt.Get();
                double Lbar=accIt.Get();
            
                double weight;
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
};
