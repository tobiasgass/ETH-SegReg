#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkAntiAliasBinaryImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{
    //feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  double PixelType;
    typedef double OutputPixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::SizeType SizeType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;

    typedef Image<OutputPixelType,D> OutputImageType;
    typedef OutputImageType::Pointer OutputImagePointerType;
 
    argstream * as=new argstream(argc,argv);
    string inFile, outFile, outSurface="";
    double thresh=0.07;
    int iter=20;
    double spacingScalingFactor=2.0;
    bool gaussian=false;
    (*as) >> parameter ("in", inFile, " filename...", true);
    (*as) >> parameter ("out", outFile, " filename...", true);
    (*as) >> parameter ("outSurf", outSurface, " Output Zero Level Set Surface filename...", true);
    (*as) >> parameter ("iter", iter, " number of AA iterations (max)...", false);
    (*as) >> parameter ("thresh", thresh, "max desired RMSE change for convergence...", false);
    (*as) >> parameter ("scale", spacingScalingFactor, "weird...", false);
    (*as) >> option ("gauss", gaussian, "use gaussian smoothing of surface instead of AA filter. thresh becomes size of kernel relative to image spacing");

    (*as) >> help();
    as->defaultErrorHandling();

    ImagePointerType img = FilterUtils<ImageType>::binaryThresholdingLow(ImageUtils<ImageType>::readImage(inFile),1);
    img->DisconnectPipeline();
    OutputImagePointerType out;

    if (!gaussian){
        img->SetSpacing(img->GetSpacing()*spacingScalingFactor);
    
        typedef itk::AntiAliasBinaryImageFilter <ImageType, OutputImageType>
            AntiAliasBinaryImageFilterType;
    
        AntiAliasBinaryImageFilterType::Pointer antiAliasFilter
            = AntiAliasBinaryImageFilterType::New ();
        antiAliasFilter->SetInput(img);
        //antiAliasFilter->UseImageSpacingOff();
    
        antiAliasFilter->SetNumberOfIterations(iter);
        antiAliasFilter->SetMaximumRMSError(thresh);

        antiAliasFilter->Update();

        // For increased code coverage.  Does nothing.
        antiAliasFilter->GetMaximumRMSError();


        antiAliasFilter->Update();

        std::cout << "Maximum RMS change value threshold was: "<<thresh << std::endl;
        std::cout << "Last RMS change value was: " << antiAliasFilter->GetRMSChange() << std::endl;

        std::cout<< antiAliasFilter->GetElapsedIterations() <<endl;

        std::cout<<antiAliasFilter->GetLowerBinaryValue()<<" "<<    antiAliasFilter->GetUpperBinaryValue()<<std::endl;

        out=antiAliasFilter->GetOutput();
        out->SetSpacing(out->GetSpacing()/spacingScalingFactor);

    }else{

        typedef itk::SignedMaurerDistanceMapImageFilter< ImageType, OutputImageType > DistanceTransformType;
        DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
        distanceTransform->SetInput(img);
        distanceTransform->SquaredDistanceOff ();
        distanceTransform->UseImageSpacingOn();
        distanceTransform->Update();
        //OutputImagePointerType out=FilterUtils<OutputImageType>::gaussian(FilterUtils<ImageType,OutputImageType>::distanceMapByFastMarcher(img,1),img->GetSpacing());
        out=distanceTransform->GetOutput();
        out=FilterUtils<OutputImageType>::gaussian(out,img->GetSpacing()*thresh);
    
    }
    ImageUtils<OutputImageType>::writeImage(outFile,out);

	return 1;
}

