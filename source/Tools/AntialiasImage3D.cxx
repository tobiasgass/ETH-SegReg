#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkAntiAliasBinaryImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"


using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{
    //feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  int PixelType;
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
 
    string inFile, outFile, outSurface="";
    double thresh=0.001;
    int iter=50;
    int layers=2;
    double spacingScalingFactor=2.0;
    PixelType label=1;
    bool gaussian=false;


    ArgumentParser * as=new ArgumentParser(argc,argv);
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    //as->parameter ("outSurf", outSurface, " Output Zero Level Set Surface filename...", true);
    as->parameter ("label", label, "label to select from input volume", false);
    as->parameter ("iter", iter, " number of AA iterations (max)...", false);
    as->parameter ("layers", layers, " number of AA layers (??)...", false);
    as->parameter ("thresh", thresh, "max desired RMSE change for convergence...", false);
    as->parameter ("scale", spacingScalingFactor, "multiply spacing with scaling factor. Can reduce artefacts if the spacing of the input volume is very small. Somewhat weird...", false);
    as->option ("gauss", gaussian, "use gaussian smoothing of surface instead of AA filter. thresh becomes size of kernel relative to image spacing");
    as->parse();
    


    ImagePointerType img = FilterUtils<ImageType>::select(ImageUtils<ImageType>::readImage(inFile),label);
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
        antiAliasFilter->SetNumberOfLayers(layers);

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

