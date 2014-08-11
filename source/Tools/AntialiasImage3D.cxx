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

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    typedef float OutputPixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;

    typedef Image<OutputPixelType,D> OutputImageType;
    typedef OutputImageType::Pointer OutputImagePointerType;
 
    argstream * as=new argstream(argc,argv);
    string inFile, outFile;
    double thresh=0.0;
    (*as) >> parameter ("in", inFile, " filename...", true);
    (*as) >> parameter ("out", outFile, " filename...", true);

    (*as) >> help();
    as->defaultErrorHandling();

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);
#if 0
    typedef itk::AntiAliasBinaryImageFilter <ImageType, OutputImageType>
        AntiAliasBinaryImageFilterType;
    
    AntiAliasBinaryImageFilterType::Pointer antiAliasFilter
        = AntiAliasBinaryImageFilterType::New ();
    antiAliasFilter->SetInput(img);
    antiAliasFilter->SetNumberOfIterations(200);
    antiAliasFilter->SetMaximumRMSError(0.02);

    antiAliasFilter->Update();
    OutputImagePointerType out=antiAliasFilter->GetOutput();
#else
    typedef itk::SignedMaurerDistanceMapImageFilter< ImageType, OutputImageType > DistanceTransformType;
    DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
    distanceTransform->SetInput(img);
    distanceTransform->SquaredDistanceOff ();
    distanceTransform->UseImageSpacingOn();
    distanceTransform->Update();
    //OutputImagePointerType out=FilterUtils<OutputImageType>::gaussian(FilterUtils<ImageType,OutputImageType>::distanceMapByFastMarcher(img,1),img->GetSpacing());
    OutputImagePointerType out=distanceTransform->GetOutput();
    out=FilterUtils<OutputImageType>::gaussian(out,img->GetSpacing());
#endif
    ImageUtils<OutputImageType>::writeImage(outFile,out);

	return 1;
}

