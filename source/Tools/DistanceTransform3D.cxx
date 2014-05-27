#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkSignedMaurerDistanceMapImageFilter.h"
using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
   

    typedef ImageType::Pointer ImagePointerType;


    typedef Image<float,D> FloatImageType;
    typedef FloatImageType::Pointer FloatImagePointerType;

 
    argstream * as=new argstream(argc,argv);
    string inFile, outFile;
    double thresh=0.0;
    (*as) >> parameter ("in", inFile, " filename...", true);
    (*as) >> parameter ("out", outFile, " filename...", true);
    (*as) >> parameter ("t", thresh, "threshold", true);

    (*as) >> help();
    as->defaultErrorHandling();

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    ImagePointerType outImage=FilterUtils<ImageType>::binaryThresholdingLow(img,thresh);
    
    typedef  itk::SignedMaurerDistanceMapImageFilter< ImageType,FloatImageType > DistanceTransformType;
    DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
  
 
    distanceTransform->SetInput(outImage);
    //distanceTransform->SquaredDistanceOn();
    distanceTransform->SquaredDistanceOff();
    distanceTransform->UseImageSpacingOn();
    distanceTransform->Update();
    FloatImagePointerType result=distanceTransform->GetOutput();
    result->SetDirection(img->GetDirection());
    ImageUtils<FloatImageType>::writeImage(outFile,result);

	return 1;
}
