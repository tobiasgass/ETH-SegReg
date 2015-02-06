#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef Image<float,D> OutputImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
 
    argstream * as=new argstream(argc,argv);
    string inFile, outFile,refFile="";
    double factor=-1;
    bool noSmoothing=false;
    bool nnResampling=false;
    (*as) >> parameter ("in", inFile, " filename...", true);
    (*as) >> parameter ("out", outFile, " filename...", true);
    (*as) >> help();
    as->defaultErrorHandling();

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

 
    
    ImageUtils<OutputImageType>::writeImage(outFile,FilterUtils<ImageType,OutputImageType>::normalizeImage(img));

	return 1;
}
