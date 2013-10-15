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
    double dilutesize=0.0;
    (*as) >> parameter ("in", inFile, " filename...", true);
    (*as) >> parameter ("out", outFile, " filename...", true);
    (*as) >> parameter ("d", dilutesize, "amount of dilation", true);

    (*as) >> help();
    as->defaultErrorHandling();

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);


    

    ImagePointerType outImage=FilterUtils<ImageType>::dilation(img,dilutesize,1);  
    ImageUtils<ImageType>::writeImage(outFile,outImage);

	return 1;
}
