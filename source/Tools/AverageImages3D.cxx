#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Image<float,D> FloatImageType;
    typedef FloatImageType::Pointer FloatImagePointerType;
    typedef ImageType::IndexType IndexType;
    string inFile, outFile;
    double variance=1.0,mean=0.0;
  
    if (argc<4){
        LOG<<"Usage: AverageImages3D <outputFile> <input1> <input2> ..."<<endl;
        exit(0);
    }
    
    GaussianEstimatorScalarImage<ImageType> accumulator;
    ImagePointerType img = ImageUtils<ImageType>::readImage(argv[2]);
    accumulator.addImage(img);
    
    for (int i=3;i<argc;++i){
         ImagePointerType img2 = ImageUtils<ImageType>::readImage(argv[i]);
         accumulator.addImage(img2);
    }
    accumulator.finalize();
    
    ImageUtils<ImageType>::writeImage(argv[1],accumulator.getMean());

	return 1;
}
