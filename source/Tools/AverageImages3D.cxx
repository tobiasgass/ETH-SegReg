#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  float PixelType;
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
  
    if (argc<5){
        LOG<<"Usage: AverageImages3D <outputMean> <outputVariance> <input1> <input2> ..."<<endl;
        exit(0);
    }
    
    GaussianEstimatorScalarImage<ImageType> accumulator;
    ImagePointerType img = ImageUtils<ImageType>::readImage(argv[3]);
    accumulator.addImage(img);
    
    for (int i=4;i<argc;++i){
         ImagePointerType img2 = ImageUtils<ImageType>::readImage(argv[i]);
         accumulator.addImage(img2);
    }
    accumulator.finalize();
    FloatImagePointerType floatMean=accumulator.getFloatMean();
    ImageUtils<FloatImageType>::writeImage(argv[1],floatMean);
    ImagePointerType intMean=FilterUtils<FloatImageType,ImageType>::round(floatMean);
    //    ImageUtils<ImageType>::writeImage(argv[1],intMean);
    //ImageUtils<ImageType>::writeImage(argv[1],intMean);
    ImageUtils<FloatImageType>::writeImage(argv[2],accumulator.getFloatVariance());

	return 1;
}
