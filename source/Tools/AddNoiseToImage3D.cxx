#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <fstream>

#include <itkInverseDisplacementFieldImageFilter.h>
#include "itkVectorLinearInterpolateImageFunction.h"

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile;
    double variance=1.0,mean=0.0;
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("var", variance, "variance of additive noise", true);
    as->parameter ("mean", mean, "mean of additive noise", true);

    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    ImagePointerType noiseImage=ImageUtils<ImageType>::addNoise(img,variance,mean);
    
    ImageUtils<ImageType>::writeImage(outFile,noiseImage);

	return 1;
}
