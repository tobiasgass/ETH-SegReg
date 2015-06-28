#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkMRIBiasFieldCorrectionFilter.h"
using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  float PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef Image<float,D> OutputImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
 
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile,refFile="";
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    typedef itk::MRIBiasFieldCorrectionFilter<ImageType,ImageType,ImageType> BiasFilterType;
    BiasFilterType::Pointer filter=BiasFilterType::New();
    filter->SetInput(img);
    filter->Update();
    ImagePointerType result=filter->GetOutput();
    
    ImageUtils<OutputImageType>::writeImage(outFile,result);

	return 1;
}
