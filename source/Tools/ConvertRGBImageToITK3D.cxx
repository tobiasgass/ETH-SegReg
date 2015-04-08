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
    typedef  unsigned char PixelType;
    const unsigned int D=2;
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

    //typedef  itk::RGBPixel<unsigned char> RGBPixelType;
    typedef  itk::Vector<unsigned char,3> RGBPixelType;
    typedef  itk::Image<RGBPixelType,D > RGBImageType;
    typedef  RGBImageType::Pointer RGBImagePointerType;

    typedef VariableLengthVector< unsigned char > VectorPixelType;
    typedef  itk::Image<VectorPixelType,D > VectorImageType;
    typedef  VectorImageType::Pointer VectorImagePointerType;

    

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile;

    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);

    as->parse();
    

    RGBImagePointerType img = ImageUtils<RGBImageType>::readImage(inFile);
    
    VectorImagePointerType vecImg= ImageUtils<VectorImageType>::createEmpty(img->GetRequestedRegion(),
                                                                  img->GetOrigin(),
                                                                  img->GetSpacing(),
                                                                  img->GetDirection());


    ImagePointerType comp1=ImageUtils<ImageType>::createEmpty(img->GetRequestedRegion(),
                                                                  img->GetOrigin(),
                                                                  img->GetSpacing(),
                                                                  img->GetDirection());

    
    typedef itk::ImageRegionIterator<RGBImageType> DeformationIteratorType;
    DeformationIteratorType defIt(img,img->GetLargestPossibleRegion());
    typedef itk::ImageRegionIterator<ImageType> FloatImageIteratorType;
    FloatImageIteratorType resultIt(comp1,comp1->GetLargestPossibleRegion());
    
    for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
        resultIt.Set(defIt.Get()[0]);
    }

    ImageUtils<ImageType>::writeImage("comp1.png",comp1);
    for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
        resultIt.Set(defIt.Get()[1]);
    }
    
    ImageUtils<ImageType>::writeImage("comp2.png",comp1);
    for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
        resultIt.Set(defIt.Get()[2]);
    }

    ImageUtils<ImageType>::writeImage("comp3.png",comp1);
    ImageUtils<RGBImageType>::writeImage(outFile,img);
    //ImageUtils<VectorImageType>::writeImage(outFile,vecImg);

	return 1;
}
