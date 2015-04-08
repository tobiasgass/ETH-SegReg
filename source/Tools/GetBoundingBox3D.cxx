#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include <itkLabelMapToBinaryImageFilter.h>
#include <itkAutoCropLabelMapFilter.h>
using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  unsigned char PixelType;
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
  
    if (argc<3){
        LOG<<"Usage: GetBoundingBox3D input output <dilation>."<<endl;
        exit(0);
    }
    
    ImagePointerType img = ImageUtils<ImageType>::readImage(argv[1]);
    
   
    
    if (argc>3){
        LOG<<"Dilating binary image by "<<argv[3]<<endl;
        img=FilterUtils<ImageType>::dilation(img,atof(argv[3]));
    }
    typedef itk::BinaryImageToLabelMapFilter<ImageType> BinaryImageToLabelMapFilterType;
    BinaryImageToLabelMapFilterType::Pointer binaryImageToLabelMapFilter = BinaryImageToLabelMapFilterType::New();
    binaryImageToLabelMapFilter->SetInput(img);
    binaryImageToLabelMapFilter->SetInputForegroundValue(1);

    binaryImageToLabelMapFilter->Update();
    BinaryImageToLabelMapFilterType::OutputImageType::Pointer labelMap=binaryImageToLabelMapFilter->GetOutput();
    LOG<<"Computed Label map with "<<labelMap->GetNumberOfLabelObjects()<<" labels."<<endl;
    typedef itk::AutoCropLabelMapFilter<BinaryImageToLabelMapFilterType::OutputImageType> CropFilterType;
    CropFilterType::Pointer filter=CropFilterType::New();
    filter->SetInput( labelMap );
    
    filter->Update();
    labelMap=filter->GetOutput();
    LOG<<"autocropped labelmap to "<<labelMap->GetLargestPossibleRegion()<<endl;
    
    typedef itk::LabelMapToBinaryImageFilter<BinaryImageToLabelMapFilterType::OutputImageType,ImageType> CastFilterType;
    CastFilterType::Pointer castFilter=CastFilterType::New();
    castFilter->SetInput(labelMap);
    castFilter->Update();
    img=castFilter->GetOutput();
    LOG<<"converted label map back to binary image"<<endl;
    ImageUtils<ImageType>::writeImage(argv[2],img);

	return 1;
}
