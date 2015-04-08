#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include <itkAutoCropLabelMapFilter.h>
#include <itkLabelMap.h>
#include <itkBinaryImageToLabelMapFilter.h>
#include <itkLabelMapToBinaryImageFilter.h>
#include <itkChangeInformationImageFilter.h>

 

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

 
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile;
    double background=0.0;
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("b", background, "backgorund value to crop", true);

    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    //select background
    ImagePointerType backGroundImage=FilterUtils<ImageType>::select(img,background);
    //invert :O
    backGroundImage=FilterUtils<ImageType>::select(backGroundImage,0);
    ImageUtils<ImageType>::writeImage("background.nii",backGroundImage);
    
    typedef itk::BinaryImageToLabelMapFilter<ImageType> BinaryImageToLabelMapFilterType;
    BinaryImageToLabelMapFilterType::Pointer binaryImageToLabelMapFilter = BinaryImageToLabelMapFilterType::New();
    binaryImageToLabelMapFilter->SetInput(backGroundImage);
    binaryImageToLabelMapFilter->SetInputForegroundValue(1);
    binaryImageToLabelMapFilter->SetFullyConnected(false);
    binaryImageToLabelMapFilter->SetOutputBackgroundValue(0);
    
    //crop background image
    itk::AutoCropLabelMapFilter<BinaryImageToLabelMapFilterType::OutputImageType>::Pointer autcropfilter=itk::AutoCropLabelMapFilter<BinaryImageToLabelMapFilterType::OutputImageType>::New();
    autcropfilter->SetInput(binaryImageToLabelMapFilter->GetOutput());
   
   
    typedef itk::LabelMapToBinaryImageFilter<BinaryImageToLabelMapFilterType::OutputImageType,ImageType> LabelMapToBinaryImageFilterType;
    LabelMapToBinaryImageFilterType::Pointer binaryImageToLabelMapFilter2 = LabelMapToBinaryImageFilterType::New();
    binaryImageToLabelMapFilter2->SetInput( autcropfilter->GetOutput());
    binaryImageToLabelMapFilter2->Update();

    ImagePointerType croppedBackground=binaryImageToLabelMapFilter2->GetOutput();
    ImageUtils<ImageType>::writeImage("croppedbackground.nii",croppedBackground);
    ImagePointerType outImage=FilterUtils<ImageType>::NNResample(img,croppedBackground,false);
    itk::ChangeInformationImageFilter<ImageType>::Pointer sf=itk::ChangeInformationImageFilter<ImageType>::New();
    sf->SetInput(outImage);
    sf->SetReferenceImage(croppedBackground);
    sf->UseReferenceImageOn();
    sf->ChangeOriginOn();
    sf->Update();
    outImage=sf->GetOutput();
    LOG<<VAR(croppedBackground->GetOrigin())<<" "<<VAR(outImage->GetOrigin())<<" "<<autcropfilter->GetOutput()->GetOrigin()<<endl;
    ImageUtils<ImageType>::writeImage(outFile,outImage); 
   
	return 1;
}
