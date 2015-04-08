#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkLabelShapeKeepNObjectsImageFilter.h"

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

    typedef unsigned char OutPixelType;
    typedef Image<OutPixelType,D> OutImageType;
    typedef OutImageType::Pointer OutImagePointerType;

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile;
    double thresh=0.0;
    bool lcc=false;
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("t", thresh, "threshold", true);
    as->option ("lcc", lcc, "get only largest connected object after thresholding");

    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    //OutImagePointerType outImage=FilterUtils<ImageType,OutImageType>::binaryThresholdingHigh(img,thresh);
    OutImagePointerType outImage=FilterUtils<ImageType,OutImageType>::binaryThresholdingLow(img,thresh);
    OutPixelType maxPx=FilterUtils<OutImageType>::getMax(outImage);
    if (maxPx != 1){
        LOG<<"no pixel greater thresh found, aborting"<<endl;
        return 0;
    }
    if (lcc){
        typedef itk::ConnectedComponentImageFilter<OutImageType,OutImageType>  ConnectedComponentImageFilterType;
        typedef ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
        ConnectedComponentImageFilterPointer filter =
            ConnectedComponentImageFilterType::New();
        filter->SetInput(outImage);
        outImage=filter->GetOutput();

        outImage=FilterUtils<OutImageType>::relabelComponents(outImage);
        outImage =  FilterUtils<OutImageType>::binaryThresholding(outImage,1, 1); 
#if 0
        typedef itk::LabelShapeKeepNObjectsImageFilter< OutImageType > LabelShapeKeepNObjectsImageFilterType;
        LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
        labelShapeKeepNObjectsImageFilter->SetInput(outImage);
        labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
        labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1);
        labelShapeKeepNObjectsImageFilter->SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
        labelShapeKeepNObjectsImageFilter->Update();
        outImage =  FilterUtils<OutImageType>::binaryThresholdingLow(labelShapeKeepNObjectsImageFilter->GetOutput(), 1); 
#endif
    }
    ImageUtils<OutImageType>::writeImage(outFile,outImage);

	return 1;
}
