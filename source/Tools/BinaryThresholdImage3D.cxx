#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkLabelShapeKeepNObjectsImageFilter.h"

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
 
    argstream * as=new argstream(argc,argv);
    string inFile, outFile;
    double thresh=0.0;
    bool lcc=false;
    (*as) >> parameter ("in", inFile, " filename...", true);
    (*as) >> parameter ("out", outFile, " filename...", true);
    (*as) >> parameter ("t", thresh, "threshold", true);
    (*as) >> option ("lcc", lcc, "get only largest connected object after thresholding");

    (*as) >> help();
    as->defaultErrorHandling();

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    ImagePointerType outImage=FilterUtils<ImageType>::binaryThresholdingLow(img,thresh);
    PixelType maxPx=FilterUtils<ImageType>::getMax(outImage);
    if (maxPx != 1){
        LOG<<"no pixel greater thresh found, aborting"<<endl;
        return 0;
    }
    if (lcc){
        typedef itk::ConnectedComponentImageFilter<ImageType,ImageType>  ConnectedComponentImageFilterType;
        typedef ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
        ConnectedComponentImageFilterPointer filter =
            ConnectedComponentImageFilterType::New();
        filter->SetInput(outImage);
        outImage=filter->GetOutput();

        outImage=FilterUtils<ImageType>::relabelComponents(outImage);
        outImage =  FilterUtils<ImageType>::binaryThresholding(outImage,1, 1); 
#if 0
        typedef itk::LabelShapeKeepNObjectsImageFilter< ImageType > LabelShapeKeepNObjectsImageFilterType;
        LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
        labelShapeKeepNObjectsImageFilter->SetInput(outImage);
        labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
        labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1);
        labelShapeKeepNObjectsImageFilter->SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
        labelShapeKeepNObjectsImageFilter->Update();
        outImage =  FilterUtils<ImageType>::binaryThresholdingLow(labelShapeKeepNObjectsImageFilter->GetOutput(), 1); 
#endif
    }
    ImageUtils<ImageType>::writeImage(outFile,outImage);

	return 1;
}
