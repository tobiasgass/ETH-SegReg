
#include <iostream>



#include "itkCastImageFilter.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "ChamferDistanceTransform.h"
#include <map>
#include "ArgumentParser.h"

#define computeDistances 1

using namespace std;

static const int D=3;
typedef unsigned char	Label;
typedef itk::Image< Label, D >  LabelImage;

typedef short TLabel;
typedef itk::Image< TLabel, D >  TLabelImage;
typedef itk::Image< float, D > RealImage;


typedef LabelImage::OffsetType Offset;
typedef LabelImage::IndexType Index;
typedef LabelImage::PointType Point;


int main(int argc, char * argv [])
{

  ArgumentParser as(argc, argv);
  string automaticSegFile="", manualSegFile="", outputFile="";
  as.parameter("manual", manualSegFile, "manual segmentation", true);
  as.parameter("automatic", automaticSegFile, "automatic, binary segmentation", true);
  as.parameter("output", outputFile, "output segmentation", true);
  as.parse();

  //dilute groundtruth and substract it from estimate before joining
    LabelImage::Pointer groundTruthImage = ImageUtils<LabelImage>::readImage(manualSegFile);
    int maxLabel=FilterUtils<LabelImage>::getMax(groundTruthImage);
    //FilterUtils<LabelImage>::dilation(
    LabelImage::Pointer segmentedImage =ImageUtils<LabelImage>::readImage(automaticSegFile);
        //  (FilterUtils<LabelImage>::erosion(,4,valToReplace));

    
    LabelImage::Pointer extendedGT =  FilterUtils<LabelImage>::dilation(FilterUtils<LabelImage>::binaryThresholdingLow(groundTruthImage,1),3,1);
    LabelImage::Pointer newImage=ImageUtils<LabelImage>::createEmpty(segmentedImage);
    ImageUtils<LabelImage>::writeImage("eroded.nii",(LabelImage::ConstPointer)segmentedImage);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType gIt(groundTruthImage,groundTruthImage->GetLargestPossibleRegion());
    IteratorType gEIt(extendedGT,extendedGT->GetLargestPossibleRegion());
    IteratorType sIt(segmentedImage,groundTruthImage->GetLargestPossibleRegion());
    IteratorType nIt(newImage,groundTruthImage->GetLargestPossibleRegion());
   
    for (gEIt.GoToBegin(),gIt.GoToBegin(),sIt.GoToBegin(),nIt.GoToBegin();!gIt.IsAtEnd();++nIt,++gIt,++sIt,++gEIt){
        short int label=sIt.Get();
         if (gIt.Get()){
	   label=gIt.Get();
         }else if (label && ! gEIt.Get()) {
	   label=maxLabel+1;
         }else{
	   label=0;
         }
         nIt.Set(label);

    }
    //ImageUtils<LabelImage>::writeImage("joined.nii",(LabelImage::ConstPointer)newImage);

    typedef LabelImage::ConstPointer LConstPointer;
    typedef RealImage::ConstPointer RConstPointer;
    //ImageUtils<LabelImage>::writeImage(argv[3], (LConstPointer)FilterUtils<LabelImage>::dilation(FilterUtils<LabelImage>::erosion(newImage,5,1*multiplier),5,1*multiplier));
    ImageUtils<LabelImage>::writeImage(outputFile, (LConstPointer)(newImage));

    //ImageUtils<LabelImage>::writeImage("smoothed.png", (LConstPointer)FilterUtils<RealImage,LabelImage>::cast(extendedGT));
	return EXIT_SUCCESS;
}
