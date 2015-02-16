
#include <iostream>



#include "itkCastImageFilter.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "ChamferDistanceTransform.h"
#include <map>


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

    int multiplier=1;
    if (D==2){
        multiplier=127;
    }
    int valToReplace=multiplier;
    if (D==2){
        valToReplace=255;
    }
#if 1
    //dilute groundtruth and substract it from estimate before joining
    LabelImage::Pointer groundTruthImage = ImageUtils<LabelImage>::readImage(argv[1]);
    //FilterUtils<LabelImage>::dilation(
    LabelImage::Pointer segmentedImage =ImageUtils<LabelImage>::readImage(argv[2]);
        //  (FilterUtils<LabelImage>::erosion(,4,valToReplace));

    
    LabelImage::Pointer extendedGT =  FilterUtils<LabelImage>::dilation(groundTruthImage,3,valToReplace);
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
             label=2;
         }else if (label && ! gEIt.Get()) {
             label=1;
         }else{
             label=0;
         }
         nIt.Set(label*multiplier);

    }
    ImageUtils<LabelImage>::writeImage("joined.nii",(LabelImage::ConstPointer)newImage);

    typedef LabelImage::ConstPointer LConstPointer;
    typedef RealImage::ConstPointer RConstPointer;
    //ImageUtils<LabelImage>::writeImage(argv[3], (LConstPointer)FilterUtils<LabelImage>::dilation(FilterUtils<LabelImage>::erosion(newImage,5,1*multiplier),5,1*multiplier));
    ImageUtils<LabelImage>::writeImage(argv[3], (LConstPointer)(newImage));

#else
    //substract groundtruth, then erode and dilute secondary label
    LabelImage::Pointer groundTruthImage = ImageUtils<LabelImage>::readImage(argv[1]);
    LabelImage::Pointer segmentedImage =ImageUtils<LabelImage>::readImage(argv[2]);

    LabelImage::Pointer extendedGT = groundTruthImage;
    LabelImage::Pointer newImage=ImageUtils<LabelImage>::createEmpty(segmentedImage);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType gIt(groundTruthImage,groundTruthImage->GetLargestPossibleRegion());
    IteratorType gEIt(extendedGT,extendedGT->GetLargestPossibleRegion());
    IteratorType sIt(segmentedImage,groundTruthImage->GetLargestPossibleRegion());
    IteratorType nIt(newImage,groundTruthImage->GetLargestPossibleRegion());
   
    for (gEIt.GoToBegin(),gIt.GoToBegin(),sIt.GoToBegin(),nIt.GoToBegin();!gIt.IsAtEnd();++nIt,++gIt,++sIt,++gEIt){
        short int label=sIt.Get();
         if (gIt.Get()){
             label=2;
         }else if (label && ! gEIt.Get()) {
             label=1;
         }else{
             label=0;
         }
         nIt.Set(label*multiplier);

    }
    ImageUtils<LabelImage>::writeImage("joined.nii",(LabelImage::ConstPointer)newImage);

    typedef LabelImage::ConstPointer LConstPointer;
    typedef RealImage::ConstPointer RConstPointer;
    typedef  LabelImage::Pointer ImagePointerType;
    ImagePointerType output=FilterUtils<LabelImage>::dilation(FilterUtils<LabelImage>::erosion(newImage,1,1*multiplier),1,1*multiplier);
#if 0  
    //create gap between primary and auxiliary label
    ImagePointerType dilatedGroundTruth=FilterUtils<LabelImage>::dilation(groundTruthImage,2);
    output=FilterUtils<LabelImage>::substract(output,dilatedGroundTruth);
#endif
    ImageUtils<LabelImage>::writeImage(argv[3], output);
   
#endif
    //ImageUtils<LabelImage>::writeImage("smoothed.png", (LConstPointer)FilterUtils<RealImage,LabelImage>::cast(extendedGT));
	return EXIT_SUCCESS;
}
