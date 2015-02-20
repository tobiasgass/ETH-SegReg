#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkCastImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include <map>
#include "itkConnectedComponentImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include <itkMinimumMaximumImageCalculator.h>
#include <itkHausdorffDistanceImageFilter.h>
#include <map>
#include "ArgumentParser.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include "mmalloc.h"
#include "SegmentationMapper.hxx"
using namespace std;

const unsigned int D=2;
typedef float Label;
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;
typedef  LabelImage::Pointer LabelImagePointerType;

LabelImagePointerType selectLabel(LabelImagePointerType img, Label l){
    LabelImagePointerType result=ImageUtils<LabelImage>::createEmpty(img);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(img,img->GetLargestPossibleRegion());
    IteratorType it2(result,img->GetLargestPossibleRegion());
    for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
        it2.Set(it1.Get()==l);
    }
    return result;
}


int main(int argc, char * argv [])
{

    
    ArgumentParser as(argc, argv);
	string segmentationFilename,outputFilename="";
    int verbose=0;
    string labelList="";
    int targetLabel=-1;
    bool binary=false;
	as.parameter ("i", segmentationFilename, "segmentation image (file name)", true);
	as.parameter ("v", verbose, "verbosity level", false);

	as.parse();
	
    logSetVerbosity(verbose);
 
 
    LabelImage::Pointer img =
        ImageUtils<LabelImage>::readImage(segmentationFilename);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(img,img->GetLargestPossibleRegion());

    for (it1.GoToBegin();!it1.IsAtEnd();++it1){
        std::cout<<it1.Get()<<std::endl;
    }
	return EXIT_SUCCESS;
}

