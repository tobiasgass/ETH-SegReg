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

const unsigned int D=3;
typedef float Label;
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;
typedef  LabelImage::Pointer LabelImagePointerType;



int main(int argc, char * argv [])
{

    
    ArgumentParser as(argc, argv);
	string imageFilename,outputFilename="";
    int verbose=0;
    string labelList="";
    bool info=false, region=false, content=false;
	as.parameter ("i", imageFilename, "input image (file name)", true);
	as.option("info",info,"general info about image");
	as.option("region",region,"print region object");
	as.option("content",content,"output all image content");
	
	as.parameter ("v", verbose, "verbosity level", false);

	as.parse();
	
    logSetVerbosity(verbose);
 
 
    LabelImage::Pointer img =
        ImageUtils<LabelImage>::readImage(imageFilename);
    if (content){
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(img,img->GetLargestPossibleRegion());

    for (it1.GoToBegin();!it1.IsAtEnd();++it1){
        std::cout<<it1.Get()<<std::endl;
    }
    }
    if (region){
      std::cout<<img->GetLargestPossibleRegion()<<std::endl;
    }
    if (info){
      std::cout<<"Size: "<<img->GetLargestPossibleRegion().GetSize()<<std::endl;
      std::cout<<"Spacing: "<<img->GetSpacing()<<std::endl;
      std::cout<<"Direction:";
      for (int d=0;d<D;++d){for (int d2=0;d2<D;++d2){ std::cout<<" "<<img->GetDirection()[d][d2];}}
      std::cout<<std::endl;
      std::cout<<"Origin: "<<img->GetOrigin()<<std::endl;
		  
    }
	return EXIT_SUCCESS;
}

