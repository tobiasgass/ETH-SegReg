#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkCastImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include <map>

#include <map>
#include "argstream.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <sstream>


using namespace std;

const unsigned int D=2;
typedef unsigned char Label;
typedef itk::Image< Label, D >  ImageType;
typedef  ImageType::Pointer ImageTypePointerType;
typedef itk::Image< float, D >  FloatImageType;



int main(int argc, char * argv [])
{


    argstream as(argc, argv);
	string groundTruth,segmentationFilename,outputFilename="";
    double sigma=1;
    std::string metric="lncc";
	as >> parameter ("a", groundTruth, "image 1", true);
	as >> parameter ("b", segmentationFilename, "image2", true);
	as >> parameter ("s", sigma, "lncc kernel width", true);
	as >> parameter ("o", outputFilename, "output image (file name)", false);
	as >> parameter ("metric", metric, "metric (lncc,itklncc,lsad,lssd)", false);
  
	as >> help();
	as.defaultErrorHandling();

 
    ImageType::Pointer groundTruthImg =
        (ImageUtils<ImageType>::readImage(groundTruth));
    ImageType::Pointer segmentedImg =
        (ImageUtils<ImageType>::readImage(segmentationFilename));


    LOG<<VAR(FilterUtils<ImageType>::getMax(groundTruthImg))<<" "
       <<VAR(FilterUtils<ImageType>::getMax(segmentedImg))<<std::endl;

    FloatImageType::Pointer lncc;

    if (metric == "lncc"){
        lncc = FilterUtils<ImageType,FloatImageType>::efficientLNCC(groundTruthImg,segmentedImg,sigma);
    }else if (metric == "itklncc"){
        lncc = FilterUtils<ImageType,FloatImageType>::ITKLNCC(groundTruthImg,segmentedImg,sigma);
    }else if (metric == "lsad"){
        lncc = FilterUtils<ImageType,FloatImageType>::LSADAutoNorm(groundTruthImg,segmentedImg,sigma);
    }else if (metric == "lssd"){
        lncc = FilterUtils<ImageType,FloatImageType>::LSSDAutoNorm(groundTruthImg,segmentedImg,sigma);
    }else{
        std::cout<<"Unknown metric "<<metric<<std::endl;
        exit(0);
    }


    ImageUtils<ImageType>::writeImage(outputFilename,FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255)));
    

	return EXIT_SUCCESS;
}
