#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.hpp"

#include "ArgumentParser.h"
#include "Metrics.h"
#include "TransformationUtils.h"

using namespace std;

const unsigned int D=2;
typedef unsigned char Label;
typedef itk::Image< Label, D >  ImageType;
typedef  ImageType::Pointer ImageTypePointerType;
typedef itk::Image< float, D >  FloatImageType;



int main(int argc, char * argv [])
{


    ArgumentParser as(argc, argv);
	string first,second,outputFilename="";
    double sigma=1;
    std::string metric="lncc";
    std::string defFile="";
	as.parameter ("a", first, "image 1", true);
	as.parameter ("b", second, "image2", true);
	as.parameter ("s", sigma, "lncc kernel width", true);
	as.parameter ("o", outputFilename, "output image (file name)", false);
	as.parameter ("metric", metric, "metric (lncc,itklncc,lsad,lssd)", false);
	as.parameter ("def", defFile, "deformation field (optional)",false);
  
	as.parse();

 
    ImageType::Pointer firstImg =
        (ImageUtils<ImageType>::readImage(first));
    ImageType::Pointer secondImg =
        (ImageUtils<ImageType>::readImage(second));

    if (defFile!=""){
      TransfUtils<ImageType>::DeformationFieldPointerType def=ImageUtils<TransfUtils<ImageType>::DeformationFieldType>::readImage(defFile);
      secondImg=TransfUtils<ImageType>::warpImage(secondImg,def);

    }

    LOG<<VAR(FilterUtils<ImageType>::getMax(firstImg))<<" "
       <<VAR(FilterUtils<ImageType>::getMax(secondImg))<<std::endl;

    FloatImageType::Pointer localMetricImage=NULL;
    double similarity=0;
    if (metric == "lncc"){
        localMetricImage = Metrics<ImageType,FloatImageType>::efficientLNCC(firstImg,secondImg,sigma);
    }else if (metric == "itklncc"){
        localMetricImage = Metrics<ImageType,FloatImageType>::ITKLNCC(firstImg,secondImg,sigma);
    }else if (metric == "lsad"){
        //localMetricImage = Metrics<ImageType,FloatImageType>::LSADAutoNorm(firstImg,secondImg,sigma);
    }else if (metric == "lssd"){
        //localMetricImage = Metrics<ImageType,FloatImageType>::LSSDAutoNorm(firstImg,secondImg,sigma);
    }else if (metric == "ncc"){
      similarity = Metrics<ImageType,FloatImageType>::nCC(firstImg,secondImg);
      
    }else{
        std::cout<<"Unknown metric "<<metric<<std::endl;
        exit(0);
    }

    if (localMetricImage.IsNotNull()){
      similarity=FilterUtils<FloatImageType>::getMean(localMetricImage);
    
      if (outputFilename!="")
	ImageUtils<ImageType>::writeImage(outputFilename,FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(localMetricImage,255)));
    }
    LOG<<VAR(similarity)<<std::endl;

    return EXIT_SUCCESS;
}
