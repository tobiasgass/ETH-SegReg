#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkCastImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include <map>

#include <map>
#include "ArgumentParser.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <sstream>
#include "TransformationUtils.h"
#include "Metrics.h"
using namespace std;

const unsigned int D=2;
typedef unsigned char Label;
typedef itk::Image< Label, D >  ImageType;
typedef  ImageType::Pointer ImageTypePointerType;
typedef itk::Image< float, D >  FloatImageType;
typedef   FloatImageType::Pointer FloatImagePointerType;
typedef ImageUtils<FloatImageType>::ImageIteratorType IteratorType;


void localMax(FloatImagePointerType i1,FloatImagePointerType i2){
    IteratorType i1It(i1,i1->GetLargestPossibleRegion());
    IteratorType i2It(i2,i2->GetLargestPossibleRegion());
    for (i1It.GoToBegin(),i2It.GoToBegin();!i1It.IsAtEnd();++i1It,++i2It){
        float v1=i1It.Get();
        float v2=i2It.Get();
        i1It.Set(v1>v2?v1:v2);
    }
}

int main(int argc, char * argv [])
{


    ArgumentParser as(argc, argv);
	string groundTruth,segmentationFilename,outputFilename="";
    double sigma=1;
    std::string metric="lncc";
    int nSamples=2;
    string acc="mean";
    string eval="exp";
	as.parameter ("a", groundTruth, "image 1", true);
	as.parameter ("b", segmentationFilename, "image2", true);
	as.parameter ("s", sigma, "lncc kernel width", true);
	as.parameter ("o", outputFilename, "output image (file name)", false);
	as.parameter ("metric", metric, "metric (lncc,itklncc,lsad,lssd)", false);
	as.parameter ("nSamples", nSamples, "number of samples for autocorrelation", false);
    as.parameter ("acc", acc, "accumulater (mean,max)", false);
	as.parameter ("eval", eval, "evaluation method (exp,diff)", false);

	as.parse();

    int nTotalSamples=D*2*nSamples;
  
 
    ImageType::Pointer groundTruthImg =
        (ImageUtils<ImageType>::readImage(groundTruth));
    ImageType::Pointer segmentedImg =
        (ImageUtils<ImageType>::readImage(segmentationFilename));

    FloatImageType::Pointer centerMetric;


    if (metric == "lncc"){
        centerMetric = Metrics<ImageType,FloatImageType>::efficientLNCC(groundTruthImg,segmentedImg,sigma);
    }else if (metric == "itklncc"){
        centerMetric = Metrics<ImageType,FloatImageType>::ITKLNCC(groundTruthImg,segmentedImg,sigma);
    }else if (metric == "lsad"){
        centerMetric = Metrics<ImageType,FloatImageType>::LSAD(groundTruthImg,segmentedImg,sigma);
    }else if (metric == "lssd"){
        centerMetric = Metrics<ImageType,FloatImageType>::LSSD(groundTruthImg,segmentedImg,sigma);
    }else{
        std::cout<<"Unknown metric "<<metric<<std::endl;
        exit(0);
    }

    FloatImageType::Pointer accumulator=ImageUtils<FloatImageType>::createEmpty(centerMetric);
    accumulator->FillBuffer(0.0);
    LOG<<VAR(FilterUtils<ImageType>::getMax(groundTruthImg))<<" "
       <<VAR(FilterUtils<ImageType>::getMax(segmentedImg))<<std::endl;

    nTotalSamples=1;
    std::vector<FloatImageType::Pointer> metrics(nTotalSamples,NULL);

    typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef  DeformationFieldType::Pointer DeformationFieldPointerType;
    DeformationFieldPointerType def=TransfUtils<ImageType>::createEmpty(segmentedImg);

    
    int count=0;
    for (int d=0;d<D;++d){

        for (int sign=-1;sign<2;sign+=2){
            for (int sample=0;sample<nSamples;++sample){
                
                double defMag=sign*pow(2,sample)*sigma;
                TransfUtils<ImageType>::DisplacementType disp;
                disp.Fill(0);
                disp[d]=defMag;
                def->FillBuffer(disp);
                ImageType::Pointer warped=TransfUtils<ImageType>::warpImage(groundTruthImg,def);
                if (metric == "lncc"){
                    metrics[count] = Metrics<ImageType,FloatImageType>::efficientLNCC(warped,segmentedImg,sigma);
                }else if (metric == "itklncc"){
                    metrics[count] = Metrics<ImageType,FloatImageType>::ITKLNCC(warped,segmentedImg,sigma);
                }else if (metric == "lsad"){
                    metrics[count] = Metrics<ImageType,FloatImageType>::LSAD(warped,segmentedImg,sigma);
                }else if (metric == "lssd"){
                    metrics[count] = Metrics<ImageType,FloatImageType>::LSSD(warped,segmentedImg,sigma);
                }else{
                    std::cout<<"Unknown metric "<<metric<<std::endl;
                    exit(0);
                }

                if (acc=="mean"){
                    accumulator=FilterUtils<FloatImageType>::add(accumulator,metrics[count]);
                }else if (acc=="max"){
                    localMax(accumulator,metrics[count]);
                }else{
                    LOG<<"unknown accumulation method "<<acc<<endl;
                    exit(0);
                }

                //++count;
                                                         
            }
        }
    }
    if (acc == "mean")
        ImageUtils<FloatImageType>::multiplyImage(accumulator,1.0/(D*2*nSamples));

    if (eval=="diff"){
        // l= 0.5 + (\hat l - mu)/\hat l
        FloatImageType::Pointer diff=FilterUtils<FloatImageType>::substract(centerMetric,accumulator);
        centerMetric=ImageUtils<FloatImageType>::divideImageOutOfPlace(diff,centerMetric);
        ImageUtils<FloatImageType>::add(centerMetric,0.5);
        centerMetric=FilterUtils<FloatImageType>::thresholding(centerMetric,0,1);
    }else if (eval == "exp" ){
        

        IteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        IteratorType resultIt(centerMetric,centerMetric->GetLargestPossibleRegion());
        
        accIt.GoToBegin();
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++resultIt,++accIt){
            double Lprime=resultIt.Get();
            double Lbar=accIt.Get();
            
            double weight;
            if (Lprime != 0.0)
                weight = exp (- 0.6931 * Lbar/Lprime);
            else
                weight = 0.0;
            resultIt.Set(weight);

        }

    }else{
        LOG<<"Unknown evaluation method "<<eval<<endl;
        exit(0);
    }

    ImageUtils<ImageType>::writeImage(outputFilename,FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(centerMetric,255)));
    

	return EXIT_SUCCESS;
}
