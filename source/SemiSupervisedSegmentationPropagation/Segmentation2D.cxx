#include "Log.h"

#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"

#include "SRSConfig.h"
#include "SegmentationImageFilter.h"
#include "SegmentationGraph.h"
#include "BaseLabel.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Segmentation-Pairwise.h"
#include "Classifier.h"


using namespace std;
using namespace itk;

int main(int argc, char ** argv)
{

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);

	SRSConfig filterConfig;
	filterConfig.parseParams(argc,argv);
	//define types.
	typedef unsigned char PixelType;
	const unsigned int D=2;
	typedef Image<PixelType,D> ImageType;
    typedef Image<unsigned char,D> InputImageType;
	typedef itk::Vector<float,D> BaseLabelType;
    typedef SparseRegistrationLabelMapper<ImageType,BaseLabelType> LabelMapperType;

    typedef UnaryPotentialSegmentationUnsignedBoneMarcel< ImageType > SegmentationUnaryPotentialType;

    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;
    //typedef HandcraftedBoneSegmentationClassifierGradient<ImageType> ClassifierType;
    //    typedef SegmentationClassifierProbabilityImage<ImageType> ClassifierType;
    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;

        //typedef UnaryPotentialSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;

    //typedef SmoothnessClassifierGradient<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierGradientContrast<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierUniform<ImageType> SegmentationSmoothnessClassifierType;
    //typedef PairwisePotentialSegmentationClassifier<ImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
    typedef PairwisePotentialSegmentationMarcel<ImageType> SegmentationPairwisePotentialType;
	typedef SegmentationImageFilter<ImageType,
        LabelMapperType,
        SegmentationUnaryPotentialType,
        SegmentationPairwisePotentialType> FilterType;
    
	//create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(filterConfig);
    filter->setFixedImage(ImageUtils<ImageType>::readImage(filterConfig.targetFilename));
    filter->setMovingImage(ImageUtils<ImageType>::readImage(filterConfig.movingFilename));
    filter->setMovingSegmentation(ImageUtils<ImageType>::readImage(filterConfig.movingSegmentationFilename));
    filter->setFixedGradientImage(ImageUtils<ImageType>::readImage(filterConfig.fixedGradientFilename));

	clock_t start = clock();
	//DO IT!
	filter->Update();
	clock_t end = clock();
	float t = (float) ((double)(end - start) / CLOCKS_PER_SEC);
	LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
	return 1;
}
