#include "Log.h"

#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"

#include "SRSConfig.h"
#include "SegmentationImageFilter.h"
#include "SegmentationGraph.h"
#include "BaseLabel.h"
#include "Potential-Segmentation-Unary.h"
#include "Classifier.h"
#include "Potential-Segmentation-Pairwise.h"


using namespace std;
using namespace itk;

int main(int argc, char ** argv)
{

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);

	SRSConfig filterConfig;
	filterConfig.parseParams(argc,argv);
	//define types.
	typedef float PixelType;
	const unsigned int D=3;
	typedef Image<PixelType,D> ImageType;
	typedef itk::Vector<float,D> BaseLabelType;
      typedef SparseRegistrationLabelMapper<ImageType,BaseLabelType> LabelMapperType;
     //unary seg
      //typedef HandcraftedBoneSegmentationClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationClassifier<ImageType> ClassifierType;
      typedef SegmentationClassifierProbabilityImage<ImageType> ClassifierType;
      //typedef UnaryPotentialSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;
      typedef UnaryPotentialSegmentationUnsignedBoneMarcel< ImageType > SegmentationUnaryPotentialType;

    //pairwise seg
//    typedef UnaryPotentialSegmentation< ImageType > SegmentationUnaryPotentialType;
    //typedef SmoothnessClassifierSignedGradient<ImageType> SegmentationSmoothnessClassifierType;
      typedef SmoothnessClassifierGradientContrast<ImageType> SegmentationSmoothnessClassifierType;
      //typedef PairwisePotentialSegmentationClassifier<ImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
      typedef PairwisePotentialSegmentationMarcel<ImageType> SegmentationPairwisePotentialType;
      //typedef PairwisePotentialSegmentationUniform<ImageType> SegmentationPairwisePotentialType;
        
   typedef SegmentationImageFilter<ImageType,
        LabelMapperType,
        SegmentationUnaryPotentialType,
        SegmentationPairwisePotentialType> FilterType;
	//create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(filterConfig);
    filter->setFixedImage(ImageUtils<ImageType>::readImage(filterConfig.targetFilename));
    filter->setFixedGradientImage(ImageUtils<ImageType>::readImage(filterConfig.fixedGradientFilename));

	clock_t start = clock();
	//DO IT!
	filter->Update();
	clock_t end = clock();
	float t = (float) ((double)(end - start) / CLOCKS_PER_SEC);
	LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
	return 1;
}
