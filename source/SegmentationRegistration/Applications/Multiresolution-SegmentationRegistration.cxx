#include "Log.h"
#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"

#include "SRSConfig.h"
#include "HierarchicalSRSImageToImageFilter.h"
#include "Graph.h"
#include "Graph-ITKStyle.h"
#include "WeightingGraph.h"
#include "SubsamplingGraph.h"
#include "FastRegistrationGraph.h"
#include "BaseLabel.h"
#include "Potential-Registration-Unary.h"
#include "Potential-Registration-Pairwise.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Segmentation-Pairwise.h"
#include "Potential-SegmentationRegistration-Pairwise.h"
#include "MRF-TRW-S.h"

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
    typedef Image<PixelType,D> InternalImageType;
    typedef itk::Vector<float,D> BaseLabelType;
    //typedef DenseRegistrationLabelMapper<ImageType,BaseLabelType> LabelMapperType;
    typedef SparseRegistrationLabelMapper<ImageType,BaseLabelType> LabelMapperType;
    //    typedef UnaryPotentialSegmentationArtificial2< ImageType > SegmentationUnaryPotentialType;
    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;
    typedef HandcraftedBoneSegmentationClassifierGradient<InternalImageType> ClassifierType;
    //typedef SegmentationGaussianClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationClassifier<ImageType> ClassifierType;
    //typedef UnaryPotentialSegmentationClassifier< InternalImageType, ClassifierType > SegmentationUnaryPotentialType;
    typedef UnaryPotentialSegmentationUnsignedBoneMarcel< ImageType > SegmentationUnaryPotentialType;

    typedef SmoothnessClassifierGradient<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierGradientContrast<InternalImageType> SegmentationSmoothnessClassifierType;
    //typedef PairwisePotentialSegmentationClassifier<InternalImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
    typedef PairwisePotentialSegmentationMarcel<InternalImageType> SegmentationPairwisePotentialType;
    
    //typedef UnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef UnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef FastUnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCCWithBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCCWithDistanceBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef PairwisePotentialRegistration< LabelMapperType, ImageType > RegistrationPairwisePotentialType;
    //typedef PairwisePotentialRegistrationACP< LabelMapperType, ImageType > RegistrationPairwisePotentialType;
    
    //typedef PairwisePotentialRegistrationSigmoid< LabelMapperType, ImageType > RegistrationPairwisePotentialType;
    //typedef PairwisePotentialBoneSegmentationRegistration<  ImageType > SegmentationRegistrationPairwisePotentialType;
    typedef PairwisePotentialSegmentationRegistration<  ImageType > SegmentationRegistrationPairwisePotentialType;
    //typedef PairwisePotentialSegmentationRegistrationBinary<  ImageType > SegmentationRegistrationPairwisePotentialType;
    //typedef FastRegistrationGraphModel<
    typedef GraphModel<
    // // //typedef ITKGraphModel<
    //typedef SortedSubsamplingGraphModel<
    //typedef SubsamplingGraphModel2<
    //typedef SortedCumSumSubsamplingGraphModel<
    //typedef WeightedGraphModel<
         InternalImageType,
        RegistrationUnaryPotentialType,
        RegistrationPairwisePotentialType,
        SegmentationUnaryPotentialType,
        SegmentationPairwisePotentialType,
        SegmentationRegistrationPairwisePotentialType,
        LabelMapperType>        GraphType;
    
	typedef HierarchicalSRSImageToImageFilter<GraphType>        FilterType;
    
	//create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(filterConfig);
    filter->setFixedImage(FilterUtils<ImageType,InternalImageType>::cast(ImageUtils<ImageType>::readImage(filterConfig.targetFilename)));
    filter->setMovingImage(FilterUtils<ImageType,InternalImageType>::cast(ImageUtils<ImageType>::readImage(filterConfig.movingFilename)));
    filter->setMovingSegmentation(FilterUtils<ImageType,InternalImageType>::cast(ImageUtils<ImageType>::readImage(filterConfig.movingSegmentationFilename)));
    filter->setFixedGradientImage(FilterUtils<ImageType,InternalImageType>::cast(ImageUtils<ImageType>::readImage(filterConfig.fixedGradientFilename)));

	clock_t FULLstart = clock();
	//DO IT!
	filter->Update();
	clock_t FULLend = clock();
	float t = (float) ((double)(FULLend - FULLstart) / CLOCKS_PER_SEC);
	LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
	LOG<<"RegUnaries: "<<tUnary<<" Optimization: "<<tOpt<<std::endl;	
    LOG<<"RegPairwise: "<<tPairwise<<std::endl;
	return 1;
}
