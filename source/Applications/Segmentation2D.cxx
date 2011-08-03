
#include <stdio.h>
#include <iostream>

#include "argstream.h"

#include "SRSConfig.h"
#include "SegmentationImageFilter.h"
#include "SegmentationGraph.h"
#include "BaseLabel.h"
#include "Potential-Segmentation-Unary.h"


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
	typedef itk::Vector<float,D> BaseLabelType;
    typedef SparseRegistrationLabelMapper<ImageType,BaseLabelType> LabelMapperType;
    typedef UnaryPotentialSegmentationProb< ImageType > SegmentationUnaryPotentialType;
	typedef SegmentationImageFilter<ImageType,
        LabelMapperType,
        SegmentationUnaryPotentialType    > FilterType;
    
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
	std::cout<<"Finished computation after "<<t<<" seconds"<<std::endl;
	return 1;
}
