
#include <stdio.h>
#include <iostream>

#include "argstream.h"

#include "SRSConfig.h"
#include "HierarchicalSRSImageToImageFilter.h"
#include "Potential-SRS-MultiVariateNCC.h"
#include "Potential-SRS-SAD-SACentered.h"

using namespace std;
using namespace itk;

int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);

	SRSConfig config;
	config.parseParams(argc,argv);
	//define types.
	typedef unsigned short PixelType;
	const unsigned int D=2;
	typedef Image<PixelType,D> ImageType;
	typedef itk::Vector<float,D+1> BaseLabelType;
	//typedef BaseLabelMapper<ImageType,BaseLabelType> LabelMapperType;
	typedef SparseLabelMapper<ImageType,BaseLabelType> LabelMapperType;
	//	typedef  NCCSRSUnaryPotential< LabelMapperType, ImageType > UnaryPotentialType;
	//typedef  MultiVariateNCCSRSUnaryPotential< LabelMapperType, ImageType > UnaryPotentialType;
	//	typedef typename  NCCRegistrationUnaryPotential< LabelMapperType, ImageType, SegmentationInterpolatorType,ImageInterpolatorType > BaseUnaryPotentialType;
	//	typedef typename  SegmentationRegistrationUnaryPotential< LabelMapperType, ImageType, SegmentationInterpolatorType,ImageInterpolatorType > BaseUnaryPotentialType;
	typedef  SegmentationRegistrationUnaryPotentialPosteriorSA< LabelMapperType, ImageType > UnaryPotentialType;
	typedef HierarchicalSRSImageToImageFilter<ImageType,LabelMapperType,UnaryPotentialType > FilterType;
	//create filter
	FilterType filter(config);
	clock_t start = clock();
	//DO IT!
	filter.run();
	clock_t end = clock();
	float t = (float) ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout<<"Finished computation after "<<t<<" seconds"<<std::endl;
}
