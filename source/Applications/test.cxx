/*
 * test.cxx
 *
 *  Created on: Feb 21, 2011
 *      Author: gasst
 */


#include <stdio.h>
#include <iostream>

#include "argstream.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "itkImage.h"
#include "Potentials.h"
#include "RegistrationSegmentationPotentials-v2.h"
#include "MRF.h"
#include "Grid.h"
#include "Label.h"
#include "RegistrationSegmentationLabel.h"
#include "FAST-PD-mrf-optimisation.h"
#include <fenv.h>
#include "TRW-S-Registration.h"
#include <sstream>
#include "itkHistogramMatchingImageFilter.h"
#include "itkConstNeighborhoodIterator.h"

using namespace std;
using namespace itk;

#define _MANY_LABELS_
#define DOUBLEPAIRWISE
int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


	argstream as(argc, argv);
	string targetFilename,movingFilename,fixedSegmentationFilename, movingSegmentationFilename, outputFilename,deformableFilename,defFilename="", segmentationOutputFilename;
	double pairwiseWeight=1;
	double pairwiseSegmentationWeight=1;
	int displacementSampling=-1;
	double unaryWeight=1;
	int maxDisplacement=10;
	double simWeight=1;
	double rfWeight=1;
	double segWeight=1;

	as >> parameter ("t", targetFilename, "target image (file name)", true);

	as >> parameter ("o", outputFilename, "output image (file name)", true);

	as >> help();
	as.defaultErrorHandling();

	if (displacementSampling==-1) displacementSampling=maxDisplacement;

	//typedefs
	typedef unsigned short PixelType;
	const unsigned int D=2;
	typedef Image<PixelType,D> ImageType;
	typedef ImageType::IndexType IndexType;
	typedef ImageType::Pointer ImagePointerType;
	//	typedef Image<LabelType> LabelImageType;
	//read input images
	ImageType::Pointer targetImage =
			ImageUtils<ImageType>::readImage(targetFilename);
	typedef itk::ImageDuplicator< ImageType > DuplicatorType;
	 DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(targetImage);
	duplicator->Update();
	ImagePointerType returnImage=duplicator->GetOutput();
	typedef itk::ConstNeighborhoodIterator< ImageType > NeighborhoodIteratorType;
	 NeighborhoodIteratorType::RadiusType radius;
	radius.Fill(15);
	NeighborhoodIteratorType ImageIterator(radius,targetImage, targetImage->GetLargestPossibleRegion());
	long int i=0;
	for (ImageIterator.GoToBegin();
			!ImageIterator.IsAtEnd() ;
			++ImageIterator)
	{

		PixelType max = ImageIterator.GetCenterPixel();
		for (unsigned i = 0; i < ImageIterator.Size(); i++)
		{
			if ( ImageIterator.GetPixel(i) > max )
			{
				max = ImageIterator.GetPixel(i);
			}
		}
		ImageType::IndexType idx=ImageIterator.GetIndex();
		returnImage->SetPixel(idx,65535*ImageIterator.GetCenterPixel()/max);
	}
	ImageUtils<ImageType>::writeImage(outputFilename, returnImage);


	return 1;
}
