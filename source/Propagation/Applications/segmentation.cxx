#include "Log.h"

#include <stdio.h>
#include <iostream>

#include "argstream.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "itkImage.h"
#include "SegmentationPotentials.h"
#include "MRF.h"
#include "Grid.h"
#include "SegmentationLabel.h"
#include "FAST-PD-mrf-optimisation.h"
#include <fenv.h>
#include "TRW-S-Registration.h"

using namespace std;
using namespace itk;

#define _MANY_LABELS_

int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


	argstream as(argc, argv);
	string targetFilename,movingFilename,fixedSegmentationFilename, movingSegmentationFilename,outputFilename,deformableFilename,defFilename="";
	double pairwiseWeight=1;
	int displacementSampling=-1;
	double unaryWeight=1;
	int maxDisplacement=10;

	as >> parameter ("t", targetFilename, "target image (file name)", true);
	as >> parameter ("m", movingFilename, "moving image (file name)", true);
	as >> parameter ("s", movingSegmentationFilename, "moving segmentation image (file name)", true);
	as >> parameter ("o", outputFilename, "output image (file name)", true);
	as >> parameter ("p", pairwiseWeight,"weight for pairwise potentials", false);
	as >> parameter ("u", unaryWeight,"weight for unary potentials", false);
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
	ImageType::Pointer movingImage =
				ImageUtils<ImageType>::readImage(movingFilename);
		ImageType::Pointer movingSegmentationImage =
				ImageUtils<ImageType>::readImage(movingSegmentationFilename);
	//create Grid
	typedef Grid<ImageType> GridType;
	GridType::LabelType resolution;
	for (int d=0;d<D;++d){
		resolution[d]=1;
	}
	GridType fullimageGrid(targetImage,resolution);


	typedef unsigned short SegmentationLabelType;
	typedef Image<SegmentationLabelType,D> SegmentationImageType;
	typedef SegmentationImageType::Pointer SegmentationImagePointerType;
	typedef SegmentationLabelConverter<ImageType, SegmentationLabelType> RLCType;
	RLCType * RLC=new RLCType(targetImage,2);

	//	PairwisePotential
	typedef PairwiseSegmentationPotential<RLCType> PairwisePotentialType;
	PairwisePotentialType::Pointer potentialFunction=PairwisePotentialType::New();
	potentialFunction->SetFixedImage(targetImage);
	potentialFunction->SetGrid(&fullimageGrid);

	typedef UnarySegmentationPotential<RLCType> UnaryPotentialType;
	UnaryPotentialType::Pointer unaryFunction=UnaryPotentialType::New();
	unaryFunction->SetFixedImage(targetImage);
	unaryFunction->setLabelConverter(RLC);


	//	ok what now: create graph! solve graph! save result!Z

	typedef FastPDMRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
	//			typedef TRWS_MRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
	MRFSolverType mrfSolver(targetImage,targetImage,&fullimageGrid,potentialFunction,unaryFunction,unaryWeight,pairwiseWeight);
	LOG<<"run"<<std::endl;
	mrfSolver.optimize();



	SegmentationImagePointerType segmentedImage;
	segmentedImage=RLC->getSegmentationField(mrfSolver.getLabelImage());
	ImageUtils<SegmentationImageType>::writeImage(outputFilename, segmentedImage);



	LOG<<"done"<<std::endl;
	return 1;
}
