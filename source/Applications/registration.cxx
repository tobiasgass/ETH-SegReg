
#include <stdio.h>
#include <iostream>

#include "argstream.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "itkImage.h"
#include "Potentials.h"
#include "MRF.h"
#include "Grid.h"
#include "Label.h"
#include "FAST-PD-Registration-mrf.h"
#include <fenv.h>
#include "TRW-S-Registration.h"

using namespace std;
using namespace itk;

#define _MANY_LABELS_

int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


	argstream as(argc, argv);
	string targetFilename,movingFilename,outputFilename;
	as >> parameter ("t", targetFilename, "target image (file name)", true);
	as >> parameter ("m", movingFilename, "moving image (file name)", true);
	as >> parameter ("o", outputFilename, "output image (file name)", true);
	as >> help();
	as.defaultErrorHandling();

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

	//typedef int LabelType;
	typedef Offset<D> LabelType;

	//create Grid
	typedef Grid<ImageType> GridType;
	GridType::LabelType resolution;
	for (int d=0;d<D;++d){
		resolution[d]=1;
	}

	GridType fullimageGrid(targetImage,resolution);
	typedef RegistrationLabel<ImageType> RegistrationLabelType;
	typedef RegistrationLabelConverter<ImageType, RegistrationLabelType> RLCType;
	RLCType * RLC=new RLCType(targetImage,movingImage,fullimageGrid.getResolution(),20);
//	RegistrationLabelType rLabel(label,RLC);
#if 0
	for (int i=0;i<3*3;++i){
		rLabel=RLC->getLabel(i);
		std::cout<<i<<" "<<rLabel<<" "<<RLC->getIntegerLabel(rLabel)<<std::endl;
	}
#endif

#if 0
	itk::ImageRegionIteratorWithIndex<ImageType> it(targetImage, targetImage->GetLargestPossibleRegion());
	int i=0;
	for (it.GoToBegin() ; !it.IsAtEnd(); ++it,++i)
	{
		IndexType idx=it.GetIndex();
		std::cout<<i<<" " <<idx<< " "<<fullimageGrid.getGridPositionAtIndex(i)<<std::endl;
	}
#endif





	//	PairwisePotential
	typedef EuclideanPairwisePotential<RLCType> PairwisePotentialType;
	PairwisePotentialType::Pointer potentialFunction=PairwisePotentialType::New();


	typedef UnaryPotential<RLCType> UnaryPotentialType;
	UnaryPotentialType::Pointer unaryFunction=UnaryPotentialType::New();
	unaryFunction->SetMovingImage(movingImage);
	unaryFunction->SetFixedImage(targetImage);
	unaryFunction->setLabelConverter(RLC);


	//	ok what now: create graph! solve graph! save result!Z

	typedef FastPDMRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
//	typedef TRWS_MRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
	MRFSolverType mrfSolver(targetImage,movingImage,&fullimageGrid,potentialFunction,unaryFunction);
	std::cout<<"run"<<std::endl;
	mrfSolver.optimize();

	ImagePointerType transformedImage=mrfSolver.transformImage(movingImage);

	ImageUtils<ImageType>::writeImage(outputFilename, transformedImage);

	std::cout<<"wtf"<<std::endl;
	return 1;
}
