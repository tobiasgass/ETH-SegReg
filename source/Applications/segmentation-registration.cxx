#include "Log.h"

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
#include "Graph.h"
#include "BaseLabel.h"
#include "BasePotential.h"
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
	as >> parameter ("m", movingFilename, "moving image (file name)", true);
	as >> parameter ("s", movingSegmentationFilename, "moving segmentation image (file name)", true);
	as >> parameter ("g", fixedSegmentationFilename, "fixed segmentation image (file name)", true);

	as >> parameter ("o", outputFilename, "output image (file name)", true);
	as >> parameter ("O", segmentationOutputFilename, "output segmentation image (file name)", true);
	as >> parameter ("f", defFilename,"deformation field filename", false);
	as >> parameter ("p", pairwiseWeight,"weight for pairwise potentials", false);
	as >> parameter ("sp", pairwiseSegmentationWeight,"weight for pairwise segmentation potentials", false);

	as >> parameter ("u", unaryWeight,"weight for unary potentials", false);
	as >> parameter ("max", maxDisplacement,"maximum displacement in pixels per axis", false);
	as >> parameter ("wi", simWeight,"weight for intensity similarity", false);
	as >> parameter ("wr", rfWeight,"weight for segmentation posterior", false);
	as >> parameter ("ws", segWeight,"weight for segmentation similarity", false);

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
	ImageType::Pointer fixedSegmentationImage =
			ImageUtils<ImageType>::readImage(fixedSegmentationFilename);

	typedef itk::HistogramMatchingImageFilter<
			ImageType,
			ImageType >   MatchingFilterType;
	MatchingFilterType::Pointer matcher = MatchingFilterType::New();
	matcher->SetInput( movingImage );

#if 0
	typedef itk::ImageRegionConstIterator< ImageType > ConstIteratorType;
	typedef itk::ImageRegionIterator< ImageType>       IteratorType;
	ImageType::RegionType inputRegion;
	ImageType::RegionType::IndexType inputStart;
	ImageType::RegionType::SizeType  size;
	//	110x190+120+120
	inputStart[0] = int(0.22869*targetImage->GetLargestPossibleRegion().GetSize()[0]);//::atoi( argv[3] );
	inputStart[1] = int(0.508021*targetImage->GetLargestPossibleRegion().GetSize()[1]);
	size[0]  = int(0.25*targetImage->GetLargestPossibleRegion().GetSize()[0]);
	size[1]  = int(0.320856*targetImage->GetLargestPossibleRegion().GetSize()[1]);
	inputRegion.SetSize( size );
	inputRegion.SetIndex( inputStart );
	ImageType::RegionType outputRegion;
	ImageType::RegionType::IndexType outputStart;
	outputStart[0] = 0;
	outputStart[1] = 0;
	outputRegion.SetSize( size );
	outputRegion.SetIndex( outputStart );
	ImageType::Pointer outputImage = ImageType::New();
	outputImage->SetRegions( outputRegion );
	const ImageType::SpacingType& spacing = targetImage->GetSpacing();
	const ImageType::PointType& inputOrigin = targetImage->GetOrigin();
	double   outputOrigin[ D ];

	for(unsigned int i=0; i< D; i++)
	{
		outputOrigin[i] = inputOrigin[i] + spacing[i] * inputStart[i];
	}


	outputImage->SetSpacing( spacing );
	outputImage->SetOrigin(  outputOrigin );
	outputImage->Allocate();
	ConstIteratorType inputIt(   targetImage, inputRegion  );
	IteratorType      outputIt(  outputImage,         outputRegion );

	inputIt.GoToBegin();
	outputIt.GoToBegin();

	while( !inputIt.IsAtEnd() )
	{
		outputIt.Set(  inputIt.Get()  );
		++inputIt;
		++outputIt;
	}

	matcher->SetReferenceImage( outputImage );
#else


	matcher->SetReferenceImage( targetImage );
#endif
	matcher->SetNumberOfHistogramLevels( 120 );
	matcher->SetNumberOfMatchPoints( 7 );
	//	matcher->ThresholdAtMeanIntensityOn();
	matcher->Update();
	movingImage=matcher->GetOutput();

	typedef BaseLabel<ImageType> BaseLabelType;
	typedef BaseUnaryPotential< BaseLabelType, ImageType > BaseUnaryPotentialType;
	typedef BaseUnaryPotentialType::Pointer BaseUnaryPotentialPointerType;
	typedef GraphModel<BaseUnaryPotentialType,BaseLabelType,ImageType> GraphModelType;

	BaseUnaryPotentialPointerType unaryPot=BaseUnaryPotentialType::New();
	unaryPot->SetFixedImage(targetImage);
	unaryPot->SetMovingImage(movingImage);
	typedef ImageType::SizeType SizeType;
	SizeType spacing;
	spacing.Fill(5);
	GraphModelType Graph(targetImage,spacing);



	LOG<<movingImage->GetLargestPossibleRegion().GetSize()<<std::endl;


	//create Grid
	typedef Grid<ImageType> GridType;
	GridType::LabelType resolution;
	for (int d=0;d<D;++d){
		resolution[d]=5;
	}

	GridType fullimageGrid(targetImage,resolution);
	typedef RegistrationSegmentationLabel<ImageType> RegistrationSegmentationLabelType;
	typedef RegistrationSegmentationLabelConverter<ImageType, RegistrationSegmentationLabelType> RLCType;
	RLCType * RLC=new RLCType(targetImage,movingImage,movingSegmentationImage,2*maxDisplacement+1,2*maxDisplacement+1);


	//	PairwisePotential
	typedef JointEuclideanPairwisePotential<RLCType> PairwisePotentialType;
	PairwisePotentialType::Pointer potentialFunction=PairwisePotentialType::New();
	potentialFunction->SetFixedImage(targetImage);
	potentialFunction->SetGrid(&fullimageGrid);
	potentialFunction->setSegmentationWeight(pairwiseSegmentationWeight);
	potentialFunction->setRegistrationWeight(pairwiseWeight);

	typedef UnarySRSPotentialv2<RLCType> UnaryPotentialType;
	UnaryPotentialType::Pointer unaryFunction=UnaryPotentialType::New();
	unaryFunction->SetWeights(simWeight,rfWeight,segWeight);
	unaryFunction->SetMovingImage(movingImage);
	unaryFunction->SetMovingSegmentationImage(movingSegmentationImage);
	unaryFunction->SetFixedSegmentationImage(fixedSegmentationImage);
	unaryFunction->SetFixedImage(targetImage);
	unaryFunction->setLabelConverter(RLC);
	ImagePointerType classifiedImage=unaryFunction->trainClassifiers();
	ImageUtils<ImageType>::writeImage("classified.png", classifiedImage);

	//	ok what now: create graph! solve graph! save result!Z
	//	for (double p=1;p<4;p+=0.5){
	double p=pairwiseWeight;
	typedef FastPDMRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
	MRFSolverType mrfSolver(targetImage,movingImage,&fullimageGrid,potentialFunction,unaryFunction,unaryWeight,1, true);
	//	typedef TRWS_MRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
	//	MRFSolverType mrfSolver(targetImage,movingImage,&fullimageGrid,potentialFunction,unaryFunction,unaryWeight,p);

	LOG<<"run with p="<<p<<std::endl;
	mrfSolver.optimize();

	//deformed image
	ostringstream deformedFilename;
	deformedFilename<<outputFilename<<"-p"<<p<<".png";
	ImagePointerType deformedImage;
//	deformedImage=RLC->transformImage(movingSegmentationImage,mrfSolver.getLabelImage());
	deformedImage=RLC->transformImage(movingImage,mrfSolver.getLabelImage());
	//	ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedImage);
	ImageUtils<ImageType>::writeImage(outputFilename, deformedImage);
	deformedImage=RLC->transformImage(movingSegmentationImage,mrfSolver.getLabelImage());
	ImageUtils<ImageType>::writeImage("deformedSegmentation.png", deformedImage);

	//deformation
	if (defFilename!=""){
		typedef RLCType::DisplacementFieldType DisplacementFieldType;
		typedef DisplacementFieldType::Pointer DisplacementFieldPointerType;
		DisplacementFieldPointerType defField=RLC->getDisplacementField(mrfSolver.getLabelImage());
		ImageUtils<DisplacementFieldType>::writeImage(defFilename,defField);
	}

	//segmentation
	ostringstream segmentedFilename;
	segmentedFilename<<segmentationOutputFilename<<"-p"<<p<<".png";


	ImagePointerType segmentedImage;
	segmentedImage=RLC->getSegmentationField(mrfSolver.getLabelImage());
	//	ImageUtils<ImageType>::writeImage(segmentedFilename.str().c_str(), segmentedImage);
	ImageUtils<ImageType>::writeImage(segmentationOutputFilename, segmentedImage);


	//	}

	return 1;
}
