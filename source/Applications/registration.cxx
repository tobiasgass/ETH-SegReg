
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
#include "FAST-PD-mrf-optimisation.h"
#include <fenv.h>
#include "TRW-S-Registration.h"
#include "itkHistogramMatchingImageFilter.h"

using namespace std;
using namespace itk;

#define _MANY_LABELS_

int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


	argstream as(argc, argv);
	string targetFilename,movingFilename,outputFilename,deformableFilename,defFilename="";
	double pairwiseWeight=1;
	int displacementSampling=-1;
	double unaryWeight=1;
	int maxDisplacement=10;

	as >> parameter ("t", targetFilename, "target image (file name)", true);
	as >> parameter ("m", movingFilename, "moving image (file name)", true);
	as >> parameter ("o", outputFilename, "output image (file name)", true);
	as >> parameter ("d", deformableFilename, "deformable image (file name)", false);
	as >> parameter ("f", defFilename,"deformation field filename", false);
	as >> parameter ("p", pairwiseWeight,"weight for pairwise potentials", false);
	as >> parameter ("u", unaryWeight,"weight for unary potentials", false);
	as >> parameter ("max", maxDisplacement,"maximum displacement in pixels per axis", false);
	as >> parameter ("n", displacementSampling,"number of samples for each displacement axis", false);
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
	RLCType * RLC=new RLCType(targetImage,movingImage,2*maxDisplacement+1,2*maxDisplacement+1);
	RegistrationLabelType rLabel;
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
//		typedef TRWS_MRFSolver<UnaryPotentialType,PairwisePotentialType> MRFSolverType;
	MRFSolverType mrfSolver(targetImage,movingImage,&fullimageGrid,potentialFunction,unaryFunction,unaryWeight,pairwiseWeight);
	std::cout<<"run"<<std::endl;
	mrfSolver.optimize();

	ImagePointerType transformedImage;
	if (deformableFilename!=""){
		ImageType::Pointer deformableImage =
				ImageUtils<ImageType>::readImage(deformableFilename);
		transformedImage=RLC->transformImage(deformableImage,mrfSolver.getLabelImage());
	}else{
		transformedImage=RLC->transformImage(movingImage,mrfSolver.getLabelImage());
	}

	ImageUtils<ImageType>::writeImage(outputFilename, transformedImage);

	if (defFilename!=""){
		typedef RLCType::DisplacementFieldType DisplacementFieldType;
		typedef DisplacementFieldType::Pointer DisplacementFieldPointerType;

		DisplacementFieldPointerType defField=RLC->getDisplacementField(mrfSolver.getLabelImage());
		ImageUtils<DisplacementFieldType>::writeImage(defFilename,defField);

	}
	return 1;
}
