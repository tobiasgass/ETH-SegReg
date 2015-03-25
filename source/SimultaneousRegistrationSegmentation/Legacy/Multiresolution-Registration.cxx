#include "Log.h"

#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include <itkNumericTraitsVectorPixel.h>


#include <fenv.h>
#include <sstream>
#include "itkHistogramMatchingImageFilter.h"
#include "Graph.h"
#include "BaseLabel.h"
#include "BasePotential.h"
#include "MRF-FAST-PD.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

using namespace std;
using namespace itk;

#define _MANY_LABELS_
#define DOUBLEPAIRWISE
typedef unsigned short PixelType;
const unsigned int D=2;
typedef Image<PixelType,D> ImageType;
typedef ImageType::IndexType IndexType;
typedef itk::Vector<float,D> BaseLabelType;
typedef SparseLabelMapper<ImageType,BaseLabelType> LabelMapperType;
template<> int LabelMapperType::nLabels=-1;
template<> int LabelMapperType::nDisplacements=-1;
template<> int LabelMapperType::nSegmentations=-1;
template<> int LabelMapperType::nDisplacementSamples=-1;
template<> int LabelMapperType::k=-1;



int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


	ArgumentParser as(argc, argv);
	string targetFilename,movingFilename,fixedSegmentationFilename, movingSegmentationFilename, outputFilename,deformableFilename,defFilename="", segmentationOutputFilename;
	double pairwiseWeight=1;
	double pairwiseSegmentationWeight=1;
	int displacementSampling=-1;
	double unaryWeight=1;
	int maxDisplacement=10;
	double simWeight=1;
	double rfWeight=1;
	double segWeight=1;

	as.parameter ("t", targetFilename, "target image (file name)", true);
	as.parameter ("m", movingFilename, "moving image (file name)", true);
	as.parameter ("s", movingSegmentationFilename, "moving segmentation image (file name)", true);
	as.parameter ("g", fixedSegmentationFilename, "some gradient of the fixed (file name)", true);

	as.parameter ("o", outputFilename, "output image (file name)", true);
	as.parameter ("O", segmentationOutputFilename, "output segmentation image (file name)", true);
	as.parameter ("f", defFilename,"deformation field filename", false);
	as.parameter ("p", pairwiseWeight,"weight for pairwise potentials", false);
	as.parameter ("sp", pairwiseSegmentationWeight,"weight for pairwise segmentation potentials", false);

	as.parameter ("u", unaryWeight,"weight for unary potentials", false);
	as.parameter ("max", maxDisplacement,"maximum displacement in pixels per axis", false);
	as.parameter ("wi", simWeight,"weight for intensity similarity", false);
	as.parameter ("wr", rfWeight,"weight for segmentation posterior", false);
	as.parameter ("ws", segWeight,"weight for segmentation similarity", false);

	as.parse();
	

	if (displacementSampling==-1) displacementSampling=maxDisplacement;
	bool verbose=true;
	//typedefs

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
	typedef itk::ImageRegionIterator< ImageType>       IteratorType;

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

	matcher->SetNumberOfHistogramLevels( 30 );
	matcher->SetNumberOfMatchPoints( 7 );
	//	matcher->ThresholdAtMeanIntensityOn();
	matcher->Update();
	movingImage=matcher->GetOutput();

	//	typedef RegistrationLabel<ImageType> BaseLabelType;

	LabelMapperType * labelmapper=new LabelMapperType(1,maxDisplacement);
	for (int l=0;l<LabelMapperType::nLabels;++l){
		LOG<<l<<" "<<LabelMapperType::getLabel(l)<<" "<<LabelMapperType::getIndex(LabelMapperType::getLabel(l))<<std::endl;
	}
	typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
	typedef SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;


	typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
	typedef ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
	typedef RegistrationUnaryPotential< LabelMapperType, ImageType, ImageInterpolatorType > BaseUnaryPotentialType;
	typedef BaseUnaryPotentialType::Pointer BaseUnaryPotentialPointerType;
	typedef GraphModel<BaseUnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
	ImagePointerType deformedImage,deformedSegmentationImage;
	deformedImage=ImageType::New();
	deformedImage->SetRegions(targetImage->GetLargestPossibleRegion());
	deformedImage->SetOrigin(targetImage->GetOrigin());
	deformedImage->SetSpacing(targetImage->GetSpacing());
	deformedImage->SetDirection(targetImage->GetDirection());
	deformedImage->Allocate();
	deformedSegmentationImage=ImageType::New();
	deformedSegmentationImage->SetRegions(targetImage->GetLargestPossibleRegion());
	deformedSegmentationImage->SetOrigin(targetImage->GetOrigin());
	deformedSegmentationImage->SetSpacing(targetImage->GetSpacing());
	deformedSegmentationImage->SetDirection(targetImage->GetDirection());
	deformedSegmentationImage->Allocate();
	typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
	typedef MRFSolverType::LabelImageType LabelImageType;
	typedef itk::ImageRegionIterator< LabelImageType>       LabelIteratorType;
	typedef MRFSolverType::LabelImagePointerType LabelImagePointerType;
	typedef VectorLinearInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
	typedef LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
	typedef itk::VectorResampleImageFilter< LabelImageType , LabelImageType>	LabelResampleFilterType;
	LabelImagePointerType fullDeformation,previousFullDeformation;
	previousFullDeformation=LabelImageType::New();
	previousFullDeformation->SetRegions(targetImage->GetLargestPossibleRegion());
	previousFullDeformation->SetOrigin(targetImage->GetOrigin());
	previousFullDeformation->SetSpacing(targetImage->GetSpacing());
	previousFullDeformation->SetDirection(targetImage->GetDirection());
	previousFullDeformation->Allocate();
	BaseUnaryPotentialPointerType unaryPot=BaseUnaryPotentialType::New();
	unaryPot->SetFixedImage(targetImage);
	ImageInterpolatorPointerType movingInterpolator=ImageInterpolatorType::New();
	SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
	segmentationInterpolator->SetInputImage(movingSegmentationImage);
	//unaryPot->SetMovingImage(movingImage);
	unaryPot->SetMovingInterpolator(movingInterpolator);
	typedef ImageType::SpacingType SpacingType;
	int nLevels=4;
	int levels[]={2,4,8,40};

	for (int l=0;l<nLevels;++l){
		int level=levels[l];
		SpacingType spacing;
		for (int d=0;d<ImageType::ImageDimension;++d){
			spacing[d]=targetImage->GetLargestPossibleRegion().GetSize()[d]/level;
		}
		LOG<<"spacing at level "<<level<<" :"<<spacing<<std::endl;

		double labelScalingFactor=1;
		for (int i=0;i<5;++i){
			LOG<<std::endl<<std::endl<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<std::endl<<std::endl;
			movingInterpolator->SetInputImage(movingImage);

			GraphModelType graph(targetImage,unaryPot,spacing,labelScalingFactor,0,pairwiseWeight);
			unaryPot->SetDisplacementFactor(graph.getDisplacementFactor());
			unaryPot->SetBaseLabelMap(previousFullDeformation);
			graph.setLabelImage(previousFullDeformation);
			LOG<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
			LOG<<"Current grid size :"<<graph.getGridSize()<<std::endl;
			LOG<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
			//	ok what now: create graph! solve graph! save result!Z
			typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
			MRFSolverType mrfSolver(&graph,1,1, false);
			mrfSolver.optimize();

			//Apply/interpolate Transformation

			//Get label image (deformation)
			LabelImagePointerType deformation=mrfSolver.getLabelImage();
			//initialise interpolator

			LabelInterpolatorPointerType labelInterpolator=LabelInterpolatorType::New();
			labelInterpolator->SetInputImage(deformation);
			//initialise resampler

			LabelResampleFilterType::Pointer resampler = LabelResampleFilterType::New();
			//resample deformation field to fixed image dimension
			resampler->SetInput( deformation );
			resampler->SetInterpolator( labelInterpolator );
			resampler->SetOutputOrigin ( targetImage->GetOrigin() );
			//	resampler->SetOutputSpacing ( targetImage->GetSpacing() );
			resampler->SetOutputDirection ( targetImage->GetDirection() );
			resampler->SetSize ( targetImage->GetLargestPossibleRegion().GetSize() );
			LOGV(1)<<"interpolating deformation field"<<std::endl;
			resampler->Update();
			//apply deformation to moving image

			IteratorType fixedIt(targetImage,targetImage->GetLargestPossibleRegion());
			fullDeformation=resampler->GetOutput();
			LabelIteratorType labelIt(fullDeformation,fullDeformation->GetLargestPossibleRegion());
			LabelIteratorType newLabelIt(previousFullDeformation,previousFullDeformation->GetLargestPossibleRegion());

			for (newLabelIt.GoToBegin(),fixedIt.GoToBegin(),labelIt.GoToBegin();!fixedIt.IsAtEnd();++fixedIt,++labelIt,++newLabelIt){
				ImageInterpolatorType::ContinuousIndexType idx(fixedIt.GetIndex());

				if (false){
					LOG<<"Current displacement at "<<fixedIt.GetIndex()<<" ="<<LabelMapperType::getDisplacement(labelIt.Get())<<" with factors:"<<graph.getDisplacementFactor()<<" ="<<LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
					LOG<<"Total displacement including previous iterations ="<<LabelMapperType::getDisplacement(newLabelIt.Get())+LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
					LOG<<"Resulting point in moving image :"<<idx+LabelMapperType::getDisplacement(newLabelIt.Get())+LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
				}
				idx+=LabelMapperType::getDisplacement(newLabelIt.Get());
				idx+=LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(labelIt.Get(),graph.getDisplacementFactor()));
				newLabelIt.Set(newLabelIt.Get()+labelIt.Get().elementMult(graph.getDisplacementFactor()));
				deformedImage->SetPixel(fixedIt.GetIndex(),movingInterpolator->EvaluateAtContinuousIndex(idx));
				deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),segmentationInterpolator->EvaluateAtContinuousIndex(idx));
			}
			labelScalingFactor*=0.7;
			ostringstream deformedFilename;
			deformedFilename<<outputFilename<<"-l"<<l<<"-i"<<i<<".nii";
			//			ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedSegmentationImage);

		}
	}
	ImageUtils<ImageType>::writeImage(outputFilename, deformedImage);
	ImageUtils<ImageType>::writeImage(segmentationOutputFilename, deformedSegmentationImage);


	//deformation
	if (defFilename!=""){
		//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
		ImageUtils<LabelImageType>::writeImage(defFilename,previousFullDeformation);
		//
	}

#if 0
	//deformed image
	ostringstream deformedFilename;
	deformedFilename<<outputFilename<<"-p"<<p<<".nii";
	ImagePointerType deformedSegmentationImage;
	//	deformedSegmentationImage=RLC->transformImage(movingSegmentationImage,mrfSolver.getLabelImage());
	deformedSegmentationImage=RLC->transformImage(movingImage,mrfSolver.getLabelImage());
	//	ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedSegmentationImage);
	ImageUtils<ImageType>::writeImage(outputFilename, deformedSegmentationImage);
	deformedSegmentationImage=RLC->transformImage(movingSegmentationImage,mrfSolver.getLabelImage());
	ImageUtils<ImageType>::writeImage("deformedSegmentation.nii", deformedSegmentationImage);


	//segmentation
	ostringstream segmentedFilename;
	segmentedFilename<<segmentationOutputFilename<<"-p"<<p<<".nii";


	ImagePointerType segmentedImage;
	segmentedImage=RLC->getSegmentationField(mrfSolver.getLabelImage());
	//	ImageUtils<ImageType>::writeImage(segmentedFilename.str().c_str(), segmentedImage);
	ImageUtils<ImageType>::writeImage(segmentationOutputFilename, segmentedImage);
#endif

	//	}

	return 1;
}
