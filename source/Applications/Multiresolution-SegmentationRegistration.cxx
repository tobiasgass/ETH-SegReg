
#include <stdio.h>
#include <iostream>

#include "argstream.h"
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
#include "SRSPotential.h"
#include "MRF-FAST-PD.h"
#include "MRF-TRW-S.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageIterator.h"
#include "itkImageConstIteratorWithIndex.h"
#include "itkImageIteratorWithIndex.h"
#include "itkImageConstIterator.h"

#include <itkImageAdaptor.h>
#include <itkAddPixelAccessor.h>

#include "itkBSplineInterpolateImageFunction.h"

using namespace std;
using namespace itk;

#define _MANY_LABELS_
#define DOUBLEPAIRWISE
typedef unsigned short PixelType;
const unsigned int D=2;
typedef Image<PixelType,D> ImageType;
typedef ImageType::IndexType IndexType;
typedef itk::Vector<float,D+1> BaseLabelType;
//typedef BaseLabelMapper<ImageType,BaseLabelType> LabelMapperType;
typedef SparseLabelMapper<ImageType,BaseLabelType> LabelMapperType;
template<> int LabelMapperType::nLabels=-1;
template<> int LabelMapperType::nDisplacements=-1;
template<> int LabelMapperType::nSegmentations=-1;
template<> int LabelMapperType::nDisplacementSamples=-1;
template<> int LabelMapperType::k=-1;



int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


	argstream as(argc, argv);
	string targetFilename,movingFilename,fixedSegmentationFilename, outputDeformedSegmentationFilename,movingSegmentationFilename, outputDeformedFilename,deformableFilename,defFilename="", segmentationOutputFilename;
	double pairwiseRegistrationWeight=1;
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

	as >> parameter ("o", outputDeformedFilename, "output image (file name)", true);
	as >> parameter ("S", outputDeformedSegmentationFilename, "output image (file name)", true);
	as >> parameter ("O", segmentationOutputFilename, "output segmentation image (file name)", true);
	as >> parameter ("f", defFilename,"deformation field filename", false);
	as >> parameter ("rp", pairwiseRegistrationWeight,"weight for pairwise registration potentials", false);
	as >> parameter ("sp", pairwiseSegmentationWeight,"weight for pairwise segmentation potentials", false);

	as >> parameter ("u", unaryWeight,"weight for unary potentials", false);
	as >> parameter ("max", maxDisplacement,"maximum displacement in pixels per axis", false);
	as >> parameter ("wi", simWeight,"weight for intensity similarity", false);
	as >> parameter ("wr", rfWeight,"weight for segmentation posterior", false);
	as >> parameter ("ws", segWeight,"weight for segmentation similarity", false);

	as >> help();
	as.defaultErrorHandling();
	int nSegmentations=2;
	if (segWeight==0 && pairwiseSegmentationWeight==0 && rfWeight==0 ){
		nSegmentations=1;
	}
	if (displacementSampling==-1) displacementSampling=maxDisplacement;
	bool verbose=false;
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

	matcher->SetNumberOfHistogramLevels( 64 );
	matcher->SetNumberOfMatchPoints( 2 );
	//	matcher->ThresholdAtMeanIntensityOn();
	matcher->Update();
	//	movingImage=matcher->GetOutput();

	//	typedef RegistrationLabel<ImageType> BaseLabelType;

	LabelMapperType * labelmapper=new LabelMapperType(nSegmentations,maxDisplacement);

	typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
	typedef SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;


	typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
	typedef ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
	typedef SegmentationRegistrationUnaryPotential< LabelMapperType, ImageType, SegmentationInterpolatorType,ImageInterpolatorType > BaseUnaryPotentialType;
	typedef BaseUnaryPotentialType::Pointer BaseUnaryPotentialPointerType;
	typedef GraphModel<BaseUnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
	ImagePointerType deformedImage,deformedSegmentationImage,segmentationImage;
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
	segmentationImage=ImageType::New();
	segmentationImage->SetRegions(targetImage->GetLargestPossibleRegion());
	segmentationImage->SetOrigin(targetImage->GetOrigin());
	segmentationImage->SetSpacing(targetImage->GetSpacing());
	segmentationImage->SetDirection(targetImage->GetDirection());
	segmentationImage->Allocate();
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
	itk::Vector<float, D+1> tmpVox(0.0);
	previousFullDeformation->FillBuffer(tmpVox);
	BaseUnaryPotentialPointerType unaryPot=BaseUnaryPotentialType::New();
	unaryPot->SetFixedImage(targetImage);
	ImageInterpolatorPointerType movingInterpolator=ImageInterpolatorType::New();
	movingInterpolator->SetInputImage(movingImage);
	SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
	segmentationInterpolator->SetInputImage(movingSegmentationImage);
	unaryPot->SetMovingImage(movingImage);
	unaryPot->SetMovingInterpolator(movingInterpolator);
	unaryPot->SetSegmentationInterpolator(segmentationInterpolator);
	unaryPot->SetWeights(simWeight,rfWeight,segWeight);
	unaryPot->SetMovingSegmentation(movingSegmentationImage);
	ImagePointerType classified;
	classified=unaryPot->trainClassifiers();
	//	if (classified)
	//		ImageUtils<ImageType>::writeImage("classified.nii",classified);

	typedef ImageType::SpacingType SpacingType;
	int nLevels=4;
	if (nSegmentations>1) nLevels++;
	nLevels=maxDisplacement>0?nLevels:1;
	//one more level for segmentation


//	int levels[]={1,16,50, 200};

//	double levels[]={1.5,2,4,16,32,64,100, 200};
//	int levels[]={5,9,27,91,100, 200};
	int levels[]={3,16,32,64,100, 200};
//		int levels[]={8,16,32,64,128};
	//	int levels[]={64,312};
	int nIterPerLevel=5;
	int iterationCount=0;
	for (int l=0;l<nLevels;++l){
		double level=levels[l];
		SpacingType spacing;
		double labelScalingFactor=1;
//		if (l>0)labelScalingFactor=5;
		int minSpacing=999999;
		for (int d=0;d<ImageType::ImageDimension;++d){
			std::cout<<level<<" "<<targetImage->GetLargestPossibleRegion().GetSize()[d]/(level-1)<<std::endl;
			if(targetImage->GetLargestPossibleRegion().GetSize()[d]/(level-1) < minSpacing){
				minSpacing=targetImage->GetLargestPossibleRegion().GetSize()[d]/(level-1)-1;
//				divisor=1.0*targetImage->GetLargestPossibleRegion().GetSize()[d]/minSpacing;
			}
		}
		std::cout<<minSpacing<<std::endl;
		for (int d=0;d<ImageType::ImageDimension;++d){
			int div=targetImage->GetLargestPossibleRegion().GetSize()[d]/minSpacing;
			div=div>0?div:1;
			spacing[d]=(1.0*targetImage->GetLargestPossibleRegion().GetSize()[d]/div)-1.0/(level-1);
			std::cout<<targetImage->GetLargestPossibleRegion().GetSize()[d]<<" "<<minSpacing<<" "<<div<<" "<<spacing[d]<<std::endl;
		}
		//at 4th level, we switch to full image grid but allow only 1 displacement in each direction
		if (l==nLevels-1 &&nSegmentations>1){
			//			LabelMapperType * labelmapper2=new LabelMapperType(nSegmentations,maxDisplacement>0?1:0);
			LabelMapperType * labelmapper2=new LabelMapperType(nSegmentations,0);
			spacing.Fill(1.0);
			labelScalingFactor=0.5;
			nIterPerLevel=1;
		}
		std::cout<<"spacing at level "<<level<<" :"<<spacing<<std::endl;

		for (int i=0;i<nIterPerLevel;++i,++iterationCount){
			std::cout<<std::endl<<std::endl<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<std::endl<<std::endl;

			GraphModelType graph(targetImage,unaryPot,spacing,labelScalingFactor, pairwiseSegmentationWeight, pairwiseRegistrationWeight );
//			for (int n=0;n<graph.nNodes();++n){
//				std::cout<<n<<" "<<graph.getImagePositionAtIndex(n)<<" "<<graph.getGridPositionAtIndex(n)<<" "<<graph.getIntegerIndex(graph.getGridPositionAtIndex(n))<<std::endl;
//			}
			graph.setGradientImage(fixedSegmentationImage);

			unaryPot->SetDisplacementFactor(graph.getDisplacementFactor());
			if (iterationCount){
				unaryPot->SetBaseLabelMap(previousFullDeformation);
				graph.setLabelImage(previousFullDeformation);
			}
			std::cout<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
			std::cout<<"Current grid size :"<<graph.getGridSize()<<std::endl;
			std::cout<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
			//	ok what now: create graph! solve graph! save result!Z
			LabelImagePointerType deformation;
			if (nIterPerLevel>1){
				typedef TRWS_MRFSolver<GraphModelType> MRFSolverType;
				//			typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
				MRFSolverType mrfSolver(&graph,1,1, false);
				mrfSolver.optimize();
				deformation=mrfSolver.getLabelImage();

			}else{
				//pixel level grid, only use simple MRF
				typedef TRWS_SimpleMRFSolver<GraphModelType> MRFSolverType;
							MRFSolverType mrfSolver(&graph,1,1, false);
				mrfSolver.optimize();
				deformation=mrfSolver.getLabelImage();
			}
			//initialise interpolator
			//deformation

			LabelInterpolatorPointerType labelInterpolator=LabelInterpolatorType::New();
			labelInterpolator->SetInputImage(deformation);
			//initialise resampler

			LabelResampleFilterType::Pointer resampler = LabelResampleFilterType::New();
			//resample deformation field to fixed image dimension
			resampler->SetInput( deformation );
			resampler->SetInterpolator( labelInterpolator );
			resampler->SetOutputOrigin(graph.getOrigin());//targetImage->GetOrigin());
			resampler->SetOutputSpacing ( targetImage->GetSpacing() );
			resampler->SetOutputDirection ( targetImage->GetDirection() );
			resampler->SetSize ( targetImage->GetLargestPossibleRegion().GetSize() );
			if (verbose) std::cout<<"interpolating deformation field"<<std::endl;
			resampler->Update();
			fullDeformation=resampler->GetOutput();


			//apply deformation to moving image
			IteratorType fixedIt(targetImage,targetImage->GetLargestPossibleRegion());
			graph.checkConstraints(fullDeformation);
			LabelIteratorType labelIt(fullDeformation,fullDeformation->GetLargestPossibleRegion());
			LabelIteratorType newLabelIt(previousFullDeformation,previousFullDeformation->GetLargestPossibleRegion());
			for (newLabelIt.GoToBegin(),fixedIt.GoToBegin(),labelIt.GoToBegin();!fixedIt.IsAtEnd();++fixedIt,++labelIt,++newLabelIt){
				ImageInterpolatorType::ContinuousIndexType idx=unaryPot->getMovingIndex(fixedIt.GetIndex());

				if (false){
					std::cout<<"Current displacement at "<<fixedIt.GetIndex()<<" ="<<LabelMapperType::getDisplacement(labelIt.Get())<<" with factors:"<<graph.getDisplacementFactor()<<" ="<<LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
					std::cout<<"Total displacement including previous iterations ="<<LabelMapperType::getDisplacement(newLabelIt.Get())+LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
					std::cout<<"Resulting point in moving image :"<<idx+LabelMapperType::getDisplacement(newLabelIt.Get())+LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
					std::cout<<"Total Label :"<<labelIt.Get()<<std::endl;
				}
				BaseLabelType displacement=LabelMapperType::scaleDisplacement(labelIt.Get(),graph.getDisplacementFactor());
				if (iterationCount){
					displacement+=(newLabelIt.Get());
				}
				idx+=LabelMapperType::getDisplacement(displacement);

				if (segmentationInterpolator->IsInsideBuffer(idx)){
					deformedImage->SetPixel(fixedIt.GetIndex(),movingInterpolator->EvaluateAtContinuousIndex(idx));
					deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),segmentationInterpolator->EvaluateAtContinuousIndex(idx));

				}else{
					deformedImage->SetPixel(fixedIt.GetIndex(),0);
					deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),0);
				}
				segmentationImage->SetPixel(fixedIt.GetIndex(),LabelMapperType::getSegmentation(labelIt.Get())*65535);
				newLabelIt.Set(displacement);

			}
			IndexType idx={{0,0}};
					std::cout<<deformation->GetPixel(idx)<<" "<<fullDeformation->GetPixel(idx)<<" "<<previousFullDeformation->GetPixel(idx)<<std::endl;
			labelScalingFactor*=0.8;
#if 1
			ostringstream deformedFilename;
			deformedFilename<<outputDeformedFilename<<"-l"<<l<<"-i"<<i<<".png";
			ostringstream deformedSegmentationFilename;
			deformedSegmentationFilename<<outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<".png";
			ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedImage);
			ostringstream tmpSegmentationFilename;
			tmpSegmentationFilename<<segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<".png";
			ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(), segmentationImage);
			ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(), deformedSegmentationImage);
			//deformation
			if (defFilename!=""){
				ostringstream tmpDeformationFilename;
				tmpDeformationFilename<<defFilename<<"-l"<<l<<"-i"<<i<<".mha";
				//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
				ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
//				ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),				deformation);

				//
			}
#endif

		}
	}
	ImageUtils<ImageType>::writeImage(outputDeformedFilename, deformedImage);
	ImageUtils<ImageType>::writeImage(segmentationOutputFilename, segmentationImage);
	ImageUtils<ImageType>::writeImage(outputDeformedSegmentationFilename, deformedSegmentationImage);


	//deformation
	if (defFilename!=""){
		//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
		ImageUtils<LabelImageType>::writeImage(defFilename,previousFullDeformation);
		//
	}



	//	}

	return 1;
}
