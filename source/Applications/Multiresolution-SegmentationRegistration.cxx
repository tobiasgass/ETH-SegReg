
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
#include "Graph-ITKStyle.h"
#include "SRSPotential.h"
#include "Potential-Registration-NCC.h"
#include "MRF-FAST-PD.h"
#include "Potential-SRS-NCC.h"
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
#include "itkDisplacementFieldCompositionFilter.h"
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

	typedef itk::ImageRegionIterator< ImageType>       IteratorType;

	//	typedef RegistrationLabel<ImageType> BaseLabelType;

	LabelMapperType * labelmapper=new LabelMapperType(nSegmentations,maxDisplacement);

	typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
	typedef SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;


	typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
	typedef ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
	//	typedef NCCSRSUnaryPotential< LabelMapperType, ImageType, SegmentationInterpolatorType,ImageInterpolatorType > BaseUnaryPotentialType;
	//	typedef NCCRegistrationUnaryPotential< LabelMapperType, ImageType, SegmentationInterpolatorType,ImageInterpolatorType > BaseUnaryPotentialType;
	typedef SegmentationRegistrationUnaryPotential< LabelMapperType, ImageType, SegmentationInterpolatorType,ImageInterpolatorType > BaseUnaryPotentialType;
	typedef BaseUnaryPotentialType::Pointer BaseUnaryPotentialPointerType;
	typedef BaseUnaryPotentialType::RadiusType RadiusType;
	//	typedef ITKGraphModel<BaseUnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
	typedef GraphModel<BaseUnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
	ImagePointerType deformedImage,deformedSegmentationImage,segmentationImage;
	deformedImage=ImageUtils<ImageType>::createEmpty(targetImage);
	deformedSegmentationImage=ImageUtils<ImageType>::createEmpty(targetImage);
	segmentationImage=ImageUtils<ImageType>::createEmpty(targetImage);

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
	unaryPot->trainClassifiers();
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
	//	int levels[]={2,4,8,16,32,64,100, 200};
	//	int levels[]={2,4,6,8,32,64,100, 200};
//	int levels[]={2,3,5,9,17,33,32,64,100, 200};
		int levels[]={3,5,9,17,33,32,64,100, 200};

	//		int levels[]={8,16,32,64,128};
	//	int levels[]={64,312};
	int nIterPerLevel=5;
	int iterationCount=0;

	GraphModelType checkerGraph(targetImage,unaryPot,9999999,1, pairwiseSegmentationWeight, pairwiseRegistrationWeight );

	for (int l=0;l<nLevels;++l){
		int level=levels[l];
		double labelScalingFactor=1;

		//at 4th level, we switch to full image grid but allow only 1 displacement in each direction
		if (l==nLevels-1 &&nSegmentations>1){
			//			LabelMapperType * labelmapper2=new LabelMapperType(nSegmentations,maxDisplacement>0?1:0);
			LabelMapperType * labelmapper2=new LabelMapperType(nSegmentations,0);
			level=99999999999;
			labelScalingFactor=0.5;
			nIterPerLevel=1;
		}
		GraphModelType graph(targetImage,unaryPot,level,labelScalingFactor, pairwiseSegmentationWeight, pairwiseRegistrationWeight );
		graph.setGradientImage(fixedSegmentationImage);
		unaryPot->setRadius(graph.getSpacing());
		std::cout<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
		std::cout<<"Current grid size :"<<graph.getGridSize()<<std::endl;
		std::cout<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
		for (int i=0;i<graph.nNodes();++i){
			//			std::cout<<i<<" "<<graph.getGridPositionAtIndex(i)<<" "<<graph.gridToImageIndex(graph.getGridPositionAtIndex(i))<<" "<<graph.getImagePositionAtIndex(i)<<" "<<graph.getIntegerIndex(graph.getGridPositionAtIndex(i))<<std::endl;
		}
		for (int i=0;i<nIterPerLevel;++i,++iterationCount){
			std::cout<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";//std::endl<<std::endl;
			graph.setDisplacementFactor(labelScalingFactor);

			unaryPot->SetDisplacementFactor(graph.getDisplacementFactor());
			unaryPot->SetBaseLabelMap(previousFullDeformation);
			graph.setLabelImage(previousFullDeformation);
			graph.calculateBackProjections();
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
			std::cout<<"]"<<std::endl;

			//initialise interpolator
			//deformation

			fullDeformation=graph.getFullLabelImage(deformation);
			//apply deformation to moving image
			IteratorType fixedIt(targetImage,targetImage->GetLargestPossibleRegion());
			ostringstream gridCosts,imageCosts;
			gridCosts<<"costsGrid-l"<<l<<"-i"<<i<<".png";
			imageCosts<<"costsImage-l"<<l<<"-i"<<i<<".png";
			checkerGraph.setLabelImage(previousFullDeformation);
			//			graph.checkConstraints(deformation,gridCosts.str().c_str());
			//			checkerGraph.checkConstraints(fullDeformation,imageCosts.str());

			typedef itk::DisplacementFieldCompositionFilter<LabelImageType,LabelImageType> CompositionFilterType;
			CompositionFilterType::Pointer composer=CompositionFilterType::New();
			composer->SetInput(0,previousFullDeformation);
			composer->SetInput(1,fullDeformation);
			composer->Update();
			LabelImagePointerType composedDeformation=composer->GetOutput();

			LabelIteratorType labelIt(composedDeformation,composedDeformation->GetLargestPossibleRegion());
			for (fixedIt.GoToBegin(),labelIt.GoToBegin();!fixedIt.IsAtEnd();++fixedIt,++labelIt){
				IndexType index=fixedIt.GetIndex();
				ImageInterpolatorType::ContinuousIndexType idx=unaryPot->getMovingIndex(index);
				BaseLabelType displacement=labelIt.Get();
				//				std::cout<<displacement<<std::endl;
				idx+=LabelMapperType::getDisplacement(displacement);
				if (segmentationInterpolator->IsInsideBuffer(idx)){
					deformedImage->SetPixel(fixedIt.GetIndex(),movingInterpolator->EvaluateAtContinuousIndex(idx));
					deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),segmentationInterpolator->EvaluateAtContinuousIndex(idx));

				}else{
					deformedImage->SetPixel(fixedIt.GetIndex(),0);
					deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),0);
				}
				segmentationImage->SetPixel(fixedIt.GetIndex(),(LabelMapperType::getSegmentation(fullDeformation->GetPixel(index))>0)*65535);
			}
			previousFullDeformation=composedDeformation;
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
		std::cout<<std::endl<<std::endl;
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
