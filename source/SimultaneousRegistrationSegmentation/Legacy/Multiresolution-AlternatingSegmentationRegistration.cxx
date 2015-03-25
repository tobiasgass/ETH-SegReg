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
const unsigned int D=3;
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


	ArgumentParser as(argc, argv);
	string targetFilename,movingFilename,fixedSegmentationFilename, outputDeformedSegmentationFilename,movingSegmentationFilename, outputDeformedFilename,deformableFilename,defFilename="", segmentationOutputFilename;
	double pairwiseRegistrationWeight=1;
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
	as.parameter ("g", fixedSegmentationFilename, "fixed segmentation image (file name)", true);

	as.parameter ("o", outputDeformedFilename, "output image (file name)", true);
	as.parameter ("S", outputDeformedSegmentationFilename, "output image (file name)", true);
	as.parameter ("O", segmentationOutputFilename, "output segmentation image (file name)", true);
	as.parameter ("f", defFilename,"deformation field filename", false);
	as.parameter ("rp", pairwiseRegistrationWeight,"weight for pairwise registration potentials", false);
	as.parameter ("sp", pairwiseSegmentationWeight,"weight for pairwise segmentation potentials", false);

	as.parameter ("u", unaryWeight,"weight for unary potentials", false);
	as.parameter ("max", maxDisplacement,"maximum displacement in pixels per axis", false);
	as.parameter ("wi", simWeight,"weight for intensity similarity", false);
	as.parameter ("wr", rfWeight,"weight for segmentation posterior", false);
	as.parameter ("ws", segWeight,"weight for segmentation similarity", false);

	as.parse();
	
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
	BaseLabelType zeroLabel(0.0);
	previousFullDeformation->FillBuffer(zeroLabel);
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
//	ImagePointerType classified;
//	classified=unaryPot->trainClassifiers();
	//	if (classified)
	//		ImageUtils<ImageType>::writeImage("classified.nii",classified);

	typedef ImageType::SpacingType SpacingType;
	LabelImagePointerType segmentation,deformation;
	int iterationCount=0;
	while (true){
		//overlap (DICE) of old and new segmentation
		double overlap=0.0;
		//estimate segmentation

		{
			unaryPot->setFixedSegmentation(false);
			labelmapper->setSegmentationLabels(2);
			labelmapper->setDisplacementSamples(0);
			SpacingType spacing;
			spacing.Fill(1);
			GraphModelType graph(targetImage,unaryPot,spacing,1, pairwiseSegmentationWeight, pairwiseRegistrationWeight );
			graph.setGradientImage(fixedSegmentationImage);
			unaryPot->SetDisplacementFactor(graph.getDisplacementFactor());
			unaryPot->SetBaseLabelMap(previousFullDeformation);
			graph.setLabelImage(previousFullDeformation);
//			typedef TRWS_MRFSolver<GraphModelType> MRFSolverType;
			typedef TRWS_SimpleMRFSolver<GraphModelType> MRFSolverType;

			//			typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
			MRFSolverType mrfSolver(&graph,1,1, false);
			mrfSolver.optimize();
			//get Labelling
			segmentation=mrfSolver.getLabelImage();
			//update segmentation in final label image
			LabelIteratorType labelIt(segmentation,segmentation->GetLargestPossibleRegion());
			LabelIteratorType newLabelIt(previousFullDeformation,previousFullDeformation->GetLargestPossibleRegion());
			int tp=1,fp=0,fn=0,tn=0;
			for (newLabelIt.GoToBegin(),labelIt.GoToBegin();!labelIt.IsAtEnd();++labelIt,++newLabelIt){
				//get segmentation label from optimisation
				int segmentation=LabelMapperType::getSegmentation(labelIt.Get());
				int oldSegmentation=LabelMapperType::getSegmentation(newLabelIt.Get());
				if (segmentation>0){
					if (oldSegmentation>0){
						tp+=1;
					}
					else{
						fp+=1;
					}
				}
				else{
					if (oldSegmentation>0){
						fn+=1;
					}
					else{
						tn+=1;
					}
				}
				//update segmentation label in global label map
				LabelMapperType::setSegmentation(newLabelIt.Value(),segmentation);
				LabelMapperType::setDisplacement(newLabelIt.Value(),itk::Vector<float,ImageType::ImageDimension>(0.0));
				segmentationImage->SetPixel(newLabelIt.GetIndex(),segmentation*65535);

			}
			LOG<<"TP :"<<tp<<" "<<fp<<" "<<tn<<" "<<fn<<std::endl;
			overlap=2.0*tp/(2*tp+fp+fn);
		}
		LOG<<"Overlap of old and new segmentation at iteration "<<iterationCount<<" is :"<<overlap<<std::endl;
		if (overlap>0.99) break;
		//end segmentation

		//estimate registration
		{
			labelmapper->setSegmentationLabels(1);
			labelmapper->setDisplacementSamples(displacementSampling);
			int nLevels=2;

			//	int levels[]={4,16,40,100,200};
			//	int levels[]={1,2,4,8,20,40,100, 200};
//			int levels[]={2,4,16,32,64,100, 200};
			int levels[]={1,16,50, 200};

			//	int levels[]={8,16,32,64,128};
			//	int levels[]={64,312};
			int nIterPerLevel=3;
			for (int l=0;l<nLevels;++l){
				int level=levels[l];
				SpacingType spacing;
				int minSpacing=9999999999999;
				double labelScalingFactor=1;
				if (l>0)labelScalingFactor=5;

				for (int d=0;d<ImageType::ImageDimension;++d){
					if(targetImage->GetLargestPossibleRegion().GetSize()[d]/level <minSpacing){
						minSpacing=targetImage->GetLargestPossibleRegion().GetSize()[d]/level;
						//				divisor=1.0*targetImage->GetLargestPossibleRegion().GetSize()[d]/minSpacing;
					}
				}
				//		LOG<<divisor<<std::endl;
				for (int d=0;d<ImageType::ImageDimension;++d){
					int div=targetImage->GetLargestPossibleRegion().GetSize()[d]/minSpacing;
					spacing[d]=int(1.0*targetImage->GetLargestPossibleRegion().GetSize()[d]/div);
				}				//				if (l==nLevels-1){
				//					//			LabelMapperType * labelmapper2=new LabelMapperType(nSegmentations,maxDisplacement>0?1:0);
				//					labelmapper->setSegmentationLabels(1);
				//					labelmapper->setDisplacementSamples(2);
				//					spacing.Fill(1.0);
				//					labelScalingFactor=0.5;
				//					nIterPerLevel=2;
				//				}

				//				LOG<<"spacing at level "<<level<<" :"<<spacing<<std::endl;

				for (int i=0;i<nIterPerLevel;++i){
					//					LOG<<std::endl<<std::endl<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<std::endl<<std::endl;

					GraphModelType graph(targetImage,unaryPot,spacing,labelScalingFactor, pairwiseSegmentationWeight, pairwiseRegistrationWeight );
					graph.setGradientImage(fixedSegmentationImage);
					//			for (int f=0;f<graph.nNodes();++f){
					//				LOG<<f<<" "<<graph.getGridPositionAtIndex(f)<<" "<<graph.getImagePositionAtIndex(f)<<std::endl;
					//			}
					unaryPot->SetDisplacementFactor(graph.getDisplacementFactor());
					unaryPot->SetBaseLabelMap(previousFullDeformation);
					unaryPot->setFixedSegmentation(true);
					graph.setLabelImage(previousFullDeformation);

					//					LOG<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
					//					LOG<<"Current grid size :"<<graph.getGridSize()<<std::endl;
					//					LOG<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
					//	ok what now: create graph! solve graph! save result!Z
					typedef TRWS_MRFSolver<GraphModelType> MRFSolverType;
					//			typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
					MRFSolverType mrfSolver(&graph,1,1, false);
					mrfSolver.optimize();

					//Apply/interpolate Transformation

					//Get label image (deformation)
					deformation=mrfSolver.getLabelImage();
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
					//					LOGV(1)<<"interpolating deformation field"<<std::endl;
					resampler->Update();
					//			if (defFilename!=""){
					//				//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
					//				ostringstream labelfield;
					//				labelfield<<defFilename<<"-l"<<l<<"-i"<<i<<".mha";
					//				ImageUtils<LabelImageType>::writeImage(labelfield.str().c_str(),deformation);
					//				ostringstream labelfield2;
					//				labelfield2<<defFilename<<"FULL-l"<<l<<"-i"<<i<<".mha";
					//				ImageUtils<LabelImageType>::writeImage(labelfield2.str().c_str(),resampler->GetOutput());
					//				//
					//			}
					//apply deformation to moving image

					IteratorType fixedIt(targetImage,targetImage->GetLargestPossibleRegion());
					fullDeformation=resampler->GetOutput();
					graph.checkConstraints(fullDeformation);
					LabelIteratorType labelIt(fullDeformation,fullDeformation->GetLargestPossibleRegion());
					LabelIteratorType newLabelIt(previousFullDeformation,previousFullDeformation->GetLargestPossibleRegion());
					for (newLabelIt.GoToBegin(),fixedIt.GoToBegin(),labelIt.GoToBegin();!fixedIt.IsAtEnd();++fixedIt,++labelIt,++newLabelIt){
						ImageInterpolatorType::ContinuousIndexType idx=unaryPot->getMovingIndex(fixedIt.GetIndex());

						if (false){
							LOG<<"Current displacement at "<<fixedIt.GetIndex()<<" ="<<LabelMapperType::getDisplacement(labelIt.Get())<<" with factors:"<<graph.getDisplacementFactor()<<" ="<<LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
							LOG<<"Total displacement including previous iterations ="<<LabelMapperType::getDisplacement(newLabelIt.Get())+LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
							LOG<<"Resulting point in moving image :"<<idx+LabelMapperType::getDisplacement(newLabelIt.Get())+LabelMapperType::getDisplacement(labelIt.Get()).elementMult(graph.getDisplacementFactor())<<std::endl;
							LOG<<"Total Label :"<<labelIt.Get()<<std::endl;
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

						int oldSegmentation=LabelMapperType::getSegmentation(segmentation->GetPixel(fixedIt.GetIndex()));
						LabelMapperType::setSegmentation(newLabelIt.Value(),oldSegmentation);
						LabelMapperType::setDisplacement(newLabelIt.Value(),LabelMapperType::getDisplacement(displacement));

					}
					labelScalingFactor*=0.8;


				}
			}
		}//end registration
#if 0
		ostringstream deformedFilename;
		deformedFilename<<outputDeformedFilename<<"-l"<<l<<"-i"<<i<<".nii";
		ostringstream deformedSegmentationFilename;
		deformedSegmentationFilename<<outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<".nii";
		ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedImage);
		ostringstream tmpSegmentationFilename;
		tmpSegmentationFilename<<segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<".nii";
		ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(), segmentationImage);
		ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(), deformedSegmentationImage);
		//deformation
		if (defFilename!=""){
			ostringstream tmpDeformationFilename;
			tmpDeformationFilename<<defFilename<<"-l"<<l<<"-i"<<i<<".mha";
			//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
			ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
			//
		}
#endif

		++iterationCount;
	}//while
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
