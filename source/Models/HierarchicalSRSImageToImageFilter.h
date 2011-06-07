/*less
 * HierarchicalSRSImageToImageFilter.h
 *
 *  Created on: Apr 12, 2011
 *      Author: gasst
 */

#ifndef HIERARCHICALSRSIMAGETOIMAGEFILTER_H_
#define HIERARCHICALSRSIMAGETOIMAGEFILTER_H_
#include "SRSConfig.h"
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
//#include "Potential-Registration-NCC.h"
#include "MRF-FAST-PD.h"
#include "Potential-SRS-NCC.h"
#include "MRF-TRW-S.h"
#include "MRF-GC.h"
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
using namespace itk;
template<class TImage, class TLabelMapper, class TUnaryPotential>
class HierarchicalSRSImageToImageFilter{
public:
	typedef TImage ImageType;
	typedef TLabelMapper LabelMapperType;
	typedef TUnaryPotential UnaryPotentialType;
	typedef typename LabelMapperType::LabelType LabelType;
	static const int D=ImageType::ImageDimension;
private:
	SRSConfig m_config;
public:
	HierarchicalSRSImageToImageFilter(SRSConfig c){
		m_config=c;
	}
	bool run(){

		//typedef typename s
		typedef typename  ImageType::Pointer ImagePointerType;
		typedef typename ImageType::IndexType IndexType;

		//	typedef typename  Image<LabelType> LabelImageType;
		//read input images
		ImagePointerType targetImage =
				ImageUtils<ImageType>::readImage(m_config.targetFilename);
		ImagePointerType movingImage =
				ImageUtils<ImageType>::readImage(m_config.movingFilename);
		ImagePointerType movingSegmentationImage =
				ImageUtils<ImageType>::readImage(m_config.movingSegmentationFilename);
		ImagePointerType fixedSegmentationImage =
				ImageUtils<ImageType>::readImage(m_config.fixedSegmentationFilename);

		typedef typename  itk::ImageRegionIterator< ImageType>       IteratorType;

		//	typedef typename  RegistrationLabel<ImageType> BaseLabelType;


		typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
		typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;


		typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
		typedef typename  ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
		typedef typename  UnaryPotentialType::Pointer UnaryPotentialPointerType;
		typedef typename  UnaryPotentialType::RadiusType RadiusType;
        //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
        typedef  GraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
		ImagePointerType deformedImage,deformedSegmentationImage,segmentationImage;
		deformedImage=ImageUtils<ImageType>::createEmpty(targetImage);
		deformedSegmentationImage=ImageUtils<ImageType>::createEmpty(targetImage);
		segmentationImage=ImageUtils<ImageType>::createEmpty(targetImage);

		typedef  NewFastPDMRFSolver<GraphModelType> MRFSolverType;
		typedef typename  MRFSolverType::LabelImageType LabelImageType;
		typedef   itk::ImageRegionIterator< LabelImageType>       LabelIteratorType;
		typedef typename  MRFSolverType::LabelImagePointerType LabelImagePointerType;
		typedef   VectorLinearInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
		typedef typename  LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
		typedef   itk::VectorResampleImageFilter< LabelImageType , LabelImageType>	LabelResampleFilterType;
		LabelImagePointerType fullDeformation,previousFullDeformation;
		previousFullDeformation=LabelImageType::New();
		previousFullDeformation->SetRegions(targetImage->GetLargestPossibleRegion());
		previousFullDeformation->SetOrigin(targetImage->GetOrigin());
		previousFullDeformation->SetSpacing(targetImage->GetSpacing());
		previousFullDeformation->SetDirection(targetImage->GetDirection());
		previousFullDeformation->Allocate();
		Vector<float, D+1> tmpVox(0.0);
		previousFullDeformation->FillBuffer(tmpVox);
		UnaryPotentialPointerType unaryPot=UnaryPotentialType::New();
		unaryPot->SetFixedImage(targetImage);
		ImageInterpolatorPointerType movingInterpolator=ImageInterpolatorType::New();
		movingInterpolator->SetInputImage(movingImage);
		SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
		segmentationInterpolator->SetInputImage(movingSegmentationImage);
		unaryPot->SetMovingImage(movingImage);
		unaryPot->SetMovingInterpolator(movingInterpolator);
		unaryPot->SetSegmentationInterpolator(segmentationInterpolator);
		unaryPot->SetWeights(m_config.simWeight,m_config.rfWeight,m_config.segWeight);
		unaryPot->SetMovingSegmentation(movingSegmentationImage);
		if (m_config.train){
			unaryPot->trainSegmentationClassifier(m_config.segmentationProbsFilename);
			unaryPot->trainPairwiseClassifier(m_config.pairWiseProbsFilename);

		}else{
			unaryPot->loadSegmentationProbs(m_config.segmentationProbsFilename);
			unaryPot->loadPairwiseProbs(m_config.pairWiseProbsFilename);
		}
		unaryPot->m_groundTruthImage=fixedSegmentationImage;
		//		unaryPot->trainPairwiseLikelihood("dummy.bin");
		//	if (classified)
		//		ImageUtils<ImageType>::writeImage("classified.nii",classified);

		LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        for (int i=0;i<LabelMapperType::nLabels;++i){
            //cout<<i<<" "<<LabelMapperType::getLabel(i)<<" "<<LabelMapperType::getIndex(LabelMapperType::getLabel(i))<<endl;
        }
		typedef typename  ImageType::SpacingType SpacingType;

		if (m_config.nSegmentations>1) m_config.nLevels++;
		m_config.nLevels=m_config.maxDisplacement>0?m_config.nLevels:1;
		int iterationCount=0;

		double minFactor=9999999999;
		for (int l=0;l<m_config.nLevels;++l){

			int level=m_config.levels[l];
			double labelScalingFactor=1;

			//at 4th level, we switch to full image grid but allow no displacements
			if (l==m_config.nLevels-1 &&m_config.nSegmentations>1){
				//			LabelMapperType * labelmapper2=new LabelMapperType(nSegmentations,maxDisplacement>0?1:0);
				labelmapper->setDisplacementSamples(0);//=new LabelMapperType(m_config.nSegmentations,0);
				level=99999999999;
				labelScalingFactor=1;
				if (LabelMapperType::nDisplacementSamples ==0) m_config.iterationsPerLevel=1;
				//				std::cout<<"Segmentation only! "<<LabelMapperType::nLabels<<" labels."<<std::endl;
			}



			//			unaryPot->SetWeights(m_config.simWeight,m_config.rfWeight/(m_config.nLevels-l),m_config.segWeight/(m_config.nLevels-l));
			std::cout<<"init graph"<<std::endl;
			GraphModelType graph(targetImage,unaryPot,level,labelScalingFactor,m_config.pairwiseSegmentationWeight, m_config.pairwiseRegistrationWeight );
			double segmentationFactor=1.0;
			double totalArea=1.0,patchArea=1.0;

			for (int d=0;d<D;++d){
				totalArea*=targetImage->GetLargestPossibleRegion().GetSize()[d];
				patchArea*=graph.getGridSize()[d];
			}
			segmentationFactor=1;//exp(-(120/level-4));//sqrt(sqrt(sqrt(patchArea/totalArea)));

			if (segmentationFactor<minFactor){
				minFactor=segmentationFactor;
			}
			//			segmentationFactor=(segmentationFactor-minFactor)/(1-minFactor)*(1-0.5)+0.5;
			std::cout<<level<<" "<<segmentationFactor<<std::endl;
			unaryPot->SetWeights(m_config.simWeight,m_config.rfWeight,m_config.segWeight*segmentationFactor);
			//			unaryPot->SetWeights(m_config.simWeight,m_config.rfWeight,m_config.segWeight);
			//			graph.setRegistrationWeight(segmentationFactor*m_config.pairwiseRegistrationWeight );
			//			graph.setSegmentationWeight(segmentationFactor*m_config.pairwiseRegistrationWeight );
			graph.setGradientImage(fixedSegmentationImage);
			unaryPot->setRadius(graph.getSpacing()*1);
            //			unaryPot->setRadius(graph.getSpacing()*1);//0.5);
			std::cout<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
			std::cout<<"Current grid size :"<<graph.getGridSize()<<std::endl;
			std::cout<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
			for (int i=0;i<graph.nNodes();++i){
				//				std::cout<<i<<	" "<<graph.getGridPositionAtIndex(i)<<" "<<graph.gridToImageIndex(graph.getGridPositionAtIndex(i))<<" "<<graph.getImagePositionAtIndex(i)<<" "<<graph.getIntegerIndex(graph.getGridPositionAtIndex(i))<<std::endl;
			}
			for (int i=0;i<m_config.iterationsPerLevel;++i,++iterationCount){
				std::cout<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";//std::endl<<std::endl;
				graph.setDisplacementFactor(labelScalingFactor);

				unaryPot->SetDisplacementFactor(graph.getDisplacementFactor());
				unaryPot->SetBaseLabelMap(previousFullDeformation);
				graph.setLabelImage(previousFullDeformation);
				graph.calculateBackProjections();
				//	ok what now: create graph! solve graph! save result!Z
				LabelImagePointerType deformation;
				if (LabelMapperType::nDisplacementSamples >0){
					typedef   TRWS_MRFSolver<GraphModelType> MRFSolverType;
					//										typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
					MRFSolverType mrfSolver(&graph,1,1, false);
					mrfSolver.optimize();
					deformation=mrfSolver.getLabelImage();

				}else{
					//pixel level grid, only use simple binary MRF
					typedef GC_MRFSolver<GraphModelType> MRFSolverType;
					//					typedef   TRWS_SimpleMRFSolver<GraphModelType> MRFSolverType;
					//					typedef   NewSimpleFastPDMRFSolver<GraphModelType> MRFSolverType;
					MRFSolverType mrfSolver(&graph,1,1, false);
					mrfSolver.optimize();
					deformation=mrfSolver.getLabelImage();
				}
				std::cout<<"]"<<std::endl;

				//initialise interpolator
				//deformation

				fullDeformation=graph.getFullLabelImage(deformation);
				LabelIteratorType testIt(deformation,deformation->GetLargestPossibleRegion());
				for (testIt.GoToBegin();!testIt.IsAtEnd();++testIt){
                    //std::cout<<testIt.GetIndex()<<" "<<graph.gridToImageIndex(testIt.GetIndex())<<" "<<testIt.Get()<<" "<<fullDeformation->GetPixel(graph.gridToImageIndex(testIt.GetIndex()))<<std::endl;
				}
				//apply deformation to moving image
				IteratorType fixedIt(targetImage,targetImage->GetLargestPossibleRegion());
				ostringstream gridCosts,imageCosts,backProj;
				gridCosts<<"costsGrid-l"<<l<<"-i"<<i<<".png";
				imageCosts<<"costsImage-l"<<l<<"-i"<<i<<".png";
				backProj<<"backProjImg-l"<<l<<"-i"<<i<<".png";
				//				graph.saveBackProj(backProj.str());
				//				checkerGraph.setLabelImage(previousFullDeformation);
				//							graph.checkConstraints(deformation,gridCosts.str().c_str());
				//							checkerGraph.checkConstraints(fullDeformation,imageCosts.str());
#if 1
				typedef   itk::DisplacementFieldCompositionFilter<LabelImageType,LabelImageType> CompositionFilterType;
				typename CompositionFilterType::Pointer composer=CompositionFilterType::New();
				composer->SetInput(1,fullDeformation);
				composer->SetInput(0,previousFullDeformation);
				composer->Update();
				LabelImagePointerType composedDeformation=composer->GetOutput();
				LabelIteratorType labelIt(composedDeformation,composedDeformation->GetLargestPossibleRegion());
#else
				//
				LabelImagePointerType composedDeformation=LabelImageType::New();
				composedDeformation->SetRegions(previousFullDeformation->GetLargestPossibleRegion());
				composedDeformation->Allocate();
				LabelIteratorType labelIt(composedDeformation,composedDeformation->GetLargestPossibleRegion());
				LabelIteratorType label2It(previousFullDeformation,previousFullDeformation->GetLargestPossibleRegion());
				LabelIteratorType label3It(fullDeformation,fullDeformation->GetLargestPossibleRegion());
				for (label3It.GoToBegin(),label2It.GoToBegin(),labelIt.GoToBegin();!labelIt.IsAtEnd();++label2It,++labelIt,++label3It){
					LabelType label=label2It.Get()+label3It.Get();
					typename ImageInterpolatorType::ContinuousIndexType idx=labelIt.GetIndex();
					typename LabelImageType::SizeType size=composedDeformation->GetLargestPossibleRegion().GetSize();
					idx+=LabelMapperType::getDisplacement(label);
					labelIt.Set(label);
				}
#endif
				for (int ng=0;ng<graph.nNodes();++ng){
					IndexType gridIndex=graph.getGridPositionAtIndex(ng);
					IndexType imgIndex=graph.gridToImageIndex(gridIndex);
				}
				for (fixedIt.GoToBegin(),labelIt.GoToBegin();!fixedIt.IsAtEnd();++fixedIt,++labelIt){
					IndexType index=fixedIt.GetIndex();
					typename ImageInterpolatorType::ContinuousIndexType idx=unaryPot->getMovingIndex(index);
					LabelType displacement=labelIt.Get();
					//					std::cout<<displacement<<std::endl;
					idx+=LabelMapperType::getDisplacement(displacement);
					if (segmentationInterpolator->IsInsideBuffer(idx)){
						deformedImage->SetPixel(fixedIt.GetIndex(),movingInterpolator->EvaluateAtContinuousIndex(idx));
						deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),65535*(segmentationInterpolator->EvaluateAtContinuousIndex(idx)>0));

					}else{
						deformedImage->SetPixel(fixedIt.GetIndex(),0);
						deformedSegmentationImage->SetPixel(fixedIt.GetIndex(),0);
					}
					segmentationImage->SetPixel(fixedIt.GetIndex(),(LabelMapperType::getSegmentation(fullDeformation->GetPixel(index)))*65535);
				}
				previousFullDeformation=composedDeformation;
				labelScalingFactor*=0.3;
#if 1
				std::string suff;
				if (ImageType::ImageDimension==2){
					suff=".png";
				}
				if (ImageType::ImageDimension==3){
					suff=".nii";
				}
				ostringstream deformedFilename;
				deformedFilename<<m_config.outputDeformedFilename<<"-l"<<l<<"-i"<<i<<suff;
				ostringstream deformedSegmentationFilename;
				deformedSegmentationFilename<<m_config.outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<suff;
				ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedImage);
				ostringstream tmpSegmentationFilename;
				tmpSegmentationFilename<<m_config.segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<suff;
				ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(), segmentationImage);
				ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(), deformedSegmentationImage);
				//deformation
				if (m_config.defFilename!=""){
					ostringstream tmpDeformationFilename;
					tmpDeformationFilename<<m_config.defFilename<<"-l"<<l<<"-i"<<i<<".mha";
					//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
					ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
					//					ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),deformation);

					//
				}
#endif
			}
			std::cout<<std::endl<<std::endl;
		}
		ImageUtils<ImageType>::writeImage(m_config.outputDeformedFilename, deformedImage);
		ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename, segmentationImage);
		ImageUtils<ImageType>::writeImage(m_config.outputDeformedSegmentationFilename, deformedSegmentationImage);


		//deformation
		if (m_config.defFilename!=""){
			//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
			ImageUtils<LabelImageType>::writeImage(m_config.defFilename,previousFullDeformation);
			//
		}



		//	}

		return 1;
	}

};
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
