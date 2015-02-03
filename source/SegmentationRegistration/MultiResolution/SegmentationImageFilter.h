#include "Log.h"
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
#include "itkImageRegionConstIterator.h"
#include "ImageUtils.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include <itkNumericTraitsVectorPixel.h>
#include <fenv.h>
#include <sstream>
#include "itkHistogramMatchingImageFilter.h"
#include "SegmentationGraph.h"
#include "BaseLabel.h"
#include "MRF-TRWS-SimpleGraph.h"
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
#include "FilterUtils.hpp"
#include <itkImageAdaptor.h>
#include <itkAddPixelAccessor.h>
#include "itkDisplacementFieldCompositionFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkVectorImage.h"
#include "itkConstNeighborhoodIterator.h"
#include <itkVectorResampleImageFilter.h>
#include <itkResampleImageFilter.h>
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include <itkImageToImageFilter.h>
#include <itkBSplineDeformableTransform.h>
#include "MRF-FAST-PD.h"

namespace itk{
template<class TImage, 
         class TLabelMapper,
         class TUnarySegmentationPotential,
         class TPairwiseSegmentationPotential>
class SegmentationImageFilter: public itk::ImageToImageFilter<TImage,TImage>{
public:
    typedef SegmentationImageFilter Self;
    typedef ImageToImageFilter< TImage, TImage > Superclass;
    typedef SmartPointer< Self >        Pointer;
    
    /** Method for creation through the object factory. */
    itkNewMacro(Self);
    /** Run-time type information (and related methods). */
    itkTypeMacro(SegmentationImageToImageFilter, ImageToImageFilter);
    
	typedef TImage ImageType;
    static const int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
    typedef typename  ImageType::PixelType PixelType;
    typedef typename  ImageType::ConstPointer ConstImagePointerType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename  ImageType::SpacingType SpacingType;
    typedef typename  itk::ImageRegionIterator< ImageType>       IteratorType;
    typedef typename  itk::ImageRegionConstIterator< ImageType>       ConstIteratorType;

    typedef typename  itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
    typedef typename  ImageInterpolatorType::Pointer ImageInterpolatorPointerType;


	typedef TLabelMapper LabelMapperType;
    typedef typename TLabelMapper::LabelType LabelType;
    typedef typename LabelMapperType::LabelImageType LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointerType;
	typedef itk::ImageRegionIterator< LabelImageType>       LabelIteratorType;
    typedef VectorLinearInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
    typedef typename  LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
    typedef itk::VectorResampleImageFilter< LabelImageType , LabelImageType>	LabelResampleFilterType;
    
    typedef TPairwiseSegmentationPotential PairwiseSegmentationPotentialType;
    typedef typename  PairwiseSegmentationPotentialType::Pointer PairwiseSegmentationPotentialPointerType;

    typedef TUnarySegmentationPotential UnarySegmentationPotentialType;
    typedef typename  UnarySegmentationPotentialType::Pointer UnarySegmentationPotentialPointerType;
    typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
    typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
    
    //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
    typedef  SegmentationGraphModel<ImageType, 
                                    UnarySegmentationPotentialType,
                                    PairwiseSegmentationPotentialType,
                                    LabelMapperType> GraphModelType;
private:
	SRSConfig m_config;
    
public:
	SegmentationImageFilter(){
        this->SetNumberOfRequiredInputs(5);
    }
    
    void setConfig(SRSConfig c){
		m_config=c;
	}


   void setFixedImage(ImagePointerType img){
            SetNthInput(0,img);
        }
        void setMovingImage(ImagePointerType img){
            SetNthInput(1,img);
        }
        void setMovingSegmentation(ImagePointerType img){
            SetNthInput(2,img);
        }
        void setFixedGradientImage(ImagePointerType img){
            SetNthInput(3,img);
        }
        void setMovingGradientImage(ImagePointerType img){
            SetNthInput(4,img);
        }

	virtual void Update(){


		//define input images
		const ConstImagePointerType targetImage = this->GetInput(0);
        const ConstImagePointerType fixedGradientImage = this->GetInput(3);
        const ConstImagePointerType movingImage = this->GetInput(1);
        ConstImagePointerType movingGradient = this->GetInput(4);
        ConstImagePointerType movingSegmentationImage;
        if (D==2){
            //2d segmentations pngs [from matlab] may have screwed up intensities
            movingSegmentationImage = fixSegmentationImage(this->GetInput(2));
        }else{
            movingSegmentationImage = (this->GetInput(2));
        }
            
        //results
        ImagePointerType segmentationImage;
        
        //allocate memory
        //instantiate potentials

		UnarySegmentationPotentialPointerType unarySegmentationPot=UnarySegmentationPotentialType::New();
        PairwiseSegmentationPotentialPointerType m_pairwiseSegmentationPot=PairwiseSegmentationPotentialType::New();;

        //instantiate interpolators
        SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
		ImageInterpolatorPointerType movingInterpolator=ImageInterpolatorType::New();
        
        
		LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        
        ImagePointerType segmentation;
        
        //init graph
        LOG<<"init graph"<<std::endl;
        GraphModelType graph;
        graph.setFixedImage(targetImage);
        graph.initGraph(1);

        ImagePointerType movingGradientImage=ImageUtils<ImageType>::readImage(m_config.movingGradientFilename);
        movingGradientImage=FilterUtils<ImageType>::LinearResample((ConstImagePointerType)movingGradientImage,targetImage);
        unarySegmentationPot->SetFixedImage(targetImage);
        unarySegmentationPot->SetFixedGradientImage((ConstImagePointerType)fixedGradientImage);
        unarySegmentationPot->SetReferenceImage(movingImage);
        unarySegmentationPot->SetReferenceGradient((ConstImagePointerType)movingGradientImage);
        unarySegmentationPot->SetReferenceSegmentation(movingSegmentationImage);
        unarySegmentationPot->SetGradientScaling(m_config.pairwiseSegmentationWeight);
        unarySegmentationPot->Init();
        
        m_pairwiseSegmentationPot->SetFixedImage(targetImage);
        m_pairwiseSegmentationPot->SetFixedGradient((ConstImagePointerType)fixedGradientImage);
        m_pairwiseSegmentationPot->SetReferenceImage(movingImage);
        m_pairwiseSegmentationPot->SetReferenceGradient((ConstImagePointerType)movingGradientImage);
        m_pairwiseSegmentationPot->SetReferenceSegmentation(movingSegmentationImage);
        unarySegmentationPot->SetTissuePrior((ConstImagePointerType)ImageUtils<ImageType>::readImage(m_config.tissuePriorFilename));
        m_pairwiseSegmentationPot->Init();
        if (ImageType::ImageDimension==2){
            m_pairwiseSegmentationPot->evalImage(targetImage,(ConstImagePointerType)fixedGradientImage);
        }
        

        //register images and potentials
        graph.setUnarySegmentationFunction(unarySegmentationPot);
        graph.setPairwiseSegmentationFunction(m_pairwiseSegmentationPot);
        
        //	ok what now: create graph! solve graph! save result!Z
		
        //typedef TRWS_MRFSolver<GraphModelType> MRFSolverType;
        typedef GC_MRFSolver<GraphModelType> MRFSolverType;
        //typedef TRWS_SimpleMRFSolver<GraphModelType> MRFSolverType;
        //typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
                
        MRFSolverType mrfSolver(&graph,
                                m_config.rfWeight,
                                m_config.pairwiseSegmentationWeight, 
                                m_config.verbose);
        mrfSolver.optimize(m_config.optIter);
        LOG<<" ]"<<std::endl;
        segmentation=graph.getSegmentationImage(mrfSolver.getLabels());
        if (D==2){
            //segmentation=fixSegmentationImage((ConstImagePointerType)segmentation);
            typedef itk::Image<unsigned char,D> OutImageType;
            ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename, (makePngFromLabelImage(ConstImagePointerType(segmentation),m_config.nSegmentations)));

        }else{
            ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename, segmentation);
        }

        	
	}
    ConstImagePointerType makePngFromLabelImage(ConstImagePointerType segmentationImage, int nSegmentations){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            double multiplier=std::numeric_limits<PixelType>::max()/(nSegmentations-1);
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                //    LOG<<imageIt.Get()*multiplier<<" "<<multiplier<<endl;
                imageIt2.Set(imageIt.Get()*multiplier);
            }
            return (ConstImagePointerType)newImage;
        }
    ImagePointerType fixSegmentationImage(ConstImagePointerType segmentationImage){
        ImagePointerType newImage=ImageUtils<ImageType>::createEmpty((ConstImagePointerType)segmentationImage);
        typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
        ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
        hash_map<int, int> map;
        for (imageIt2.GoToBegin(),imageIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++imageIt2){
            imageIt2.Set(imageIt.Get()*255);//std::numeric_limits<PixelType>::max());
        }
        return newImage;
    }
}; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
