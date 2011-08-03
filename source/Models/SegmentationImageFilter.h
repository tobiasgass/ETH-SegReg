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

namespace itk{
template<class TImage, 
         class TLabelMapper,
         class TUnarySegmentationPotential>
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

    typedef TUnarySegmentationPotential UnarySegmentationPotentialType;
    typedef typename  UnarySegmentationPotentialType::Pointer UnarySegmentationPotentialPointerType;
    typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
    typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;

    //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
    typedef  SegmentationGraphModel<ImageType, 
                                    UnarySegmentationPotentialType,
                                    LabelMapperType> GraphModelType;
private:
	SRSConfig m_config;
    
public:
	SegmentationImageFilter(){
        this->SetNumberOfRequiredInputs(2);
    }
    
    void setConfig(SRSConfig c){
		m_config=c;
	}

    void setFixedImage(ImagePointerType img){
        SetNthInput(0,img);
    }
    void setFixedGradientImage(ImagePointerType img){
        SetNthInput(1,img);
    }
	virtual void Update(){


		//define input images
		const ConstImagePointerType targetImage = this->GetInput(0);
        const ConstImagePointerType fixedGradientImage = this->GetInput(1);
    
        //results
        ImagePointerType segmentationImage;
        
        //allocate memory
        //instantiate potentials

		UnarySegmentationPotentialPointerType unarySegmentationPot=UnarySegmentationPotentialType::New();

        //instantiate interpolators
        SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
		ImageInterpolatorPointerType movingInterpolator=ImageInterpolatorType::New();
        
        
		LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        
        ImagePointerType segmentation;
        
        //init graph
        std::cout<<"init graph"<<std::endl;
        GraphModelType graph;
        graph.setFixedImage(targetImage);
        graph.initGraph();
        //setup segmentation potentials
        unarySegmentationPot->SetFixedImage(targetImage);
        unarySegmentationPot->SetGradientImage(fixedGradientImage);
        //register images and potentials
        graph.setUnarySegmentationFunction(unarySegmentationPot);
        //	ok what now: create graph! solve graph! save result!Z
		
        typedef TRWS_MRFSolver<GraphModelType> MRFSolverType;
        //typedef GC_MRFSolver<GraphModelType> MRFSolverType;
        //        typedef TRWS_SimpleMRFSolver<GraphModelType> MRFSolverType;
                
        MRFSolverType mrfSolver(&graph,
                                m_config.rfWeight,
                                m_config.pairwiseSegmentationWeight, 
                                false);
        mrfSolver.optimize();
        std::cout<<" ]"<<std::endl;
        segmentation=graph.getSegmentationImage(mrfSolver.getLabels());
        if (D==2){
            segmentation=fixSegmentationImage(segmentation);
        }
		ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename, segmentation);

        	
	}
    ImagePointerType fixSegmentationImage(ImagePointerType segmentationImage){
        ImagePointerType newImage=ImageUtils<ImageType>::createEmpty((ConstImagePointerType)segmentationImage);
        typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
        ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
        hash_map<int, int> map;
        for (imageIt2.GoToBegin(),imageIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++imageIt2){
            imageIt2.Set(imageIt.Get()*std::numeric_limits<PixelType>::max());
        }
        return newImage;
    }
}; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
