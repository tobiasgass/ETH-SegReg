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
#include "Graph.h"
#include "BaseLabel.h"
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
         class TUnaryRegistrationPotential, 
         class TUnarySegmentationPotential,
         class TPairwiseRegistrationPotential,
         class TPairwiseSegmentationRegistrationPotential>
class HierarchicalSRSImageToImageFilter: public itk::ImageToImageFilter<TImage,TImage>{
public:
    typedef HierarchicalSRSImageToImageFilter Self;
    typedef ImageToImageFilter< TImage, TImage > Superclass;
    typedef SmartPointer< Self >        Pointer;
    
    /** Method for creation through the object factory. */
    itkNewMacro(Self);
    /** Run-time type information (and related methods). */
    itkTypeMacro(HierarchicalSRSImageToImageFilter, ImageToImageFilter);
    
	typedef TImage ImageType;
    static const int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
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

    
    typedef TUnaryRegistrationPotential UnaryRegistrationPotentialType;
    typedef TUnarySegmentationPotential UnarySegmentationPotentialType;
    typedef TPairwiseRegistrationPotential PairwiseRegistrationPotentialType;
    typedef TPairwiseSegmentationRegistrationPotential PairwiseSegmentationRegistrationPotentialType; 
    typedef typename  UnaryRegistrationPotentialType::Pointer UnaryRegistrationPotentialPointerType;
    typedef typename  UnarySegmentationPotentialType::Pointer UnarySegmentationPotentialPointerType;
    typedef typename  UnaryRegistrationPotentialType::RadiusType RadiusType;
    typedef typename  PairwiseRegistrationPotentialType::Pointer PairwiseRegistrationPotentialPointerType;
    typedef typename  PairwiseSegmentationRegistrationPotentialType::Pointer PairwiseSegmentationRegistrationPotentialPointerType;

    typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
    typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;

    //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
    typedef  GraphModel<ImageType, 
                                 UnaryRegistrationPotentialType,
                                 PairwiseRegistrationPotentialType,
                                 UnarySegmentationPotentialType,
                                 PairwiseSegmentationRegistrationPotentialType,
                                 LabelMapperType> GraphModelType;
    typedef  typename itk::DisplacementFieldCompositionFilter<LabelImageType,LabelImageType> CompositionFilterType;
private:
	SRSConfig m_config;
    
public:
	HierarchicalSRSImageToImageFilter(){
        this->SetNumberOfRequiredInputs(4);
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
	virtual void Update(){


		//define input images
		const ConstImagePointerType targetImage = this->GetInput(0);
		const ConstImagePointerType movingImage = this->GetInput(1);
		const ConstImagePointerType movingSegmentationImage = this->GetInput(2);
        const ConstImagePointerType fixedGradientImage = this->GetInput(3);
    
        //results
        ImagePointerType deformedImage,deformedSegmentationImage,segmentationImage;
		LabelImagePointerType fullDeformation,previousFullDeformation;
        
        //allocate memory
		previousFullDeformation=LabelImageType::New();
		previousFullDeformation->SetRegions(targetImage->GetLargestPossibleRegion());
		previousFullDeformation->SetOrigin(targetImage->GetOrigin());
		previousFullDeformation->SetSpacing(targetImage->GetSpacing());
		previousFullDeformation->SetDirection(targetImage->GetDirection());
		previousFullDeformation->Allocate();
		Vector<float, D> tmpVox(0.0);
		previousFullDeformation->FillBuffer(tmpVox);
        
        //instantiate potentials
		UnaryRegistrationPotentialPointerType unaryRegistrationPot=UnaryRegistrationPotentialType::New();
		UnarySegmentationPotentialPointerType unarySegmentationPot=UnarySegmentationPotentialType::New();
		PairwiseRegistrationPotentialPointerType pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
		PairwiseSegmentationRegistrationPotentialPointerType pairwiseSegmentationRegistrationPot=PairwiseSegmentationRegistrationPotentialType::New();

        //instantiate interpolators
        SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
		ImageInterpolatorPointerType movingInterpolator=ImageInterpolatorType::New();
        
        
		LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        
        for (int i=0;i<LabelMapperType::nLabels;++i){
            //cout<<i<<" "<<LabelMapperType::getLabel(i)<<" "<<LabelMapperType::getIndex(LabelMapperType::getLabel(i))<<endl;
        }

        int iterationCount=0;

		double minFactor=9999999999;
        int level;
        double scale=-1, oldscale=1.0;

        //start pyramid

        LabelImagePointerType deformation;
        ImagePointerType segmentation;
		for (int l=0;l<m_config.nLevels;++l){
            
            level=m_config.levels[l];
			double labelScalingFactor=1;
            double sigma=1;
            int offset=1;
            //roughly compute downscaling
            scale=0.5;//7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0];
            scale=scale<1.0?scale:1.0;
            std::cout<<"scale :"<<scale<<std::endl;
            //downsample images if wanted
            ConstImagePointerType downSampledTarget,downSampledReference,downSampledReferenceSegmentation,downSampledTargetSheetness;
            if (scale<1.0){
                downSampledTarget=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(targetImage,sigma),scale);
                downSampledReference=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(movingImage,sigma),scale);
                downSampledReferenceSegmentation=FilterUtils<ImageType>::NNResample((movingSegmentationImage),scale);
                //downSampledTargetSheetness=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(fixedGradientImage,sigma),scale);
                downSampledTargetSheetness=FilterUtils<ImageType>::NNResample((fixedGradientImage),scale);
            }
            else{
                downSampledTarget=targetImage;
                downSampledReference=movingImage;
                downSampledReferenceSegmentation=movingSegmentationImage;
                downSampledTargetSheetness=fixedGradientImage;
            }
            std::cout<<"Downsampled images to: "<<downSampledTarget->GetLargestPossibleRegion().GetSize()<<std::endl;

            //init graph
            std::cout<<"init graph"<<std::endl;
			GraphModelType graph;
            graph.setFixedImage(downSampledTarget);
            graph.setDisplacementFactor(labelScalingFactor);
            graph.initGraph(level);


            movingInterpolator->SetInputImage(downSampledReference);
            segmentationInterpolator->SetInputImage(downSampledReferenceSegmentation);

            //setup registration potentials
            unaryRegistrationPot->SetRadius(graph.getSpacing());
            unaryRegistrationPot->SetFixedImage(downSampledTarget);
            unaryRegistrationPot->SetMovingImage(downSampledReference);
            unaryRegistrationPot->SetScale(1);
            unaryRegistrationPot->Init();
            
            pairwiseRegistrationPot->SetFixedImage(downSampledTarget);

            //setup segmentation potentials
            unarySegmentationPot->SetFixedImage(downSampledTarget);
            unarySegmentationPot->SetGradientImage(downSampledTargetSheetness);

            //setup segreg potentials
            pairwiseSegmentationRegistrationPot->SetMovingSegmentationInterpolator(segmentationInterpolator);
            pairwiseSegmentationRegistrationPot->SetMovingInterpolator(movingInterpolator);
            pairwiseSegmentationRegistrationPot->SetFixedImage(downSampledTarget);
            

            //register images and potentials
			graph.setUnaryRegistrationFunction(unaryRegistrationPot);
            graph.setPairwiseRegistrationFunction(pairwiseRegistrationPot);
            graph.setUnarySegmentationFunction(unarySegmentationPot);
            graph.setPairwiseSegmentationRegistrationFunction(pairwiseSegmentationRegistrationPot);
          
            
            //resample the deformation from last iteration to the current image resolution.
            if (scale!=oldscale)
                previousFullDeformation=bSplineInterpolateLabelImage(previousFullDeformation, downSampledTarget);

            //now scale it according to spacing difference between this and the previous iteration
            SpacingType sp;
            sp.Fill(1.0);
            if ( oldscale!=-1) {
                sp=sp*1.0* scale/oldscale;//level/m_config.levels[l-1];
                previousFullDeformation=scaleLabelImage(previousFullDeformation,sp);
            }
            oldscale=scale;
			std::cout<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
			std::cout<<"Current grid size :"<<graph.getGridSize()<<std::endl;
			std::cout<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
		
            for (int i=0;i<m_config.iterationsPerLevel;++i,++iterationCount){
				std::cout<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";
                // displacementfactor decreases with iterations
				graph.setDisplacementFactor(labelScalingFactor);

                //register deformation from previous iteration
                unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                pairwiseSegmentationRegistrationPot->SetBaseLabelMap(previousFullDeformation);
		
				//	ok what now: create graph! solve graph! save result!Z
			
                typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                MRFSolverType mrfSolver(&graph,
                                        m_config.simWeight,
                                        m_config.pairwiseRegistrationWeight, 
                                        m_config.rfWeight,
                                        m_config.pairwiseSegmentationWeight, 
                                        m_config.segWeight,
                                        false);
                mrfSolver.optimize();
                std::cout<<" ]"<<std::endl;
                deformation=graph.getDeformationImage(mrfSolver.getDeformationLabels());
                segmentation=graph.getSegmentationImage(mrfSolver.getSegmentationLabels());

				//initialise interpolator
				//deformation

				fullDeformation=bSplineInterpolateLabelImage(deformation,downSampledTarget);
                //         fullDeformation=scaleLabelImage(fullDeformation,graph.getDisplacementFactor());
           
				//apply deformation to moving image
				ConstIteratorType fixedIt(downSampledTarget,downSampledTarget->GetLargestPossibleRegion());
#if 1
	
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
                
                deformedImage=deformImage(downSampledReference,composedDeformation);
                deformedSegmentationImage=deformSegmentationImage(downSampledReferenceSegmentation,composedDeformation);
                previousFullDeformation=composedDeformation;
				labelScalingFactor*=m_config.displacementRescalingFactor;
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
				ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentation);
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

        
        LabelImagePointerType finalDeformation=bSplineInterpolateLabelImage(previousFullDeformation, targetImage);
        ImagePointerType finalSegmentation=(segmentation);

        //if last level of pyramid was not 1:1, the final deformation has to be scaled and the segmentation needs to be resampled
        if (scale!=1.0){
            SpacingType sp;
            sp.Fill(1.0);
            finalDeformation=scaleLabelImage(finalDeformation,sp/scale);
            finalSegmentation=FilterUtils<ImageType>::NNResample(segmentation,1/scale);
        }


        ImagePointerType finalDeformedReference=deformImage(movingImage,finalDeformation);
        ImagePointerType finalDeformedReferenceSegmentation=deformSegmentationImage(movingSegmentationImage,finalDeformation);

      
      
		ImageUtils<ImageType>::writeImage(m_config.outputDeformedFilename, finalDeformedReference);
		ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename, finalSegmentation);
		ImageUtils<ImageType>::writeImage(m_config.outputDeformedSegmentationFilename, finalDeformedReferenceSegmentation);


		//deformation
		if (m_config.defFilename!=""){
			//		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
			ImageUtils<LabelImageType>::writeImage(m_config.defFilename,previousFullDeformation);
			//
		}



		//	}

	
	}
    LabelImagePointerType bSplineInterpolateLabelImage(LabelImagePointerType labelImg, ConstImagePointerType reference){
        typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
        const unsigned int SplineOrder = 3;
        typedef typename itk::Image<float,ImageType::ImageDimension> ParamImageType;
        typedef typename itk::ResampleImageFilter<ParamImageType,ParamImageType> ResamplerType;
        typedef typename itk::BSplineResampleImageFunction<ParamImageType,double> FunctionType;
        typedef typename itk::BSplineDecompositionImageFilter<ParamImageType,ParamImageType>			DecompositionType;
        typedef typename  itk::ImageRegionIterator<ParamImageType> Iterator;
        std::vector<typename ParamImageType::Pointer> newImages(ImageType::ImageDimension);
        //interpolate deformation
        for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
            {
                //			std::cout<<k<<" setup"<<std::endl;
                typename ParamImageType::Pointer paramsK=ParamImageType::New();
                paramsK->SetRegions(labelImg->GetLargestPossibleRegion());
                paramsK->SetOrigin(labelImg->GetOrigin());
                paramsK->SetSpacing(labelImg->GetSpacing());
                paramsK->SetDirection(labelImg->GetDirection());
                paramsK->Allocate();
                Iterator itCoarse( paramsK, paramsK->GetLargestPossibleRegion() );
                LabelIterator itOld(labelImg,labelImg->GetLargestPossibleRegion());
                for (itCoarse.GoToBegin(),itOld.GoToBegin();!itCoarse.IsAtEnd();++itOld,++itCoarse){
                    itCoarse.Set((itOld.Get()[k]));//*(k<ImageType::ImageDimension?getDisplacementFactor()[k]:1));
                    //				std::cout<<itCoarse.Get()<<std::endl;
                }
                //bspline interpolation for the displacements
                typename ResamplerType::Pointer upsampler = ResamplerType::New();
                typename FunctionType::Pointer function = FunctionType::New();
                upsampler->SetInput( paramsK );
                upsampler->SetInterpolator( function );
                upsampler->SetSize(reference->GetLargestPossibleRegion().GetSize() );
                upsampler->SetOutputSpacing( reference->GetSpacing() );
                upsampler->SetOutputOrigin( reference->GetOrigin());
                upsampler->SetOutputDirection( reference->GetDirection());
                typename DecompositionType::Pointer decomposition = DecompositionType::New();
                decomposition->SetSplineOrder( SplineOrder );
                decomposition->SetInput( upsampler->GetOutput() );
                decomposition->Update();
                newImages[k] = decomposition->GetOutput();
                
            }
    
        std::vector< Iterator*> iterators(ImageType::ImageDimension+1);
        for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
            {
                iterators[k]=new Iterator(newImages[k],newImages[k]->GetLargestPossibleRegion());
                iterators[k]->GoToBegin();
            }
        LabelImagePointerType fullLabelImage=LabelImageType::New();
        fullLabelImage->SetRegions(reference->GetLargestPossibleRegion());
        fullLabelImage->SetOrigin(reference->GetOrigin());
        fullLabelImage->SetSpacing(reference->GetSpacing());
        fullLabelImage->SetDirection(reference->GetDirection());
        fullLabelImage->Allocate();
        LabelIterator lIt(fullLabelImage,fullLabelImage->GetLargestPossibleRegion());
        lIt.GoToBegin();
        for (;!lIt.IsAtEnd();++lIt){
            LabelType l;
            for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ ){
                //				std::cout<<k<<" label: "<<iterators[k]->Get()<<std::endl;
                l[k]=iterators[k]->Get();
                ++(*(iterators[k]));
            }

            //			lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
            lIt.Set(l);
        }
        for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
            {
                delete iterators[k];
            }
        return fullLabelImage;
    }
    
    LabelImagePointerType scaleLabelImage(LabelImagePointerType labelImg, SpacingType scalingFactors){
        typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
        LabelIterator lIt(labelImg,labelImg->GetLargestPossibleRegion());
		lIt.GoToBegin();
		for (;!lIt.IsAtEnd();++lIt){
			lIt.Set(LabelMapperType::scaleDisplacement(lIt.Get(),scalingFactors));
        }
        return labelImg;
    }
 
    ImagePointerType deformImage(ConstImagePointerType image, LabelImagePointerType deformation){
        //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
        typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        LabelIterator deformationIt(deformation,deformation->GetLargestPossibleRegion());

        typedef typename itk::LinearInterpolateImageFunction<ImageType, double> ImageInterpolatorType;
        typename ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New();

        interpolator->SetInputImage(image);
        ImagePointerType deformed=ImageType::New();//ImageUtils<ImageType>::createEmpty(image);
      
        deformed->SetRegions(deformation->GetLargestPossibleRegion());
        deformed->SetOrigin(deformation->GetOrigin());
        deformed->SetSpacing(deformation->GetSpacing());
        deformed->SetDirection(deformation->GetDirection());
        deformed->Allocate();
        ImageIterator imageIt(deformed,deformed->GetLargestPossibleRegion());        
        for (imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename ImageInterpolatorType::ContinuousIndexType idx(index);
            LabelType displacement=deformationIt.Get();
            idx+=(displacement);
            if (interpolator->IsInsideBuffer(idx)){
                imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
                //deformed->SetPixel(imageIt.GetIndex(),interpolator->EvaluateAtContinuousIndex(idx));

            }else{
                imageIt.Set(0);
                //                deformed->SetPixel(imageIt.GetIndex(),0);
            }
        }
        return deformed;
    }
     ImagePointerType deformSegmentationImage(ConstImagePointerType segmentationImage, LabelImagePointerType deformation){
        //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
        typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        LabelIterator deformationIt(deformation,deformation->GetLargestPossibleRegion());
        
        typedef typename itk::NearestNeighborInterpolateImageFunction<ImageType, double> ImageInterpolatorType;
        typename ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New();

        interpolator->SetInputImage(segmentationImage);
        ImagePointerType deformed=ImageUtils<ImageType>::createEmpty(segmentationImage);
        deformed->SetRegions(deformation->GetLargestPossibleRegion());
        deformed->SetOrigin(deformation->GetOrigin());
        deformed->SetSpacing(deformation->GetSpacing());
        deformed->SetDirection(deformation->GetDirection());
        deformed->Allocate();
        ImageIterator imageIt(deformed,deformed->GetLargestPossibleRegion());        

        for (imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename ImageInterpolatorType::ContinuousIndexType idx(index);
            LabelType displacement=deformationIt.Get();
            idx+=(displacement);
            if (interpolator->IsInsideBuffer(idx)){
                imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx));

            }else{
                imageIt.Set(0);
            }
        }
        return deformed;
    }
    void setUpDistanceTransformImage(ImagePointerType segmentationImage){
        typedef typename itk::SignedMaurerDistanceMapImageFilter< ImageType, ImageType > DistanceTransformType;
        typename DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
        distanceTransform->SetInput(segmentationImage);
        distanceTransform->Update();
        //ImageUtils<ImageType>::writeImage("dt.nii",distanceTransform->GetOutput());
        //m_distanceToDeformedSegmentationMap=distanceTransform->GetOutput();
    

    }
}; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
