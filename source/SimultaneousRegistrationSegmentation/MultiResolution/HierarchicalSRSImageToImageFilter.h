/*
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
#include <sstream>
#include "itkHistogramMatchingImageFilter.h"
#include "Graph.h"
#include "BaseLabel.h"
#include "BaseMRF.h"
#include "Log.h"

#ifdef WITH_TRWS
#include "MRF-TRW-S.h"
#endif
#ifdef WITH_GCO
#include "MRF-GCO.h"
#endif
#ifdef WITH_OPENGM
#include "MRF-opengm.h"
#endif
#ifdef WITH_GC
#include "MRF-GC.h"
#endif
#include <boost/lexical_cast.hpp>
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
#include "itkVectorImage.h"
#include "itkConstNeighborhoodIterator.h"
#include <itkVectorResampleImageFilter.h>
#include <itkResampleImageFilter.h>
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include <itkImageToImageFilter.h>
#include <itkBSplineDeformableTransform.h>
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineDeformableTransform.h"
#include <itkWarpImageFilter.h>
#include "itkImageRegionConstIteratorWithIndex.h"
#include "ChamferDistanceTransform.h"
#include "itkCastImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include <float.h>
#include "TransformationUtils.h"    
#include <algorithm>

namespace SRS{
    template<class TGraph>
    class HierarchicalSRSImageToImageFilter: public itk::ImageToImageFilter<typename TGraph::ImageType,typename TGraph::ImageType>{
    public:
        typedef  TGraph GraphModelType;
        typedef typename TGraph::ImageType ImageType;
        typedef HierarchicalSRSImageToImageFilter Self;
        typedef itk::ImageToImageFilter<ImageType,ImageType > Superclass;
        typedef itk::SmartPointer< Self >        Pointer;
    
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Run-time type information (and related methods). */
        itkTypeMacro(HierarchicalSRSImageToImageFilter, ImageToImageFilter);
    
      
        static const int D=ImageType::ImageDimension;
        typedef typename  ImageType::PixelType PixelType;
        typedef typename  ImageType::Pointer ImagePointerType;
        typedef typename  ImageType::ConstPointer ConstImagePointerType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename  ImageType::SpacingType SpacingType;
        typedef typename  itk::ImageRegionIterator< ImageType>       IteratorType;
        typedef typename  itk::ImageRegionConstIterator< ImageType>       ConstIteratorType;
        
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        
        typedef typename  itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename  ImageInterpolatorType::Pointer ImageInterpolatorPointerType;


        //typedef typename GraphModelType::LabelMapperType LabelMapperType;
        //typedef typename LabelMapperType::LabelType LabelType;
        //typedef typename LabelMapperType::LabelImageType DeformationFieldType;

        typedef typename TransfUtils<ImageType>::DisplacementType LabelType;
        typedef typename TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
        typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
        typedef SparseRegistrationLabelMapper<ImageType,LabelType> SparseLabelMapperType;
        typedef BaseLabelMapper<ImageType,LabelType> BaseLabelMapperType;

        typedef itk::ImageRegionIterator< DeformationFieldType>       LabelIteratorType;
        typedef itk::VectorLinearInterpolateImageFunction<DeformationFieldType, double> LabelInterpolatorType;
        typedef typename  LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
        typedef itk::VectorResampleImageFilter< DeformationFieldType , DeformationFieldType>	LabelResampleFilterType;
 
    
        typedef  typename GraphModelType::UnaryRegistrationFunctionType UnaryRegistrationPotentialType;
        typedef  typename GraphModelType::UnarySegmentationFunctionType UnarySegmentationPotentialType;
        typedef  typename  GraphModelType::PairwiseSegmentationFunctionType PairwiseSegmentationPotentialType;
        typedef typename  GraphModelType::PairwiseRegistrationFunctionType PairwiseRegistrationPotentialType;
        typedef typename  GraphModelType::PairwiseCoherenceFunctionType PairwiseCoherencePotentialType; 
        typedef typename  UnaryRegistrationPotentialType::Pointer UnaryRegistrationPotentialPointerType;
        typedef typename  UnarySegmentationPotentialType::Pointer UnarySegmentationPotentialPointerType;
        typedef typename  UnaryRegistrationPotentialType::RadiusType RadiusType;
        typedef typename  PairwiseSegmentationPotentialType::Pointer PairwiseSegmentationPotentialPointerType;
        typedef typename  PairwiseRegistrationPotentialType::Pointer PairwiseRegistrationPotentialPointerType;
        typedef typename  PairwiseCoherencePotentialType::Pointer PairwiseCoherencePotentialPointerType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;

        //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
    
    private:
        SRSConfig * m_config;
        DeformationFieldPointerType m_finalDeformation,m_bulkTransform;
        bool m_useBulkTransform;
        ImagePointerType m_finalSegmentation;
        ConstImagePointerType m_targetImage;
        ConstImagePointerType m_atlasImage;
        ConstImagePointerType m_atlasMaskImage;
        ConstImagePointerType m_atlasSegmentationImage;
        ConstImagePointerType m_targetGradientImage;
        ConstImagePointerType m_atlasGradientImage;
        ConstImagePointerType m_targetSegmentationImage;
        UnaryRegistrationPotentialPointerType m_unaryRegistrationPot;
        UnarySegmentationPotentialPointerType m_unarySegmentationPot;
        PairwiseSegmentationPotentialPointerType m_pairwiseSegmentationPot;
        PairwiseRegistrationPotentialPointerType m_pairwiseRegistrationPot;
        PairwiseCoherencePotentialPointerType m_pairwiseCoherencePot;
        double lastEnergy;
    public:
        HierarchicalSRSImageToImageFilter(){
            this->SetNumberOfRequiredInputs(5);
            m_useBulkTransform=false;
            m_targetSegmentationImage=NULL;
            //instantiate potentials
            m_unaryRegistrationPot=UnaryRegistrationPotentialType::New();
            m_unarySegmentationPot=UnarySegmentationPotentialType::New();
            m_pairwiseSegmentationPot=PairwiseSegmentationPotentialType::New();
            m_pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
            m_pairwiseCoherencePot=PairwiseCoherencePotentialType::New();
        }
        virtual double getEnergy(){return lastEnergy;}
        void setConfig(SRSConfig * c){
            m_config=c;
        }

        void setTargetImage(ImagePointerType img){
            this->SetNthInput(0,img);
        }
        void setAtlasImage(ImagePointerType img){
            this->SetNthInput(1,img);
        }
        void setAtlasMaskImage(ImagePointerType img){
            this->m_atlasMaskImage=img;
        }
        void setAtlasSegmentation(ImagePointerType img){
            this->SetNthInput(2,img);
        }
        void setTargetGradient(ImagePointerType img){
            this->SetNthInput(3,img);
        }
        void setAtlasGradient(ImagePointerType img){
            this->SetNthInput(4,img);
        }
        void setTargetAnatomyPrior(ImagePointerType img){
            this->SetNthInput(5,img);
        }
        void setBulkTransform(DeformationFieldPointerType transf){
            m_bulkTransform=transf;
            m_useBulkTransform=transf.IsNotNull();
        }
        void setTargetSegmentation(ImagePointerType seg){
            m_targetSegmentationImage=seg;
        }
        DeformationFieldPointerType affineRegistration(ConstImagePointerType m_targetImage, ConstImagePointerType m_atlasImage){
        }

        DeformationFieldPointerType getFinalDeformation(){
            return m_finalDeformation;
        }
        ImagePointerType getTargetSegmentationEstimate(){
            return m_finalSegmentation;
        }

        void setUnaryRegistrationPotentialFunction(UnaryRegistrationPotentialPointerType func){m_unaryRegistrationPot=func;}
        void setUnarySegmentationPotentialFunction(UnarySegmentationPotentialPointerType func){m_unarySegmentationPot=func;}
        void setPairwiseRegistrationPotentialFunction(PairwiseRegistrationPotentialPointerType func){m_pairwiseRegistrationPot=func;}
        void setPairwiseSegmentationPotentialFunction(PairwiseSegmentationPotentialPointerType func){m_pairwiseSegmentationPot=func;}
        void setPairwiseCoherencePotentialFunction(PairwiseCoherencePotentialPointerType func){m_pairwiseCoherencePot=func;}

        virtual void Init(){
            logSetStage("SRS initialisation");
            bool coherence= (m_config->coherence);
            bool segment=m_config->segment;
            bool regist= m_config->regist;
            if (segment){
                LOG<<"Switching on segmentation module"<<std::endl;
            }
            else{
                LOG<<"Switching off segmentation module"<<std::endl;
            }
            if (regist){
                LOG<<"Switching on registration module"<<std::endl;
            }
            else{
                LOG<<"Switching off registration module"<<std::endl;
            }
            if (coherence){
                LOG<<"Switching on coherence module"<<std::endl;
            }  
            else{
                LOG<<"Switching off coherence module"<<std::endl;
            }
           
            m_atlasImage = this->GetInput(1);
            m_atlasSegmentationImage = (this->GetInput(2));
            m_targetImage = this->GetInput(0);
            m_targetGradientImage = this->GetInput(3);
            m_atlasGradientImage=this->GetInput(4);

          

            if (regist || coherence){
                m_unaryRegistrationPot->setThreshold(m_config->thresh_UnaryReg);
                m_unaryRegistrationPot->setLogPotential(m_config->log_UnaryReg);
                m_unaryRegistrationPot->setNoOutsidePolicy(m_config->penalizeOutside);
                m_unaryRegistrationPot->SetAtlasLandmarksFile(m_config->atlasLandmarkFilename);
                m_unaryRegistrationPot->SetTargetLandmarksFile(m_config->targetLandmarkFilename);
                m_unaryRegistrationPot->setNormalizeImages(m_config->normalizeImages);
                //MOVED HERE, HOPE THIS DOES NOT BREAK ANYTHING
                m_unaryRegistrationPot->SetTargetImage(m_targetImage);
                m_unaryRegistrationPot->SetAtlasImage(m_atlasImage);
                // /MOVED
                m_pairwiseRegistrationPot->setThreshold(m_config->thresh_PairwiseReg);
                m_pairwiseRegistrationPot->setFullRegularization(m_config->fullRegPairwise);
                

                
            }
            if (segment){

              
                m_unarySegmentationPot->SetTargetImage(m_targetImage);
                m_unarySegmentationPot->SetTargetGradient((ConstImagePointerType)m_targetGradientImage);
                m_pairwiseSegmentationPot->SetTargetImage(m_targetImage);
                m_pairwiseSegmentationPot->SetTargetGradient((ConstImagePointerType)m_targetGradientImage);
                if (m_config->targetRGBImageFilename==""){
                  
                }else{
                    m_unarySegmentationPot->SetTargetImage(m_config->targetRGBImageFilename);
                    //m_pairwiseSegmentationPot->SetRGBTargetImage(m_config->targetRGBImageFilename);

                }
                m_unarySegmentationPot->SetAtlasImage(m_atlasImage);
                m_unarySegmentationPot->SetAtlasGradient((ConstImagePointerType)m_atlasGradientImage);
                m_pairwiseSegmentationPot->SetAtlasImage(m_atlasImage);
                m_pairwiseSegmentationPot->SetAtlasGradient((ConstImagePointerType)m_atlasGradientImage);
                if (m_config->atlasRGBImageFilename==""){
                   
                }else{
                    m_unarySegmentationPot->SetAtlasImage(m_config->atlasRGBImageFilename);
                    //m_pairwiseSegmentationPot->SetRGBAtlasImage(m_config->atlasRGBImageFilename);
                }

                m_unarySegmentationPot->SetAtlasSegmentation(m_atlasSegmentationImage);
                m_unarySegmentationPot->SetGradientScaling(m_config->pairwiseSegmentationWeight);
                m_unarySegmentationPot->SetNSegmentationLabels(m_config->nSegmentations);
                if (m_config->useTargetAnatomyPrior){
                    m_unarySegmentationPot->SetTargetAnatomyPrior(this->GetInput(5));
                    m_unarySegmentationPot->SetUseTargetAnatomyPrior(m_config->useTargetAnatomyPrior);
                }
                if (m_config->segmentationUnaryProbFilename!=""){
                    m_unarySegmentationPot->SetProbFile(m_config->segmentationUnaryProbFilename);
                }
                m_unarySegmentationPot->Init();
            
                m_pairwiseSegmentationPot->SetAtlasSegmentation(m_atlasSegmentationImage);
                m_pairwiseSegmentationPot->SetNSegmentationLabels(m_config->nSegmentations);
                m_pairwiseSegmentationPot->SetAlpha(m_config->alpha);
                m_pairwiseSegmentationPot->SetTheta(m_config->theta);

                m_pairwiseSegmentationPot->Init();//m_config->pairWiseProbsFilename,m_config->train);
                m_pairwiseSegmentationPot->evalImage(m_targetImage,(ConstImagePointerType)m_targetGradientImage);
            }
            logResetStage;
            
        }
        virtual void Update(){
            //THIS DOES ALL THE GOOD STUFF!


            //suffixes for temporary images
            std::string suff;
            if (ImageType::ImageDimension==2){
                suff=".nii";
            }
            if (ImageType::ImageDimension==3){
                suff=".nii";
            }
            bool bSpline=!m_config->linearDeformationInterpolation;
            bool coherence= (m_config->coherence);
            bool segment=m_config->segment;
            bool regist= m_config->regist;
            //results
            ImagePointerType deformedAtlasImage,deformedAtlasSegmentation,segmentationImage, deformedAtlasMaskImage;
            DeformationFieldPointerType fullDeformation,previousFullDeformation;
            if (regist || coherence){
                if (m_useBulkTransform){
                    LOGV(1)<<"Initializing with bulk transform." <<std::endl;
                    previousFullDeformation=m_bulkTransform;
                }else{
                    //allocate memory
                    previousFullDeformation=DeformationFieldType::New();
                    previousFullDeformation->SetRegions(m_targetImage->GetLargestPossibleRegion());
                    previousFullDeformation->SetOrigin(m_targetImage->GetOrigin());
                    previousFullDeformation->SetSpacing(m_targetImage->GetSpacing());
                    previousFullDeformation->SetDirection(m_targetImage->GetDirection());
                    LOGV(1)<<"Initializing registration with identity transform." <<std::endl;
                    previousFullDeformation->Allocate();
                    itk::Vector<float, D> tmpVox(0.0);
                    previousFullDeformation->FillBuffer(tmpVox);
                }
                deformedAtlasImage=TransfUtils<ImageType>::warpImage(m_atlasImage,previousFullDeformation);
                if (m_atlasMaskImage.IsNotNull()){
                    LOGV(6)<<"Deforming moving mask.."<<std::endl;
                    deformedAtlasMaskImage=TransfUtils<ImageType>::warpImage(m_atlasMaskImage,previousFullDeformation);
                }
                else
                    deformedAtlasMaskImage=NULL;
                m_unaryRegistrationPot->resetNormalize();
            }

            ConstImagePointerType m_inputTargetImage=m_targetImage;
            
            SparseLabelMapperType * labelmapper=new SparseLabelMapperType(m_config->nSegmentations,m_config->nRegSamples[0]);
            LOGV(5)<<VAR(m_config->nSegmentations)<<" "<<VAR(labelmapper->getNumberOfSegmentationLabels())<<std::endl;
            int iterationCount=0; 
            int level;

            //start pyramid
            //asm volatile("" ::: "memory");
            DeformationFieldPointerType deformation;
            ImagePointerType segmentation=NULL;
            if (regist || coherence){
                deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(m_atlasSegmentationImage,previousFullDeformation,true);
                if (coherence){
                    m_pairwiseCoherencePot->SetNumberOfSegmentationLabels(m_config->nSegmentations);
		    m_pairwiseCoherencePot->SetAuxLabel(m_config->auxiliaryLabel);
		}
                //m_pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)deformedAtlasSegmentation);
            }

            if (!coherence && !regist){
                m_config->nLevels=1;
                m_config->iterationsPerLevel=1;
            }
            bool computeLowResolutionBsplineIfPossible=m_config->useLowResBSpline;
            LOGV(2)<<VAR(computeLowResolutionBsplineIfPossible)<<std::endl;
            typename GraphModelType::Pointer graph=GraphModelType::New();
            graph->setConfig(*m_config);
             //register images and potentials
            graph->setUnaryRegistrationFunction(m_unaryRegistrationPot);
            graph->setPairwiseRegistrationFunction(m_pairwiseRegistrationPot);
            graph->setUnarySegmentationFunction(m_unarySegmentationPot);
            graph->setPairwiseCoherenceFunction(m_pairwiseCoherencePot);
            graph->setPairwiseSegmentationFunction(m_pairwiseSegmentationPot);
            //graph->setLabelMapper(static_cast<BaseLabelMapperType *>(labelmapper));
            graph->setLabelMapper((labelmapper));
            int l=0;
            //check if any registration labels exist, if not, don't to multi-resolution stuff
            if (labelmapper->getNumberOfDisplacementSamplesPerAxis() == 0 ) l=m_config->nLevels-1;
            bool pixelGrid = false;

            double tolerance=1000;

            
            //START OF MULTI-RES Hierarchy
            //------------------------------------------------------------------------------------------------------------------------------------------------------------
            for (;l<m_config->nLevels  ;++l){
                
                logSetStage("Multiresolution level "+boost::lexical_cast<std::string>(l));
                //compute scaling factor for downsampling the images in the registration potential
                labelmapper->setNumberOfDisplacementSamplesPerAxis(m_config->nRegSamples[l]);
                double mantisse=(1/m_config->scale);
                int exponent=max(0,m_config->nLevels-l-1);
               
                if (m_config->imageLevels>0){
                    exponent=max(0,m_config->imageLevels-l-1);
                }
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1.0/reductionFactor;

                scaling=m_config->resamplingFactors[max(0,m_config->imageLevels-l-1)];

                LOGV(1)<<"Image downsampling factor for registration unary computation : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<std::endl;

                level=m_config->levels[l];
                double labelScalingFactor=m_config->displacementScaling;
                
                double segmentationScalingFactor=1.0;

		//computing downscaling factor for segmentation estimation if requested
                if (m_config->segmentationScalingFactor != 0.0 && m_config->nSegmentationLevels>1){
                    segmentationScalingFactor=pow(m_config->segmentationScalingFactor,m_config->nSegmentationLevels-l-1);
                    //segmentationScalingFactor=max(m_config->segmentationScalingFactor,m_config->resamplingFactors[max(0,m_config->nSegmentationLevels-l-1)]);
                    LOGV(4)<<VAR(segmentationScalingFactor)<<std::endl;
                    m_targetImage=FilterUtils<ImageType>::LinearResample(m_inputTargetImage,segmentationScalingFactor,true);
                }else if (m_config->segmentationScalingFactor == 0.0){
                    LOG<<"Using same grid control point resolution for both registration and segmentation sub-graph!"<<std::endl;
                    logSetStage("segmentation grid size estimation");
                    //use same level of detail as used for the graph
                    //first set up dummy graph from original target image
                    graph->setConfig(*m_config);
                    graph->setTargetImage(m_inputTargetImage);
                    graph->setDisplacementFactor(labelScalingFactor);
                    graph->initGraph(level);
                    //use coarse graph image for resampling..
                    LOGV(4)<<VAR(graph->getCoarseGraphImage()->GetSpacing())<<std::endl;
                    //quite crude method ;)
                    segmentationScalingFactor = 1.0*graph->getCoarseGraphImage()->GetLargestPossibleRegion().GetSize()[0]/m_inputTargetImage->GetLargestPossibleRegion().GetSize()[0];
                    LOGV(4)<<VAR(segmentationScalingFactor)<<std::endl;

                    m_targetImage=FilterUtils<ImageType>::LinearResample(m_inputTargetImage,(ConstImagePointerType)graph->getCoarseGraphImage(),true);
                    LOGV(4)<<"downsampled image" << endl;
                    LOGV(4)<<VAR(graph->getCoarseGraphImage()->GetSpacing())<<std::endl;
                    logResetStage;
                } else{
                    m_targetImage = m_inputTargetImage;

                }
                
                                                                     
                //init graph
                LOG<<"Initializing graph structure."<<std::endl;
                graph->setTargetImage(m_targetImage);
                graph->setDisplacementFactor(labelScalingFactor);
                graph->initGraph(level);
                graph->SetTargetSegmentation(m_targetSegmentationImage);

                

                if (graph->getCoarseGraphImage()->GetLargestPossibleRegion().GetSize() == m_inputTargetImage->GetLargestPossibleRegion().GetSize()){
                    //do not continue after this iteration if the grid resolution is equal to the input resolution
                    pixelGrid=true;
                    LOG<<"Last iteration, since control grid resolution equals target image resolution" << endl;
                    LOG<<VAR(graph->getCoarseGraphImage()->GetLargestPossibleRegion().GetSize())<<" "<<VAR( m_inputTargetImage->GetLargestPossibleRegion().GetSize())<<std::endl;
                }

                if (regist||coherence){
                    //setup registration potentials
                    m_unaryRegistrationPot->SetScale(scaling);
                  
                    m_unaryRegistrationPot->SetAtlasMaskImage(m_atlasMaskImage);
                    if (m_config->alpha > 0 && segment){
                        LOG<<"WARNING, alpha used for both seg and reg potentials, this will likely break stuff"<<std::endl;
                    }
                    m_unaryRegistrationPot->SetAlpha(m_config->alpha);
#if 0
                    LOG<<"WARNING: patch size 11x11 for unary registration potential " << endl;
                    m_unaryRegistrationPot->SetRadius(graph->getSpacing()*5);
#else
                    m_unaryRegistrationPot->SetRadius(graph->getSpacing());
#endif
#if 0
                    m_unaryRegistrationPot->SetAtlasSegmentation(m_atlasSegmentationImage);
                    m_unaryRegistrationPot->SetTargetGradient(m_targetImageGradient);
#endif          
                    
                    m_unaryRegistrationPot->Init();
                    m_pairwiseRegistrationPot->SetTargetImage(m_inputTargetImage);
                    m_pairwiseRegistrationPot->SetSpacing(graph->getSpacing());
                    
                }

                if (segment){
                    //setup segmentation potentials
                    m_unarySegmentationPot->SetGradientScaling(m_config->pairwiseSegmentationWeight);
                    m_unarySegmentationPot->ResamplePotentials(segmentationScalingFactor);
                    m_pairwiseSegmentationPot->ResamplePotentials(segmentationScalingFactor);
                }
                if (coherence){
                    //setup segreg potentials
                    m_pairwiseCoherencePot->SetAtlasImage(m_atlasImage);
                    m_pairwiseCoherencePot->SetTargetImage(m_targetImage);
                    m_pairwiseCoherencePot->SetAsymmetryWeight(m_config->asymmetry);
                }

               


                if (regist && ! pixelGrid){
                    previousFullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, (ConstImagePointerType)graph->getCoarseGraphImage(),false);
                }
                if (regist){
                    m_unaryRegistrationPot->SetBaseDisplacementMap(previousFullDeformation);
                    //
                }

                //now scale it according to spacing difference between this and the previous iteration
                SpacingType sp;
                sp.Fill(1.0);
                LOGV(1)<<"Current displacementFactor :"<<graph->getDisplacementFactor()<<std::endl;
                LOGV(1)<<"Current grid size :"<<graph->getGridSize()<<std::endl;
                LOGV(1)<<"Current grid spacing :"<<graph->getSpacing()<<std::endl;

    
                //tolerance=max(1.0,0.5*(graph->getSpacing()[0]));
                double oldtolerance=pow(m_config->toleranceBase,exponent+1);
                if (l==0){ //calculate acumulated maximum displacement 'capture range'
                    tolerance=0.0;
                    LOGV(5)<<"Calculating tolerance based on maximum possible displacement (approximatively)"<<std::endl;
                    double maxDispAtFirstLevel=graph->getMaxDisplacementFactor()*m_config->nRegSamples[0];
                    for (int l2=0;l2<m_config->nLevels;++l2){
                        double disp=maxDispAtFirstLevel;
                        if (m_config->nRegSamples[l2]>0){
                            for (int it2=0;it2<m_config->iterationsPerLevel;++it2){
                                tolerance+=disp;
                                LOGV(5)<<VAR(l2)<<" "<<VAR(it2)<<" "<<VAR(tolerance)<<" "<<VAR(disp)<<" "<<VAR(maxDispAtFirstLevel)<<std::endl;
                                disp*=m_config->displacementRescalingFactor;
                            }
                        }
                        maxDispAtFirstLevel/=2;
                    }
                }
                if (m_config->ARSTolerance>0.0){
                    tolerance=m_config->ARSTolerance;
                }                

                bool converged=false;
                double oldEnergy=1,newEnergy=01,oldWorseEnergy=-1.0;
                int i=0;
                std::vector<int> defLabels,segLabels, oldDefLabels,oldSegLabels;
                if (labelmapper->getNumberOfDisplacementSamplesPerAxis() == 0 ) i=m_config->iterationsPerLevel-1;
                double tol=tolerance<4.0?2.0:sqrt(tolerance);
                LOGV(4)<<"tolerance :"<<tol<<" "<<VAR(oldtolerance)<<std::endl;
                //tolerance gets set only at levels to avoid that the energy changes during inner iterations. if tolerance would change within the inner iterations, convergence criteria based on energy would not be well-defined any more
                m_pairwiseCoherencePot->SetTolerance(tol);

                //INNER ITERATIONS AT EACH LEVEL OF HIERARCHY
                //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                for (;!converged && i<m_config->iterationsPerLevel;++i,++iterationCount){
                    logSetStage(":iter"+boost::lexical_cast<std::string>(i));
                    logSetStage(":InitIter");
                    LOGV(7)<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<std::endl;
                    // displacementfactor decreases with iterations
                    LOGV(2)<<VAR(labelScalingFactor)<<std::endl;
                    graph->setDisplacementFactor(labelScalingFactor);


                    LOGV(2)<<VAR(graph->getMaxDisplacementFactor())<<std::endl;
                    //register deformation from previous iteration
                    if (regist){

                        m_unaryRegistrationPot->SetBaseDisplacementMap(previousFullDeformation);

                        m_pairwiseRegistrationPot->SetBaseDisplacementMap(previousFullDeformation);

                        //when switching levels of multiresolution, compute normalization factor to equalize the effect of smaller patches in the reg unary.
                        if (! m_config->dontNormalizeRegUnaries) m_unaryRegistrationPot->setNormalize( i==0 && l>0);
                    }
                    if (coherence || (regist && m_config->verbose>6)){
                        
                        DeformationFieldPointerType scaledDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation,m_targetImage,false);
                        deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(m_atlasSegmentationImage,scaledDeformation,true);
                        if (coherence){
                            TIME(m_pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)deformedAtlasSegmentation));
                        }
                        if (m_config->verbose>6){
                            ostringstream deformedSegmentationFilename;
                            deformedSegmentationFilename<<m_config->outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<suff;
                          
                                if (regist) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedAtlasSegmentation);
                            
                        }
                    }
                 

                    if (segment && coherence && m_config->segDistThresh!= -1){
                        graph->ReduceSegmentationNodesByCoherencePotential(m_config->segDistThresh);
                    }
                    //	ok what now: create graph! solve graph! save result!Z
                 
                    //#define TRUNC
                    LOGV(5)<<VAR(coherence)<<" "<<VAR(segment)<<" "<<VAR(regist)<<std::endl;
                    logUpdateStage(":Optimization");
                    if (m_config->solver=="GC" && m_config->nSegmentations == 2 && segment && !coherence && !regist){
#ifdef WITH_GC

                        typedef  GC_MRFSolverSeg<GraphModelType> SolverType;
                        SolverType  *mrfSolverGC= new SolverType(graph, m_config->unarySegmentationWeight,
                                                                 m_config->pairwiseSegmentationWeight,m_config->verbose);
                        
                        mrfSolverGC->createGraph();   
                        mrfSolverGC->optimize(1);
                        segmentation=graph->getSegmentationImage(mrfSolverGC->getLabels());
                        delete mrfSolverGC;
#else
                            LOG<<"OPTIMIZER NOT INCLUDED, ABORTING"<<std::endl;
#endif

                    }else{
                       
                        
                        BaseMRFSolver<GraphModelType>  *mrfSolver;

                        if (m_config->TRW){
#ifdef WITH_TRWS

                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                            mrfSolver = new MRFSolverType(graph,
                                                          m_config->unaryRegistrationWeight,///pow(sqrt(2.0),l),
                                                          m_config->pairwiseRegistrationWeight, 
                                                          m_config->unarySegmentationWeight,
                                                          m_config->pairwiseSegmentationWeight,//*segmentationScalingFactor,
                                                          m_config->pairwiseCoherenceWeight,//*pow( m_config->coherenceMultiplier,l),
                                                          m_config->verbose);
#else
                            LOG<<"OPTIMIZER NOT INCLUDED, ABORTING"<<std::endl;
#endif
                        }else if (m_config->GCO){
#ifdef WITH_GCO

                            typedef GCO_SRSMRFSolver<GraphModelType> MRFSolverType;
                            mrfSolver = new MRFSolverType(graph,
                                                          m_config->unaryRegistrationWeight,
                                                          m_config->pairwiseRegistrationWeight, 
                                                          m_config->unarySegmentationWeight,
                                                          m_config->pairwiseSegmentationWeight,//*(segmentationScalingFactor),
                                                          m_config->pairwiseCoherenceWeight,//*pow( m_config->coherenceMultiplier,l),
                                                          m_config->verbose);
#else
                            LOG<<"OPTIMIZER NOT INCLUDED, ABORTING"<<std::endl;
#endif
                        }else if (m_config->OPENGM){
#ifdef WITH_OPENGM

                            typedef OPENGM_SRSMRFSolver<GraphModelType> MRFSolverType;
                            mrfSolver = new MRFSolverType(graph,
                                                          m_config->unaryRegistrationWeight,
                                                          m_config->pairwiseRegistrationWeight, 
                                                          m_config->unarySegmentationWeight,
                                                          m_config->pairwiseSegmentationWeight,//*(segmentationScalingFactor),
                                                          m_config->pairwiseCoherenceWeight,//*pow( m_config->coherenceMultiplier,l),
                                                          m_config->verbose);
#else
                            LOG<<"OPTIMIZER NOT INCLUDED, ABORTING"<<std::endl;
                            exit(0);
#endif


                        }else{
                            
                            LOG<<"No valid optimizer was chosen, aborting"<<std::endl;
                            exit(0);
                        }

                        mrfSolver->setPotentialCaching(m_config->cachePotentials);
                        TIME(mrfSolver->createGraph());
                        if (!m_config->evalContinuously){
                            TIME(newEnergy=mrfSolver->optimize(m_config->optIter));
                            defLabels=mrfSolver->getDeformationLabels();
                            segLabels=mrfSolver->getSegmentationLabels();
                        }else{
                            double tmpOldEng=10; newEnergy=10000000;
                            bool optimization_converged=false;
                            for (int o=0;o<m_config->optIter && ! optimization_converged;++o){
                                tmpOldEng=newEnergy;
                                newEnergy=mrfSolver->optimizeOneStep(o, optimization_converged);
                                if (regist || coherence){
                                    oldDefLabels=defLabels;
                                    defLabels=mrfSolver->getDeformationLabels();
                                    if (o>0){
                                        LOGV(2)<<"Deformation labels changed :"<<computeLabelChange(oldDefLabels,defLabels)<<" ";
                                    }
                                }
                                if (segment || coherence){
                                    oldSegLabels=segLabels;
                                    segLabels=mrfSolver->getSegmentationLabels();
                                    if (o>0){
                                        LOGV(2)<<" Segmentation labels changed :"<<computeLabelChange(oldSegLabels,segLabels)<<" ";
                                    }
                                }
                                LOGV(3)<<std::endl;
                            }
                        }
                        lastEnergy=newEnergy;
                        if (regist || coherence){
                            deformation=graph->getDeformationImage(defLabels);
                        }
                        if (segment || coherence){
                            segmentation=graph->getSegmentationImage(mrfSolver->getSegmentationLabels());
                        }
                        
                        if (m_config->TRW){
#ifdef WITH_TRWS
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                            delete static_cast<MRFSolverType * >(mrfSolver);
#endif
                        }else if (m_config->GCO){
#ifdef WITH_GCO
                            typedef GCO_SRSMRFSolver<GraphModelType> MRFSolverType;
                            delete static_cast<MRFSolverType * >(mrfSolver);
#endif
                        }else if (m_config->OPENGM){
#ifdef WITH_OPENGM
                            typedef OPENGM_SRSMRFSolver<GraphModelType> MRFSolverType;
                            delete static_cast<MRFSolverType * >(mrfSolver);
#endif                      
                        }

                    }
                    
                    //convergence check after second iteration
                    //if energy difference is large, and greater than the threshold, skip this iteration and start over
                    if (false && i>0 && newEnergy>oldEnergy &&  fabs(oldEnergy-newEnergy)/fabs(oldEnergy+DBL_EPSILON) > 1e-3  ){
                        logResetStage;
                        if ( fabs(oldWorseEnergy-newEnergy)/fabs(oldWorseEnergy+DBL_EPSILON) <1e-4)
                            break;
                        oldWorseEnergy=newEnergy;
                        continue;
                    }
                    //else converge if energy difference is lower than the threshold
                    converged=(i>0) && (fabs(oldEnergy-newEnergy)/(oldEnergy+DBL_EPSILON) < 1e-4 ); 
                    LOGV(1)<<"Convergence ratio " <<100.0-100.0*fabs(newEnergy-oldEnergy)/fabs(oldEnergy+DBL_EPSILON)<<"%"<<std::endl;

                    oldEnergy=newEnergy;
                    //initialise interpolator
                    //deformation
                    DeformationFieldPointerType composedDeformation;
                    logUpdateStage(":Postprocessing");
                    
                    if (regist || coherence){

                        fullDeformation = deformation;
                        composedDeformation=TransfUtils<ImageType>::composeDeformations(fullDeformation,previousFullDeformation);
                        //composedDeformation=TransfUtils<ImageType>::composeDeformations(previousFullDeformation,fullDeformation);
                     
                    }

                    

                    previousFullDeformation=composedDeformation;
                    labelScalingFactor*=m_config->displacementRescalingFactor;
                    if (segmentation.IsNotNull()&& segmentationScalingFactor<1.0){
                        LOGV(6)<<VAR(segmentation->GetLargestPossibleRegion().GetSize())<<std::endl;
                        segmentation = FilterUtils<ImageType>::BSplineResampleSegmentation(segmentation,m_targetImage);
                    }
                 
                    if (m_config->verbose>6){
                        DeformationFieldPointerType lowResDef;
                        if (!pixelGrid){
                            if (bSpline)
                                lowResDef=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation,  (ConstImagePointerType)m_unaryRegistrationPot->GetTargetImage());
                            else
                                lowResDef=TransfUtils<ImageType>::linearInterpolateDeformationField(previousFullDeformation,  (ConstImagePointerType)m_unaryRegistrationPot->GetTargetImage());
                        }
                        else
                            lowResDef = composedDeformation;
                        deformedAtlasImage=TransfUtils<ImageType>::warpImage(m_unaryRegistrationPot->GetAtlasImage(),lowResDef);
                      
                        if (m_atlasMaskImage.IsNotNull()){
                            LOGV(6)<<"Deforming moving mask.."<<std::endl;
                            deformedAtlasMaskImage=TransfUtils<ImageType>::warpImage(m_atlasMaskImage,previousFullDeformation);
                        }
                       
                        ostringstream deformedFilename;
                        deformedFilename<<m_config->outputDeformedFilename<<"-l"<<l<<"-i"<<i<<suff;
                        ostringstream deformedSegmentationFilename;
                        deformedSegmentationFilename<<m_config->outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<suff;
                        if (regist) ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedAtlasImage);
                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<m_config->segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<suff;

                        if (m_atlasMaskImage.IsNotNull()){
                            ostringstream deformedMaskFilename;
                            deformedMaskFilename<<m_config->outputDeformedSegmentationFilename<<"-MASK-l"<<l<<"-i"<<i<<suff;
                            ImageUtils<ImageType>::writeImage(deformedMaskFilename.str().c_str(), deformedAtlasMaskImage);
                        }
                        if (segment || coherence){
                            segmentation=FilterUtils<ImageType>::fillHoles(segmentation);
                        }
                       
                            if (segment  && segmentation.IsNotNull() ) ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentation);
                            if (regist  && deformedAtlasSegmentation.IsNotNull()) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedAtlasSegmentation);
                        
                        //deformation
                        if (regist){
                            if (m_config->defFilename!=""){
                                ostringstream tmpDeformationFilename;
                                tmpDeformationFilename<<m_config->defFilename<<"-l"<<l<<"-i"<<i<<".mha";
                                ImageUtils<DeformationFieldType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);

                            }
                        }
                    }
                    //update tolerance by subtracting the maximal displacement of the current iteration
                    tolerance-=graph->getMaxDisplacementFactor()*m_config->nRegSamples[l];
                    logResetStage;//inner
                    logResetStage;//iter
                }//iter
                logResetStage;//levels
                if (pixelGrid){
                    m_config->displacementScaling*=0.5;
                }
            }//level
            if (segmentation.IsNotNull()){
                segmentation=FilterUtils<ImageType>::fillHoles(segmentation);
            }
            m_finalSegmentation=(segmentation);

	    m_finalDeformation=previousFullDeformation;

            delete labelmapper;
        }//run
      
       
        
      
        double computeLabelChange(std::vector<int> & ref, std::vector<int> & comp){
            int countDiff=0;
            if (ref.size()==0 || comp.size()!=ref.size()){
                bool notEq=(comp.size()!=ref.size());
                LOG<<VAR(ref.size())<<" "<<VAR(notEq)<<std::endl;
                exit(0);
            }
                
            for (unsigned int s=0;s<ref.size();++s){
                countDiff+=(ref[s]!=comp[s]);
            }
            return 1.0*countDiff/ref.size();
        }
    }; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
