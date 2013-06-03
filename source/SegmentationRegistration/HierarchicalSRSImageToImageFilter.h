#include "Log.h"
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
#include "MRF-TRW-S.h"
#include "MRF-GCO.h"
#include "MRF-GC.h"
//#include "MRF-FAST-PD.h"
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
#include "Classifier.h"
#include "itkHausdorffDistanceImageFilter.h"
#include <float.h>

namespace itk{
    template<class TGraph>
    class HierarchicalSRSImageToImageFilter: public itk::ImageToImageFilter<typename TGraph::ImageType,typename TGraph::ImageType>{
    public:
        typedef  TGraph GraphModelType;
        typedef typename TGraph::ImageType ImageType;
        typedef HierarchicalSRSImageToImageFilter Self;
        typedef ImageToImageFilter<ImageType,ImageType > Superclass;
        typedef SmartPointer< Self >        Pointer;
    
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


        typedef typename GraphModelType::LabelMapperType LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename LabelMapperType::LabelImageType DeformationFieldType;
        typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
        typedef itk::ImageRegionIterator< DeformationFieldType>       LabelIteratorType;
        typedef VectorLinearInterpolateImageFunction<DeformationFieldType, double> LabelInterpolatorType;
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

        typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;

        //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
    
    private:
        SRSConfig * m_config;
        DeformationFieldPointerType m_finalDeformation,m_bulkTransform;
        bool m_useBulkTransform;
        ImagePointerType m_finalSegmentation;
        ConstImagePointerType m_targetImage;
        ConstImagePointerType m_atlasImage;
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
        }
        virtual double getEnergy(){return lastEnergy;}
        void setConfig(SRSConfig * c){
            m_config=c;
        }

        void setTargetImage(ImagePointerType img){
            SetNthInput(0,img);
        }
        void setAtlasImage(ImagePointerType img){
            SetNthInput(1,img);
        }
        void setAtlasSegmentation(ImagePointerType img){
            SetNthInput(2,img);
        }
        void setTargetGradient(ImagePointerType img){
            SetNthInput(3,img);
        }
        void setAtlasGradient(ImagePointerType img){
            SetNthInput(4,img);
        }
        void setTissuePrior(ImagePointerType img){
            SetNthInput(5,img);
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
        virtual void Init(){
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

            //instantiate potentials
            m_unaryRegistrationPot=UnaryRegistrationPotentialType::New();
            m_unarySegmentationPot=UnarySegmentationPotentialType::New();
            m_pairwiseSegmentationPot=PairwiseSegmentationPotentialType::New();
            m_pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
            m_pairwiseCoherencePot=PairwiseCoherencePotentialType::New();

            if (regist || coherence){
                m_unaryRegistrationPot->setThreshold(m_config->thresh_UnaryReg);
                m_unaryRegistrationPot->setLogPotential(m_config->log_UnaryReg);
                m_unaryRegistrationPot->setNoOutsidePolicy(m_config->penalizeOutside);

                m_pairwiseRegistrationPot->setThreshold(m_config->thresh_PairwiseReg);
                m_pairwiseRegistrationPot->setFullRegularization(m_config->fullRegPairwise);

                
            }
            if (segment){
                m_unarySegmentationPot->SetTargetImage(m_targetImage);
                m_unarySegmentationPot->SetTargetGradient((ConstImagePointerType)m_targetGradientImage);
                m_unarySegmentationPot->SetAtlasImage(m_atlasImage);
                m_unarySegmentationPot->SetAtlasGradient((ConstImagePointerType)m_atlasGradientImage);
                m_unarySegmentationPot->SetAtlasSegmentation(m_atlasSegmentationImage);
                m_unarySegmentationPot->SetGradientScaling(m_config->pairwiseSegmentationWeight);
                m_unarySegmentationPot->SetNSegmentationLabels(m_config->nSegmentations);
                if (m_config->useTissuePrior){
                    m_unarySegmentationPot->SetTissuePrior(this->GetInput(5));
                    m_unarySegmentationPot->SetUseTissuePrior(m_config->useTissuePrior);
                }
                m_unarySegmentationPot->Init();
                m_pairwiseSegmentationPot->SetTargetImage(m_targetImage);
                m_pairwiseSegmentationPot->SetTargetGradient((ConstImagePointerType)m_targetGradientImage);
                m_pairwiseSegmentationPot->SetAtlasImage(m_atlasImage);
                m_pairwiseSegmentationPot->SetAtlasGradient((ConstImagePointerType)m_atlasGradientImage);
                m_pairwiseSegmentationPot->SetAtlasSegmentation(m_atlasSegmentationImage);
                m_pairwiseSegmentationPot->SetNSegmentationLabels(m_config->nSegmentations);
                m_pairwiseSegmentationPot->Init();//m_config->pairWiseProbsFilename,m_config->train);
                m_pairwiseSegmentationPot->evalImage(m_targetImage,(ConstImagePointerType)m_targetGradientImage);
            }
            
        }
        virtual void Update(){
            bool coherence= (m_config->coherence);
            bool segment=m_config->segment;
            bool regist= m_config->regist;
            //results
            ImagePointerType deformedAtlasImage,deformedAtlasSegmentation,segmentationImage;
            DeformationFieldPointerType fullDeformation,previousFullDeformation;
            if (regist || coherence){
                if (m_useBulkTransform){
                    LOGV(1)<<"Initializing with bulk transform." <<endl;
                    previousFullDeformation=m_bulkTransform;
                }else{
                    //allocate memory
                    previousFullDeformation=DeformationFieldType::New();
                    previousFullDeformation->SetRegions(m_targetImage->GetLargestPossibleRegion());
                    previousFullDeformation->SetOrigin(m_targetImage->GetOrigin());
                    previousFullDeformation->SetSpacing(m_targetImage->GetSpacing());
                    previousFullDeformation->SetDirection(m_targetImage->GetDirection());
                    LOGV(1)<<"Initializing registration with identity transform." <<endl;
                    previousFullDeformation->Allocate();
                    Vector<float, D> tmpVox(0.0);
                    previousFullDeformation->FillBuffer(tmpVox);
                }
                deformedAtlasImage=TransfUtils<ImageType>::warpImage(m_atlasImage,previousFullDeformation);
                m_unaryRegistrationPot->resetNormalize();
            }

            ConstImagePointerType m_inputTargetImage=m_targetImage;
            
            LabelMapperType * labelmapper=new LabelMapperType(m_config->nSegmentations,m_config->maxDisplacement);
            LOGV(5)<<VAR(m_config->nSegmentations)<<" "<<VAR(LabelMapperType::nSegmentations)<<endl;
            int iterationCount=0; 
            int level;

            //start pyramid
            //asm volatile("" ::: "memory");
            DeformationFieldPointerType deformation;
            ImagePointerType segmentation=NULL;
            if (coherence){
                deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(m_atlasSegmentationImage,previousFullDeformation,true);
                m_pairwiseCoherencePot->SetNumberOfSegmentationLabels(m_config->nSegmentations);
                //m_pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)deformedAtlasSegmentation);
            }

            if (!coherence && !regist){
                m_config->nLevels=1;
                m_config->iterationsPerLevel=1;
            }
            bool computeLowResolutionBsplineIfPossible=false;
            LOGV(2)<<VAR(computeLowResolutionBsplineIfPossible)<<endl;
            typename GraphModelType::Pointer graph=GraphModelType::New();
            int l=0;
            if (LabelMapperType::nDisplacementSamples == 0 ) l=m_config->nLevels-1;
            bool pixelGrid = false;
            for (;l<m_config->nLevels  ;++l){
                logSetStage("Multiresolution level "+boost::lexical_cast<std::string>(l)+":0");
                //compute scaling factor for downsampling the images in the registration potential
                double mantisse=(1/m_config->scale);
                int exponent=max(0,m_config->nLevels-l-1);
               
                if (m_config->imageLevels>0){
                    exponent=max(0,m_config->imageLevels-l-1);
                }
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1.0/reductionFactor;
                //unaryRegistrationPot->SetScale(7.0*level/m_targetImage->GetLargestPossibleRegion().GetSize()[0]);
                LOGV(1)<<"Image downsampling factor for registration unary computation : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<endl;

                level=m_config->levels[l];
                double labelScalingFactor=m_config->displacementScaling;
                
                double segmentationScalingFactor=1.0;
                if (m_config->segmentationScalingFactor>0.0 &&  m_config->segmentationScalingFactor!=1.0){
                    //segmentationScalingFactor=1.0/pow( 1.0/m_config->segmentationScalingFactor,exponent+1);
                    segmentationScalingFactor=pow(m_config->segmentationScalingFactor,m_config->nLevels-l-1);
                    //downsampling target image by a factor of m_config->segmentationScalingFactor for each level.
                    LOGV(4)<<VAR(segmentationScalingFactor)<<endl;
                    m_targetImage=FilterUtils<ImageType>::LinearResample(m_inputTargetImage,segmentationScalingFactor,true);
                }else if (m_config->segmentationScalingFactor == 0.0){
                    LOG<<"Using same grid control point resolution for both registration and segmentation sub-graph!"<<endl;
                    logSetStage("segmentation grid size estimation");
                    //use same level of detail as used for the graph
                    //first set up dummy graph from original target image
                    graph->setConfig(*m_config);
                    graph->setTargetImage(m_inputTargetImage);
                    graph->setDisplacementFactor(labelScalingFactor);
                    graph->initGraph(level);
                    //use coarse graph image for resampling..
                    LOGV(4)<<VAR(graph->getCoarseGraphImage()->GetSpacing())<<endl;
                    //quite crude method ;)
                    segmentationScalingFactor = 1.0*graph->getCoarseGraphImage()->GetLargestPossibleRegion().GetSize()[0]/m_inputTargetImage->GetLargestPossibleRegion().GetSize()[0];
                    LOGV(4)<<VAR(segmentationScalingFactor)<<endl;

                    m_targetImage=FilterUtils<ImageType>::LinearResample(m_inputTargetImage,(ConstImagePointerType)graph->getCoarseGraphImage(),true);
                    LOGV(4)<<"downsampled image" << endl;
                    LOGV(4)<<VAR(graph->getCoarseGraphImage()->GetSpacing())<<endl;
                    logResetStage;
                } else{
                    m_targetImage = m_inputTargetImage;

                }
                
                if (false && l>0 && D==3){
                    LOG<<endl;
                    LOG<<"WARNING: REDUCING NUMBER OF DISPLACEMENTSAMPLES!!"<<endl;                    LOG<<endl;
                    labelmapper->setDisplacementSamples(max(1,LabelMapperType::nDisplacementSamples-1));
                }
                                                                     
                //init graph
                LOG<<"Initializing graph structure."<<std::endl;
                graph->setConfig(*m_config);
                graph->setTargetImage(m_targetImage);
                graph->setDisplacementFactor(labelScalingFactor);
                graph->initGraph(level);
                graph->SetTargetSegmentation(m_targetSegmentationImage);

                

                if (graph->getCoarseGraphImage()->GetLargestPossibleRegion().GetSize() == m_inputTargetImage->GetLargestPossibleRegion().GetSize()){
                    //do not continue after this iteration if the grid resolution is equal to the input resolution
                    pixelGrid=true;
                    LOG<<"Last iteration, since control grid resolution equals target image resolution" << endl;
                }

                if (regist||coherence){
                    //setup registration potentials
                    m_unaryRegistrationPot->SetScale(scaling);
                    m_unaryRegistrationPot->SetTargetImage(m_inputTargetImage);
                    m_unaryRegistrationPot->SetAtlasImage(m_atlasImage);
#if 0
                    LOG<<"WARNING: patch size 11x11 for unary registration potential " << endl;
                    m_unaryRegistrationPot->SetRadius(graph->getSpacing()*5);
#else
                    m_unaryRegistrationPot->SetRadius(graph->getSpacing());
#endif
#if 0
                    m_unaryRegistrationPot->SetAtlasSegmentation(m_atlasSegmentationImage);
                    m_unaryRegistrationPot->SetAlpha(m_config->alpha);
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

                //register images and potentials
                graph->setUnaryRegistrationFunction(m_unaryRegistrationPot);
                graph->setPairwiseRegistrationFunction(m_pairwiseRegistrationPot);
                graph->setUnarySegmentationFunction(m_unarySegmentationPot);
                graph->setPairwiseCoherenceFunction(m_pairwiseCoherencePot);
                graph->setPairwiseSegmentationFunction(m_pairwiseSegmentationPot);

                if (regist && ! pixelGrid){
                    if (computeLowResolutionBsplineIfPossible && !coherence){
                        //if we don't do SRS, the deformation needs only be resampled to the image resolution within the unary registration potential
                        previousFullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, (ConstImagePointerType)m_unaryRegistrationPot->GetTargetImage());
                    }else{
                        previousFullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, m_targetImage);
                    }
                }

                if (regist){
                    if (computeLowResolutionBsplineIfPossible && !coherence){
                        m_unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    }else{
                        m_unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation,scaling);
                    }
                }

                //now scale it according to spacing difference between this and the previous iteration
                SpacingType sp;
                sp.Fill(1.0);
                LOGV(1)<<"Current displacementFactor :"<<graph->getDisplacementFactor()<<std::endl;
                LOGV(1)<<"Current grid size :"<<graph->getGridSize()<<std::endl;
                LOGV(1)<<"Current grid spacing :"<<graph->getSpacing()<<std::endl;
                
                //m_pairwiseCoherencePot->SetThreshold(max(1.0,graph->getMaxDisplacementFactor()));//*(m_config->iterationsPerLevel-i)));
                double tolerance;
                //tolerance=max(1.0,0.5*(graph->getSpacing()[0]));
                tolerance=pow(m_config->toleranceBase,exponent+1);
                if (m_config->ARSTolerance>0.0){
                    tolerance=m_config->ARSTolerance;

                }                
               
                LOGV(4)<<VAR(tolerance)<<endl;
                m_pairwiseCoherencePot->SetTolerance(tolerance);
                //m_pairwiseCoherencePot->SetThreshold(max(1.0,(graph->getSpacing()[0])/2));//*(m_config->iterationsPerLevel-i)));

                bool converged=false;
                double oldEnergy=1,newEnergy=01,oldWorseEnergy=-1.0;
                int i=0;
                std::vector<int> defLabels,segLabels, oldDefLabels,oldSegLabels;
                if (LabelMapperType::nDisplacementSamples == 0 ) i=m_config->iterationsPerLevel-1;
                logResetStage;
                for (;!converged && i<m_config->iterationsPerLevel;++i,++iterationCount){
                    logSetStage("Multiresolution level "+boost::lexical_cast<std::string>(l)+":"+boost::lexical_cast<std::string>(i));

                    LOGV(7)<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<std::endl;
                    // displacementfactor decreases with iterations
                    LOGV(2)<<VAR(labelScalingFactor)<<endl;
                    graph->setDisplacementFactor(labelScalingFactor);


                    LOGV(2)<<VAR(graph->getMaxDisplacementFactor())<<endl;
                    //register deformation from previous iteration
                    if (regist){
                        if (computeLowResolutionBsplineIfPossible && !coherence){
                            m_unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                        }else{
                            m_unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation,scaling);
                        }
                        m_pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);

                        //when switching levels of multiresolution, compute normalization factor to equalize the effect of smaller patches in the reg unary.
                        if (! m_config->dontNormalizeRegUnaries) m_unaryRegistrationPot->setNormalize( i==0 && l>0);
                    }
                    if (coherence){
                        if ( l>0 && i==0 && m_config->segmentationScalingFactor<1.0){
                            //when segmentaiton is downsampled, the first iteration at each level>0 needs to be re-warped 
                            deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(m_atlasSegmentationImage,previousFullDeformation,true);
                        }
                        m_pairwiseCoherencePot->SetBaseLabelMap(previousFullDeformation);
                        TIME(m_pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)deformedAtlasSegmentation));
                    }
                    //  unaryRegistrationPot->SetAtlasImage(deformedAtlasImage);
                    //if (segment && (l>0 || i>0)) graph->SetTargetSegmentation((ConstImagePointerType)segmentation);

                    if (segment && coherence && m_config->segDistThresh!= -1){
                        graph->ReduceSegmentationNodesByCoherencePotential(m_config->segDistThresh);
                    }
                    //	ok what now: create graph! solve graph! save result!Z
                    //double linearIncreasingWeight=1.0/(m_config->nLevels-l);
                    //double expIncreasingWeight=exp(-(m_config->nLevels-l-1));
                    //double linearDecreasingWeight=1-linearIncreasingWeight;
                    //double expDecreasingWeight=exp(-l);
                    //#define TRUNC
                    LOGV(5)<<VAR(coherence)<<" "<<VAR(segment)<<" "<<VAR(regist)<<endl;
                    
                    if (false && segment && !coherence && !regist){
                        typedef  GC_MRFSolverSeg<GraphModelType> SolverType;
                        SolverType  *mrfSolverGC= new SolverType(graph, m_config->unarySegmentationWeight,
                                                                 m_config->pairwiseSegmentationWeight,m_config->verbose);
                        
                        mrfSolverGC->createGraph();   
                        mrfSolverGC->optimize(1);
                        segmentation=graph->getSegmentationImage(mrfSolverGC->getLabels());
                        delete mrfSolverGC;

                    }else{
                        //typedef TRWS_SRSMRFSolverTruncQuadrat2D<GraphModelType> MRFSolverType;
                        //typedef GCO_SRSMRFSolver<GraphModelType> MRFSolverType;
                        //typedef Incremental_TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                        //typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
                        
                        BaseMRFSolver<GraphModelType>  *mrfSolver;

                        if (m_config->TRW){
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                            mrfSolver = new MRFSolverType(graph,
                                                          m_config->unaryRegistrationWeight,///pow(sqrt(2.0),l),
                                                          m_config->pairwiseRegistrationWeight, 
                                                          m_config->unarySegmentationWeight,
                                                          m_config->pairwiseSegmentationWeight*segmentationScalingFactor,
                                                          m_config->pairwiseCoherenceWeight,//*pow( m_config->coherenceMultiplier,l),
                                                          m_config->verbose);
                        }else if (m_config->GCO){
                            typedef GCO_SRSMRFSolver<GraphModelType> MRFSolverType;
                            mrfSolver = new MRFSolverType(graph,
                                                          m_config->unaryRegistrationWeight,
                                                          m_config->pairwiseRegistrationWeight, 
                                                          m_config->unarySegmentationWeight,
                                                          m_config->pairwiseSegmentationWeight*(segmentationScalingFactor),
                                                          m_config->pairwiseCoherenceWeight,//*pow( m_config->coherenceMultiplier,l),
                                                          m_config->verbose);
                        }

                        //typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
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
                                LOGV(3)<<endl;
                            }
                        }
                        lastEnergy=newEnergy;
                        if (regist || coherence){
                            deformation=graph->getDeformationImage(defLabels);
                        }
                        if (segment || coherence)
                            segmentation=graph->getSegmentationImage(mrfSolver->getSegmentationLabels());

                        if (m_config->TRW){
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                            delete static_cast<MRFSolverType * >(mrfSolver);
                        }else if (m_config->GCO){
                            typedef GCO_SRSMRFSolver<GraphModelType> MRFSolverType;
                            delete static_cast<MRFSolverType * >(mrfSolver);
                        }

                    }
                    
                    //convergence check after second iteration
#if 1
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
                    LOGV(1)<<"Convergence ratio " <<100.0-100.0*fabs(newEnergy-oldEnergy)/fabs(oldEnergy+DBL_EPSILON)<<"%"<<endl;
#else
                    if (i>0){
                        //check convergence
                        //count unchanged registration labels
                        double count=0.0;
                        for (int s=0;s<defLabels.size();++s){
                            count+=(defLabels[s]==LabelMapperType::nDisplacements/2);
                        }
                        bool registrationConverged = !regist || count/defLabels.size() > 0.99;                    
                        //LOGV(1)<<VAR(registrationConverged)<<" " <<VAR(count/defLabels.size())<<endl;
                        
                        count=0.0; int targetStructureCount=0;
                        for (int s=0;s<segLabels.size();++s){
                            count+=(segLabels[s]== (m_config->nSegmentations-1))&&(segLabels[s]==oldSegLabels[s]);
                            targetStructureCount+=oldSegLabels[s]==m_config->nSegmentations-1;
                        }
                        bool segmentationConverged= !segment || count/targetStructureCount>0.99;
                        //LOGV(1)<<VAR(segmentationConverged)<<" "<<VAR(count/targetStructureCount)<<std::endl;
                        converged = regist * registrationConverged || segment * segmentationConverged;
                    }
                    oldSegLabels=segLabels;
                    oldDefLabels=defLabels;
#endif
                    oldEnergy=newEnergy;
                    //initialise interpolator
                    //deformation
                    DeformationFieldPointerType composedDeformation;

                    if (regist && ! pixelGrid){
                        if (computeLowResolutionBsplineIfPossible && !coherence){
                            //if we don't do SRS, the deformation needs only be resampled to the image resolution within the unary registration potential
                            fullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation, (ConstImagePointerType)m_unaryRegistrationPot->GetTargetImage());
                        }else{
                            TIME(fullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation, m_targetImage));
                        }
                    }else if (regist){
                        fullDeformation = deformation;
                    }
   
                    //apply deformation to atlas image
                    if (regist || coherence){
                        composedDeformation=TransfUtils<ImageType>::composeDeformations(fullDeformation,previousFullDeformation);
                        //deformedAtlasImage=TransfUtils<ImageType>::warpImage(m_atlasImage,composedDeformation);
                        
                        DeformationFieldPointerType lowResDef;
                        if (!pixelGrid)
                            lowResDef=TransfUtils<ImageType>::bSplineInterpolateDeformationField(composedDeformation,  (ConstImagePointerType)m_unaryRegistrationPot->GetTargetImage());
                        else
                            lowResDef = composedDeformation;
                        deformedAtlasImage=FilterUtils<ImageType>::NNResample(TransfUtils<ImageType>::warpImage(m_unaryRegistrationPot->GetAtlasImage(),lowResDef),m_targetImage,false);
                        deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(m_atlasSegmentationImage,composedDeformation,true);
                    }
                    
      
                    //m_pairwiseCoherencePot->SetThreshold(13);
                    //m_pairwiseCoherencePot->SetThreshold(max(10.0,10*graph->getMaxDisplacementFactor()));

                    //m_pairwiseCoherencePot->SetThreshold(1000000);
                    

                    previousFullDeformation=composedDeformation;
                    labelScalingFactor*=m_config->displacementRescalingFactor;
                    if (segmentation.IsNotNull()&& segmentationScalingFactor<1.0){
                        //segmentation = FilterUtils<ImageType>::BSplineResample(segmentation,m_inputTargetImage,false);
                        LOGV(6)<<VAR(segmentation->GetLargestPossibleRegion().GetSize())<<endl;
                        segmentation = FilterUtils<ImageType>::BSplineResampleSegmentation(segmentation,m_targetImage);
                    }
                 
                    if (m_config->verbose>6){
                        std::string suff;
                        if (ImageType::ImageDimension==2){
                            suff=".png";
                        }
                        if (ImageType::ImageDimension==3){
                            suff=".nii";
                        }
                        ostringstream deformedFilename;
                        deformedFilename<<m_config->outputDeformedFilename<<"-l"<<l<<"-i"<<i<<suff;
                        ostringstream deformedSegmentationFilename;
                        deformedSegmentationFilename<<m_config->outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<suff;
                        if (regist) ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedAtlasImage);
                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<m_config->segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<suff;
                     
                        if (ImageType::ImageDimension==2){
                            if (segment && segmentation.IsNotNull()) ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)segmentation,LabelMapperType::nSegmentations));
                            if (regist) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)deformedAtlasSegmentation,LabelMapperType::nSegmentations));
                        }
                        if (ImageType::ImageDimension==3){
                            if (segment  && segmentation.IsNotNull() ) ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentation);
                            if (regist) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedAtlasSegmentation);
                        }
                        //deformation
                        if (regist){
                            if (m_config->defFilename!=""){
                                ostringstream tmpDeformationFilename;
                                tmpDeformationFilename<<m_config->defFilename<<"-l"<<l<<"-i"<<i<<".mha";
                                //		ImageUtils<DeformationFieldType>::writeImage(defFilename,deformation);
                                ImageUtils<DeformationFieldType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
                                //					ImageUtils<DeformationFieldType>::writeImage(tmpDeformationFilename.str().c_str(),deformation);

                                //
                            }
                        }
                    }
                    
                    
                    logResetStage;
                }//iter
                logResetStage;
                if (pixelGrid){
                    m_config->displacementScaling*=0.5;
                }
            }//level

            if (regist || coherence){
                if (!pixelGrid)
                    m_finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, m_inputTargetImage);
                else{
                    m_finalDeformation = previousFullDeformation;
                }
            }
            m_finalSegmentation=(segmentation);
            delete labelmapper;
        }//run
      
        ImagePointerType makePngFromLabelImage(ImagePointerType segmentationImage, int nSegmentations){
            return makePngFromLabelImage(ConstImagePointerType(segmentationImage),nSegmentations);
        }
        ImagePointerType makePngFromLabelImage(ConstImagePointerType segmentationImage, int nSegmentations){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            //nSegmentations=FilterUtils<ImageType>::getMax(segmentationImage);
            double multiplier;
            if (nSegmentations<=1){
                multiplier=std::numeric_limits<PixelType>::max();
            }else{
                multiplier=std::numeric_limits<PixelType>::max()/(nSegmentations-1);
            }

            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                //LOG<<imageIt.Get()*multiplier<<" "<<multiplier<<endl;
                imageIt2.Set(imageIt.Get()*multiplier);
            }
            return newImage;
        }
        
        
        ImagePointerType fixSegmentationImage(ImagePointerType segmentationImage, int nSegmentations){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty((ConstImagePointerType)segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        

            nSegmentations=nSegmentations>0?nSegmentations:2;
            double divisor=FilterUtils<ImageType>::getMax(segmentationImage)/(nSegmentations-1);
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                imageIt2.Set(floor(1.0*imageIt.Get()/divisor+0.51));
            }

            return (ImagePointerType)newImage;
        }
        double computeLabelChange(std::vector<int> & ref, std::vector<int> & comp){
            int countDiff=0;
            if (ref.size()==0 || comp.size()!=ref.size()){
                bool notEq=(comp.size()!=ref.size());
                LOG<<VAR(ref.size())<<" "<<VAR(notEq)<<endl;
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
