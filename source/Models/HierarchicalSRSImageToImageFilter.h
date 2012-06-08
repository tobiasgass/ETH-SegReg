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
#include "MRF-FAST-PD.h"
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
#include <google/heap-profiler.h>
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
        SRSConfig m_config;
        DeformationFieldPointerType m_finalDeformation,m_bulkTransform;
        bool m_useBulkTransform;
        ImagePointerType m_finalSegmentation;
    public:
        HierarchicalSRSImageToImageFilter(){
            this->SetNumberOfRequiredInputs(5);
            m_useBulkTransform=false;
        }
    
        void setConfig(SRSConfig c){
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
            m_useBulkTransform=true;
        }
        DeformationFieldPointerType affineRegistration(ConstImagePointerType targetImage, ConstImagePointerType atlasImage){
        }

        DeformationFieldPointerType getFinalDeformation(){
            return m_finalDeformation;
        }
        ImagePointerType getTargetSegmentationEstimate(){
            return m_finalSegmentation;
        }
        virtual void Update(){
            
            bool coherence= (m_config.coherence);
            bool segment=m_config.segment;
            bool regist= m_config.regist;

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
          //define input images
            ConstImagePointerType targetImage = this->GetInput(0);
            ConstImagePointerType atlasImage = this->GetInput(1);
            ConstImagePointerType atlasSegmentationImage;
            
            atlasSegmentationImage = (this->GetInput(2));
            
            ConstImagePointerType targetGradientImage = this->GetInput(3);
            ConstImagePointerType atlasGradientImage=this->GetInput(4);
            
            //results
            ConstImagePointerType deformedAtlasImage,deformedAtlasSegmentation,segmentationImage;
            DeformationFieldPointerType fullDeformation,previousFullDeformation;
            if (regist || coherence){
                if (m_useBulkTransform){
                    LOGV(1)<<"Initializing with bulk transform." <<endl;
                    previousFullDeformation=m_bulkTransform;
                }else{
                    //allocate memory
                    previousFullDeformation=DeformationFieldType::New();
                    previousFullDeformation->SetRegions(targetImage->GetLargestPossibleRegion());
                    previousFullDeformation->SetOrigin(targetImage->GetOrigin());
                    previousFullDeformation->SetSpacing(targetImage->GetSpacing());
                    previousFullDeformation->SetDirection(targetImage->GetDirection());
                    LOGV(1)<<"Initializing registration with identity transform." <<endl;
                    previousFullDeformation->Allocate();
                    Vector<float, D> tmpVox(0.0);
                    previousFullDeformation->FillBuffer(tmpVox);
                }
                deformedAtlasImage=TransfUtils<ImageType>::warpImage(atlasImage,previousFullDeformation);
            }

            //instantiate potentials
            UnaryRegistrationPotentialPointerType unaryRegistrationPot=UnaryRegistrationPotentialType::New();
            UnarySegmentationPotentialPointerType unarySegmentationPot=UnarySegmentationPotentialType::New();
            PairwiseSegmentationPotentialPointerType pairwiseSegmentationPot=PairwiseSegmentationPotentialType::New();
            PairwiseRegistrationPotentialPointerType pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
            PairwiseCoherencePotentialPointerType pairwiseCoherencePot=PairwiseCoherencePotentialType::New();

            //instantiate interpolators
            ImageInterpolatorPointerType atlasInterpolator=ImageInterpolatorType::New();
            if (regist || coherence){
                unaryRegistrationPot->setThreshold(m_config.thresh_UnaryReg);
                unaryRegistrationPot->setLogPotential(m_config.log_UnaryReg);
                pairwiseRegistrationPot->setThreshold(m_config.thresh_PairwiseReg);

            }
            if (segment){
                unarySegmentationPot->SetTargetImage(targetImage);
                unarySegmentationPot->SetTargetGradient((ConstImagePointerType)targetGradientImage);
                unarySegmentationPot->SetAtlasImage(atlasImage);
                unarySegmentationPot->SetAtlasGradient((ConstImagePointerType)atlasGradientImage);
                unarySegmentationPot->SetAtlasSegmentation(atlasSegmentationImage);
                unarySegmentationPot->SetGradientScaling(m_config.pairwiseSegmentationWeight);
                if (m_config.useTissuePrior){
                    unarySegmentationPot->SetTissuePrior(this->GetInput(5));
                    unarySegmentationPot->SetUseTissuePrior(m_config.useTissuePrior);
                }
                unarySegmentationPot->Init();
                
                pairwiseSegmentationPot->SetTargetImage(targetImage);
                pairwiseSegmentationPot->SetTargetGradient((ConstImagePointerType)targetGradientImage);
                pairwiseSegmentationPot->SetAtlasImage(atlasImage);
                pairwiseSegmentationPot->SetAtlasGradient((ConstImagePointerType)atlasGradientImage);
                pairwiseSegmentationPot->SetAtlasSegmentation(atlasSegmentationImage);
                pairwiseSegmentationPot->SetNSegmentationLabels(m_config.nSegmentations);
                pairwiseSegmentationPot->Init();
                pairwiseSegmentationPot->evalImage(targetImage,(ConstImagePointerType)targetGradientImage);
                
            }
            LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        
         
            int iterationCount=0; 

            
            int level;

            //start pyramid
            //asm volatile("" ::: "memory");
            DeformationFieldPointerType deformation;
            ImagePointerType segmentation;
            if (coherence){
                deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(atlasSegmentationImage,previousFullDeformation,true);
                pairwiseCoherencePot->SetNumberOfSegmentationLabels(m_config.nSegmentations);
                pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)atlasSegmentationImage);//deformedAtlasSegmentation);
            }

            if (!coherence && !regist){
                m_config.nLevels=1;
                m_config.iterationsPerLevel=1;
            }
            bool computeLowResolutionBsplineIfPossible=true;
            LOGV(2)<<VAR(computeLowResolutionBsplineIfPossible)<<endl;
            typename GraphModelType::Pointer graph=GraphModelType::New();
            for (int l=0;l<m_config.nLevels;++l){
                logSetStage("Multiresolution level "+boost::lexical_cast<std::string>(l)+":0");
                //compute scaling factor for downsampling the images in the registration potential
                double mantisse=(1/m_config.scale);
                int exponent=max(0,m_config.nLevels-l-1);
               
                if (m_config.imageLevels>0){
                    exponent=max(0,m_config.imageLevels-l-1);
                }
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1/reductionFactor;
                //unaryRegistrationPot->SetScale(7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0]);
                LOGV(1)<<"Image downsampling factor for registration unary computation : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<endl;

                level=m_config.levels[l];
                double labelScalingFactor=m_config.displacementScaling;

                //init graph
                LOG<<"Initializing graph structure."<<std::endl;
                graph->setConfig(m_config);
                graph->setTargetImage(targetImage);
                graph->setDisplacementFactor(labelScalingFactor);
                graph->initGraph(level);

            
                atlasInterpolator->SetInputImage(atlasImage);
                

                if (regist||coherence){
                    //setup registration potentials
                    unaryRegistrationPot->SetScale(scaling);
                    unaryRegistrationPot->SetTargetImage(targetImage);
                    unaryRegistrationPot->SetAtlasImage(atlasImage);
                    unaryRegistrationPot->SetRadius(graph->getSpacing());
#if 0
                    unaryRegistrationPot->SetAtlasSegmentation(atlasSegmentationImage);
                    unaryRegistrationPot->SetAlpha(m_config.alpha);
                    unaryRegistrationPot->SetTargetGradient(targetImageGradient);
#endif          
                    unaryRegistrationPot->Init();
            
                    pairwiseRegistrationPot->SetTargetImage(targetImage);
                    pairwiseRegistrationPot->SetSpacing(graph->getSpacing());
                    
                }

                if (segment){
                    //setup segmentation potentials
                    unarySegmentationPot->SetGradientScaling(m_config.pairwiseSegmentationWeight);
                }
                if (coherence){
                    //setup segreg potentials
                    //pairwiseCoherencePot->SetAtlasInterpolator(atlasInterpolator);
                    pairwiseCoherencePot->SetAtlasImage(atlasImage);
                    pairwiseCoherencePot->SetTargetImage(targetImage);
                    pairwiseCoherencePot->SetAsymmetryWeight(m_config.asymmetry);
                }

                //register images and potentials
                graph->setUnaryRegistrationFunction(unaryRegistrationPot);
                graph->setPairwiseRegistrationFunction(pairwiseRegistrationPot);
                graph->setUnarySegmentationFunction(unarySegmentationPot);
                graph->setPairwiseCoherenceFunction(pairwiseCoherencePot);
                graph->setPairwiseSegmentationFunction(pairwiseSegmentationPot);

                if (regist){
                    if (computeLowResolutionBsplineIfPossible && !coherence){
                    //if we don't do SRS, the deformation needs only be resampled to the image resolution within the unary registration potential
                        previousFullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, (ConstImagePointerType)unaryRegistrationPot->GetTargetImage());
                    }else{
                        previousFullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, targetImage);
                    }
                }

                if (regist){
                    if (computeLowResolutionBsplineIfPossible && !coherence){
                        unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    }else{
                        unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation,scaling);
                    }
                }

                //now scale it according to spacing difference between this and the previous iteration
                SpacingType sp;
                sp.Fill(1.0);
                LOGV(1)<<"Current displacementFactor :"<<graph->getDisplacementFactor()<<std::endl;
                LOGV(1)<<"Current grid size :"<<graph->getGridSize()<<std::endl;
                LOGV(1)<<"Current grid spacing :"<<graph->getSpacing()<<std::endl;
                
                //typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                bool converged=false;
                double oldEnergy,newEnergy=DBL_EPSILON;
                for (int i=0;!converged && i<m_config.iterationsPerLevel;++i,++iterationCount){
                    logSetStage("Multiresolution level "+boost::lexical_cast<std::string>(l)+":"+boost::lexical_cast<std::string>(i));
                    oldEnergy=newEnergy;
                    LOGV(7)<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<std::endl;
                    // displacementfactor decreases with iterations
                    graph->setDisplacementFactor(labelScalingFactor);
                    
                    //register deformation from previous iteration
                    if (regist){
                        if (computeLowResolutionBsplineIfPossible && !coherence){
                            unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                        }else{
                            unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation,scaling);
                        }
                        pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    }
                    if (coherence){
                            pairwiseCoherencePot->SetBaseLabelMap(previousFullDeformation);
                            //if ( l || i ) pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)deformedAtlasSegmentation);
                    }
                    //  unaryRegistrationPot->SetAtlasImage(deformedAtlasImage);
                    

                    //	ok what now: create graph! solve graph! save result!Z
                    //double linearIncreasingWeight=1.0/(m_config.nLevels-l);
                    //double expIncreasingWeight=exp(-(m_config.nLevels-l-1));
                    //double linearDecreasingWeight=1-linearIncreasingWeight;
                    //double expDecreasingWeight=exp(-l);
                    //#define TRUNC
                    {

                        if (ImageType::ImageDimension==2){
#ifdef TRUNC
                            typedef TRWS_SRSMRFSolverTruncQuadrat2D<GraphModelType> MRFSolverType;
#else
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                            //typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
#endif

                            MRFSolverType  *mrfSolver= new MRFSolverType(graph,
                                                                         m_config.unaryRegistrationWeight,
                                                                         m_config.pairwiseRegistrationWeight,// * (l>0 || i>0 ) ,
                                                                         m_config.unarySegmentationWeight,
                                                                         m_config.pairwiseSegmentationWeight,
                                                                         m_config.pairwiseCoherenceWeight,
                                                                         m_config.verbose);
                            mrfSolver->createGraph();
                            newEnergy=mrfSolver->optimize(m_config.optIter);
                            if (regist || coherence){
                                deformation=graph->getDeformationImage(mrfSolver->getDeformationLabels());
                            }
                            segmentation=graph->getSegmentationImage(mrfSolver->getSegmentationLabels());
                              
                            delete mrfSolver;

                        }else{
#ifdef TRUNC
                            typedef TRWS_SRSMRFSolverTruncQuadrat3D<GraphModelType> MRFSolverType;
#else
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
#endif
                            MRFSolverType  *mrfSolver= new MRFSolverType(graph,
                                                                         m_config.unaryRegistrationWeight,
                                                                         m_config.pairwiseRegistrationWeight, 
                                                                         m_config.unarySegmentationWeight,
                                                                         m_config.pairwiseSegmentationWeight,
                                                                         m_config.pairwiseCoherenceWeight,
                                                                         m_config.verbose);
                            mrfSolver->createGraph();
                            newEnergy=mrfSolver->optimize(m_config.optIter);
                            if (regist || coherence){
                                deformation=graph->getDeformationImage(mrfSolver->getDeformationLabels());
                            }
                            segmentation=graph->getSegmentationImage(mrfSolver->getSegmentationLabels());
                            delete mrfSolver;

                        }



                    }
                    converged=fabs(newEnergy-oldEnergy)/fabs(oldEnergy) <0.01 ; 
                    LOGV(3)<<"Convergence ratio " <<100.0*fabs(newEnergy-oldEnergy)/fabs(oldEnergy)<<"%"<<endl;
                    //initialise interpolator
                    //deformation
                    DeformationFieldPointerType composedDeformation;

                    if (regist){
                        if (computeLowResolutionBsplineIfPossible && !coherence){
                            //if we don't do SRS, the deformation needs only be resampled to the image resolution within the unary registration potential
                            fullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation, (ConstImagePointerType)unaryRegistrationPot->GetTargetImage());
                        }else{
                            fullDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation, targetImage);
                        }
                    }   //fullDeformation=scaleDeformationField(fullDeformation,graph->getDisplacementFactor());
   
                    //apply deformation to atlas image
                    if (regist || coherence){
                        composedDeformation=TransfUtils<ImageType>::composeDeformations(fullDeformation,previousFullDeformation);
                        deformedAtlasImage=TransfUtils<ImageType>::warpImage(atlasImage,composedDeformation);
                        deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(atlasSegmentationImage,composedDeformation,true);
                    }
                    
      
                    //pairwiseCoherencePot->SetThreshold(13);
                    //pairwiseCoherencePot->SetThreshold(max(10.0,10*graph->getMaxDisplacementFactor()));
                    pairwiseCoherencePot->SetThreshold(max(1.0,graph->getMaxDisplacementFactor()));//*(m_config.iterationsPerLevel-i)));
                    //pairwiseCoherencePot->SetThreshold(1000000);
                    

                    previousFullDeformation=composedDeformation;
                    labelScalingFactor*=m_config.displacementRescalingFactor;
                    if (m_config.verbose){
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
                        if (regist) ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedAtlasImage);
                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<m_config.segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<suff;
                        if (ImageType::ImageDimension==2){
                            if (segment) ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),makePngFromDeformationField((ConstImagePointerType)segmentation,LabelMapperType::nSegmentations));
                            if (regist) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromDeformationField((ConstImagePointerType)deformedAtlasSegmentation,LabelMapperType::nSegmentations));
                        }
                        if (ImageType::ImageDimension==3){
                            if (segment) ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentation);
                            if (regist) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedAtlasSegmentation);
                        }
                        //deformation
                        if (regist){
                            if (m_config.defFilename!=""){
                                ostringstream tmpDeformationFilename;
                                tmpDeformationFilename<<m_config.defFilename<<"-l"<<l<<"-i"<<i<<".mha";
                                //		ImageUtils<DeformationFieldType>::writeImage(defFilename,deformation);
                                ImageUtils<DeformationFieldType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
                                //					ImageUtils<DeformationFieldType>::writeImage(tmpDeformationFilename.str().c_str(),deformation);

                                //
                            }
                        }
                    }
                    
                    

                }
            }

            if (regist || coherence)
                m_finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(previousFullDeformation, targetImage);
            m_finalSegmentation=(segmentation);
            delete labelmapper;
	
        }
      
        ConstImagePointerType makePngFromDeformationField(ConstImagePointerType segmentationImage, int nSegmentations){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            double multiplier=std::numeric_limits<PixelType>::max()/(nSegmentations-1);
            if (!nSegmentations){
                multiplier=std::numeric_limits<PixelType>::max();
            }
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                //LOG<<imageIt.Get()*multiplier<<" "<<multiplier<<endl;
                imageIt2.Set(imageIt.Get()*multiplier);
            }
            return (ConstImagePointerType)newImage;
        }
        
        
        ImagePointerType fixSegmentationImage(ImagePointerType segmentationImage, int nSegmentations){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty((ConstImagePointerType)segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        

            nSegmentations=nSegmentations>0?nSegmentations:2;
            double divisor=std::numeric_limits<PixelType>::max()/(nSegmentations-1);
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                imageIt2.Set(floor(1.0*imageIt.Get()/divisor+0.51));
            }

            return (ImagePointerType)newImage;
        }
    }; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
