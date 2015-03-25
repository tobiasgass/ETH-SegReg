/**
 * @file   HierarchicalJRSImageToImageFilter.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 13:16:00 2015
 * 
 * @brief  Alternating/Joint Registration and Segmentation Method (ARS/JRS, same thing)
 * 
 * 
 */

#include "Log.h"


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
#include "itkImageRegionConstIteratorWithIndex.h"
#include <google/heap-profiler.h>
#include "ChamferDistanceTransform.h"
#include "itkCastImageFilter.h"
#include "Potential-SegmentationRegistration-Pairwise.h"


namespace SRS{
    template<class TImage, 
             class TLabelMapper,
             class TUnaryRegistrationPotential, 
             class TUnarySegmentationPotential,
             class TPairwiseSegmentationPotential,
             class TPairwiseRegistrationPotential          >
    class HierarchicalJRSImageToImageFilter: public itk::ImageToImageFilter<TImage,TImage>{
    public:
        typedef HierarchicalJRSImageToImageFilter Self;
        typedef ImageToImageFilter< TImage, TImage > Superclass;
        typedef itk::SmartPointer< Self >        Pointer;
    
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Run-time type information (and related methods). */
        itkTypeMacro(HierarchicalJRSImageToImageFilter, ImageToImageFilter);
    
        typedef TImage ImageType;
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
        typedef TPairwiseSegmentationPotential PairwiseSegmentationPotentialType;
        typedef TPairwiseRegistrationPotential PairwiseRegistrationPotentialType;
        typedef PairwisePotentialSegmentationRegistration<ImageType> PairwisePotentialSegmentationRegistrationType;
        typedef typename  PairwiseSegmentationPotentialType::Pointer PairwiseSegmentationPotentialPointerType;
        typedef typename  UnaryRegistrationPotentialType::Pointer UnaryRegistrationPotentialPointerType;
        typedef typename  UnarySegmentationPotentialType::Pointer UnarySegmentationPotentialPointerType;
        typedef typename  UnaryRegistrationPotentialType::RadiusType RadiusType;
        typedef typename  PairwiseRegistrationPotentialType::Pointer PairwiseRegistrationPotentialPointerType;
        
        typedef typename UnarySegmentationPotentialType::SRSPotentialType SRSPotentialType;
        typedef typename SRSPotentialType::Pointer SRSPotentialPointerType;

        typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;

        //typedef typename PairwiseSegmentationPotentialType::ClassifierType PairwiseClassifierType;
        //typedef typename PairwiseClassifierType::Pointer PairwiseClassifierPointerType;

        //typedef typename UnarySegmentationPotentialType::ClassifierType UnaryClassifierType;
        //typedef typename UnaryClassifierType::Pointer UnaryClassifierPointerType;


        
        //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
        typedef  GraphModel<ImageType, 
                            UnaryRegistrationPotentialType,
                            PairwiseRegistrationPotentialType,
                            UnarySegmentationPotentialType,
                            PairwiseSegmentationPotentialType,
                            PairwisePotentialSegmentationRegistrationType,
                            LabelMapperType> GraphModelType;
        typedef  typename itk::DisplacementFieldCompositionFilter<LabelImageType,LabelImageType> CompositionFilterType;
    private:
        SRSConfig m_config;
        //PairwiseClassifierPointerType m_smoothnessClassifier;
        //UnaryClassifierPointerType m_segmentationClassifier;
        UnaryRegistrationPotentialPointerType m_unaryRegistrationPot;
        PairwiseRegistrationPotentialPointerType m_pairwiseRegistrationPot;
        UnarySegmentationPotentialPointerType m_unarySegmentationPot;
        PairwiseSegmentationPotentialPointerType m_pairwiseSegmentationPot;
        SRSPotentialPointerType m_SRSPotential;
        ConstImagePointerType m_movingSegmentationImage;
    public:
        HierarchicalJRSImageToImageFilter(){
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

        virtual void Init(){
            //get input images
            const ConstImagePointerType targetImage = this->GetInput(0);
            const ConstImagePointerType movingImage = this->GetInput(1);
            ConstImagePointerType movingGradient = this->GetInput(4);
            
            if (D==2){
                //2d segmentations pngs [from matlab] may have screwed up intensities
                m_movingSegmentationImage = fixSegmentationImage(this->GetInput(2));
            }else{
                m_movingSegmentationImage = (this->GetInput(2));
            }
            ImageUtils<ImageType>::writeImage("test.nii",(ConstImagePointerType)m_movingSegmentationImage);
      

            //Setup Segmentation potentials
            m_unarySegmentationPot=UnarySegmentationPotentialType::New();
            m_pairwiseSegmentationPot=PairwiseSegmentationPotentialType::New();
            
            m_unarySegmentationPot->SetFixedImage(targetImage);
            m_unarySegmentationPot->SetFixedGradientImage(this->GetInput(3));
            m_pairwiseSegmentationPot->SetFixedImage(targetImage);
            m_pairwiseSegmentationPot->SetFixedGradient(this->GetInput(3));
            

            ImagePointerType movingGradientImage=ImageUtils<ImageType>::readImage(m_config.movingGradientFilename);
            m_pairwiseSegmentationPot->Init();
            m_unarySegmentationPot->Init();
            //setup registration potentials

            m_unaryRegistrationPot=UnaryRegistrationPotentialType::New();
            m_pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
            m_SRSPotential=SRSPotentialType::New();
            m_SRSPotential->SetNumberOfSegmentationLabels(m_config.nSegmentations);
            m_SRSPotential->SetReferenceSegmentation((ConstImagePointerType)m_movingSegmentationImage);

            m_unaryRegistrationPot->SetSRSPotential(m_SRSPotential);
            m_unarySegmentationPot->SetSRSPotential(m_SRSPotential);
        }
        virtual void Update(){
            Init();
            ConstImagePointerType previousSegmentation=m_movingSegmentationImage, oldSegmentation;
            ConstImagePointerType previousDeformedReferenceSegmentation=m_movingSegmentationImage;
            bool converged=false;
            int iter=0;
            //save intermediate results
            std::string suff;
            if (ImageType::ImageDimension==2){
                suff=".nii";
            }
            if (ImageType::ImageDimension==3){
                suff=".nii";
            }
            while (!converged){

                //compute updates
                double alpha=m_config.segWeight;//+(1-exp(-1.0*iter/100));
                //   if (!iter) alpha=0;
#if 0
                previousSegmentation=segmentImage(previousDeformedReferenceSegmentation,iter==0?0:alpha);

                previousDeformedReferenceSegmentation=registerImagesAndDeformSegmentation(previousSegmentation,alpha);
#else
                if (m_config.tissuePriorFilename!=""){
                    //load deformed segmentation from file
                    previousDeformedReferenceSegmentation=ImageUtils<ImageType>::readImage(m_config.tissuePriorFilename);
                    LOG<<"WARNING: READ DEFORMED SEGMENTATION FROM "<<m_config.tissuePriorFilename<<" instead of computing the deformation"<<std::endl;
                }else{
                    //compute deformation
                    previousDeformedReferenceSegmentation=registerImagesAndDeformSegmentation(previousSegmentation,iter==0?0:alpha);
                    if (m_config.verbose){
                        ostringstream deformedSegmentationFilename;
                        deformedSegmentationFilename<<m_config.outputDeformedSegmentationFilename<<"-i"<<iter<<suff;
                    
                        if (D==3){
                            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),previousDeformedReferenceSegmentation);
                        
                        }else{
                            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)previousDeformedReferenceSegmentation ,m_config.nSegmentations));
                        
                        }
                    }
                    if (m_config.verbose)        LOG<<"saved deformed atlas segmentation"<<std::endl;
                }

                previousSegmentation=segmentImage(previousDeformedReferenceSegmentation,alpha);
                if (m_config.verbose)        LOG<<"finished segmentation"<<std::endl;
#endif
                
              
                if (m_config.verbose){
                    ostringstream deformedSegmentationFilename;
                    deformedSegmentationFilename<<m_config.outputDeformedSegmentationFilename<<"-i"<<iter<<suff;
                    
                    ostringstream tmpSegmentationFilename;
                    tmpSegmentationFilename<<m_config.segmentationOutputFilename<<"-i"<<iter<<suff;
                    
                    if (D==3){
                        ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),previousSegmentation);
                        ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),previousDeformedReferenceSegmentation);
                        
                    }else{
                        ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)previousSegmentation, m_config.nSegmentations));
                        ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)previousDeformedReferenceSegmentation ,m_config.nSegmentations));
                        
                    }
                }
                

                double dice=compareSegmentations(oldSegmentation,previousSegmentation,m_config.nSegmentations);
                double dice2=compareSegmentations(previousSegmentation,previousDeformedReferenceSegmentation,m_config.nSegmentations);
                LOG<<endl<<endl<<"----------------------------------------------"<<endl;
                LOG<<D<<" Iteration :"<<iter<<", dice (oldSeg vs. newSeg)="<<dice<<", dice (newSeg vs. newDefSeg)="<<dice2<< std::endl;                   
                LOG<<"----------------------------------------------"<<endl<<endl;; 
                if (iter>=0 || (dice>0.99 && dice2>0.99) || dice>0.99999 ){
                    converged=true;
                }
                ++iter;
                oldSegmentation=previousSegmentation;
            }
            if (D==3){
                ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename,previousSegmentation);
                //ImageUtils<ImageType>::writeImage(m_config.outputDeformedSegmentationFilename,previousDeformedReferenceSegmentation);
                
            }else{
                ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename,makePngFromLabelImage((ConstImagePointerType)previousSegmentation, m_config.nSegmentations));
                ImageUtils<ImageType>::writeImage(m_config.outputDeformedSegmentationFilename,makePngFromLabelImage((ConstImagePointerType)previousDeformedReferenceSegmentation ,m_config.nSegmentations));
                    
            }
            
        }

        
        ImagePointerType registerImagesAndDeformSegmentation(ConstImagePointerType segmentationPrior, double alpha=0){
            ImagePointerType deformedSegmentation;
            
            //define input images
            const ConstImagePointerType targetImage = this->GetInput(0);
            
            const ConstImagePointerType movingImage = this->GetInput(1);
           
            m_SRSPotential->SetReferenceSegmentation(m_movingSegmentationImage);     
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
            m_SRSPotential->SetThreshold(10000000);     
            m_SRSPotential->SetBaseLabelMap(NULL);
            LabelMapperType * labelmapper=new LabelMapperType(0,m_config.maxDisplacement);
        
            int iterationCount=0;
            int level;
            double scale=-1, oldscale=1.0;

            //start pyramid
            //asm volatile("" ::: "memory");
            LabelImagePointerType deformation;
            for (int l=0;l<m_config.nLevels;++l){

                level=m_config.levels[l];
                double labelScalingFactor=1;
                double sigma=1;
                
                //roughly compute downscaling
                scale=1;//7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0];
                scale=scale<1.0?scale:1.0;
                //full resolution at last level of pyramid enforced
                //            if (l==(m_config.nLevels-1)) scale=1.0;
                LOG<<"scale :"<<scale<<std::endl;
                //downsample images if wanted
                ConstImagePointerType downSampledTarget,downSampledReference,downSampledReferenceSegmentation,downSampledTargetSheetness;
                if (scale<1.0){
                    assert(false);
                    downSampledTarget=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(targetImage,sigma),scale);
                    downSampledReference=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(movingImage,sigma),scale);
                 
                    
                }
                else{
                    downSampledTarget=targetImage;
                    downSampledReference=movingImage;
                    downSampledReferenceSegmentation=m_movingSegmentationImage;
                }


                LOG<<"Downsampled images to: "<<downSampledTarget->GetLargestPossibleRegion().GetSize()<<std::endl;

                //init graph
                LOG<<"init graph"<<std::endl;
                GraphModelType graph;
                graph.setFixedImage(downSampledTarget);
                graph.setDisplacementFactor(labelScalingFactor);
                graph.initGraph(level);

   
                double mantisse=(1/m_config.scale);
                int exponent=m_config.nLevels-l;
                if (m_config.imageLevels>0){
                    exponent=max(0,m_config.imageLevels-l);
                }
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1/reductionFactor;
                //unaryRegistrationPot->SetScale(7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0]);
                LOG<<"Scaling : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<endl;
                //setup registration potentials
                m_unaryRegistrationPot->SetScale(scaling);
                m_unaryRegistrationPot->SetRadius(graph.getSpacing());
                m_unaryRegistrationPot->SetFixedImage(downSampledTarget);
                m_unaryRegistrationPot->SetMovingImage(downSampledReference);
                //m_unaryRegistrationPot->SetAtlasSegmentation(downSampledReferenceSegmentation);
                m_unaryRegistrationPot->SetSegmentationPrior((ConstImagePointerType)segmentationPrior);
                m_unaryRegistrationPot->SetAlpha(alpha);
                m_unaryRegistrationPot->SetBeta(alpha == 0 ? 1: m_config.simWeight);

              
             
                m_unaryRegistrationPot->Init();

            
                m_pairwiseRegistrationPot->SetFixedImage(downSampledTarget);
                //downSampledReferenceSegmentation=FilterUtils<ImageType>::NNResample((m_movingSegmentationImage),scaling);

                //register images and potentials
                graph.setUnaryRegistrationFunction(m_unaryRegistrationPot);
                graph.setPairwiseRegistrationFunction(m_pairwiseRegistrationPot);
            
            
                //resample the deformation from last iteration to the current image resolution.
                if (scale!=oldscale){
                    previousFullDeformation=bSplineInterpolateLabelImage(previousFullDeformation, downSampledTarget);
                }
                //now scale it according to spacing difference between this and the previous iteration
                SpacingType sp;
                sp.Fill(1.0);
                if ( oldscale!=-1 && scale!=oldscale) {
                    sp=sp*1.0* scale/oldscale;//level/m_config.levels[l-1];
                    previousFullDeformation=scaleLabelImage(previousFullDeformation,sp);
                }
                oldscale=scale;
                LOG<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
                LOG<<"Current grid size :"<<graph.getGridSize()<<std::endl;
                LOG<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;

                typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                for (int i=0;i<m_config.iterationsPerLevel;++i,++iterationCount){
                    LOG<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";
                    graph.setDisplacementFactor(labelScalingFactor);

                    //register deformation from previous iteration
                    m_unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    m_pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    //	ok what now: create graph! solve graph! save result!Z
                    {

                        MRFSolverType  *mrfSolver= new MRFSolverType(&graph,
                                                                     1,//*exp(-i),
                                                                     m_config.pairwiseRegistrationWeight, 
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     m_config.verbose);
                        mrfSolver->createGraph();
                        mrfSolver->optimize(m_config.optIter);
                        LOG<<" ]"<<std::endl;
                        deformation=graph.getDeformationImage(mrfSolver->getDeformationLabels());
                        
                        delete mrfSolver;
                    }

                    //initialise interpolator
                    //deformation

                    fullDeformation=bSplineInterpolateLabelImage(deformation,downSampledTarget);
                    //fullDeformation=scaleLabelImage(fullDeformation,graph.getDisplacementFactor());
           
                    //apply deformation to moving image
                    ConstIteratorType fixedIt(downSampledTarget,downSampledTarget->GetLargestPossibleRegion());
                    typename CompositionFilterType::Pointer composer=CompositionFilterType::New();
                    composer->SetInput(1,fullDeformation);
                    composer->SetInput(0,previousFullDeformation);
                    composer->Update();
                    LabelImagePointerType composedDeformation=composer->GetOutput();
                    LabelIteratorType labelIt(composedDeformation,composedDeformation->GetLargestPossibleRegion());
                
                    deformedImage=deformImage(downSampledReference,composedDeformation);
                    deformedSegmentationImage=deformSegmentationImage(downSampledReferenceSegmentation,composedDeformation);
                    previousFullDeformation=composedDeformation;
                    labelScalingFactor*=m_config.displacementRescalingFactor;
                    //m_SRSPotential->SetThreshold(max(10.0,10.0*graph.getMaxDisplacementFactor()));     
                    m_SRSPotential->SetThreshold(10000000);     

                    if (false && m_config.verbose){
                        std::string suff;
                        if (ImageType::ImageDimension==2){
                            suff=".nii";
                        }
                        if (ImageType::ImageDimension==3){
                            suff=".nii";
                        }
                        ostringstream deformedFilename;
                        deformedFilename<<m_config.outputDeformedFilename<<"-l"<<l<<"-i"<<i<<suff;
                        ostringstream deformedSegmentationFilename;
                        deformedSegmentationFilename<<m_config.outputDeformedSegmentationFilename<<"-l"<<l<<"-i"<<i<<suff;
                        if (ImageType::ImageDimension==2){
                            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)deformedSegmentationImage,LabelMapperType::nSegmentations));
                        }
                        if (ImageType::ImageDimension==3){
                            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedSegmentationImage);
                        }                       
                    }
                    
                    
                }
                LOG<<std::endl<<std::endl;
            }


            LabelImagePointerType finalDeformation=bSplineInterpolateLabelImage(previousFullDeformation, targetImage);
         
            //if last level of pyramid was not 1:1, the final deformation has to be scaled and the segmentation needs to be resampled
            if (scale!=1.0){
                SpacingType sp;
                sp.Fill(1.0);
                finalDeformation=scaleLabelImage(finalDeformation,sp/scale);

            }
          
            

            //ImagePointerType finalDeformedReference=deformImage(movingImage,finalDeformation);
            ImagePointerType finalDeformedReferenceSegmentation=deformSegmentationImage(m_movingSegmentationImage,finalDeformation);
            return finalDeformedReferenceSegmentation;
        }

        //#define GC
        ImagePointerType segmentImage(ConstImagePointerType deformedSegmentation, double alpha=0){
            ImagePointerType segmentation;

            LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,0);
            LOGV(1)<<labelmapper->nSegmentations<<endl;
#ifndef GC            
            GraphModelType graph;
#else
            typedef SegmentationGraphModel<ImageType,UnarySegmentationPotentialType,PairwiseSegmentationPotentialType,LabelMapperType> SegGraphType;
            SegGraphType graph;
#endif
            const ConstImagePointerType targetImage = this->GetInput(0);
            const ConstImagePointerType fixedGradientImage = this->GetInput(3);

            graph.setFixedImage(targetImage);
            //size of the graph doesn't matter
            graph.initGraph(3);
         
#if 1
            LabelImagePointerType previousFullDeformation=LabelImageType::New();
            previousFullDeformation->SetRegions(targetImage->GetLargestPossibleRegion());
            previousFullDeformation->SetOrigin(targetImage->GetOrigin());
            previousFullDeformation->SetSpacing(targetImage->GetSpacing());
            previousFullDeformation->SetDirection(targetImage->GetDirection());
            previousFullDeformation->Allocate();
            Vector<float, D> tmpVox(0.0);
            previousFullDeformation->FillBuffer(tmpVox);
            m_SRSPotential->SetBaseLabelMap(NULL);
            m_SRSPotential->SetReferenceSegmentation((ConstImagePointerType)deformedSegmentation);     
            //m_SRSPotential->SetThreshold(10);     
            m_SRSPotential->SetThreshold(1000000);     

#endif
            m_unarySegmentationPot->SetAlpha(alpha);
            //m_unarySegmentationPot->SetDeformationPrior((ConstImagePointerType)deformedSegmentation);
            graph.setUnarySegmentationFunction(m_unarySegmentationPot);
            graph.setPairwiseSegmentationFunction(m_pairwiseSegmentationPot);

#ifndef GC
            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
            MRFSolverType  *mrfSolver= new MRFSolverType(&graph,
                                                         0,
                                                         0,
                                                         m_config.rfWeight,
                                                         m_config.pairwiseSegmentationWeight, 
                                                         0,
                                                         m_config.verbose);
#else
            typedef  GC_MRFSolver<SegGraphType> MRFSolverType;
            MRFSolverType * mrfSolver= new MRFSolverType(&graph,
                                                         m_config.rfWeight,
                                                         m_config.pairwiseSegmentationWeight, 
                                                         m_config.verbose);
            
#endif
            mrfSolver->createGraph();
            mrfSolver->optimize(m_config.optIter);
            LOG<<" ]"<<std::endl;

#ifndef GC
            segmentation=graph.getSegmentationImage(mrfSolver->getSegmentationLabels());
#else

            segmentation=graph.getSegmentationImage(mrfSolver->getLabels());
#endif
            if (m_config.verbose)        LOG<<"computed segmentation estimate"<<std::endl;
            delete mrfSolver;
            if (m_config.verbose)        LOG<<"deleted solver"<<std::endl;
            return segmentation;
        }

        LabelImagePointerType bSplineInterpolateLabelImage(LabelImagePointerType labelImg, ConstImagePointerType reference){
            typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
            LabelImagePointerType fullLabelImage;
#if 1
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
                    //			LOG<<k<<" setup"<<std::endl;
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
                        //				LOG<<itCoarse.Get()<<std::endl;
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
    
            std::vector< Iterator> iterators(ImageType::ImageDimension+1);
            for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
                {
                    iterators[k]=Iterator(newImages[k],newImages[k]->GetLargestPossibleRegion());
                    iterators[k].GoToBegin();
                }
            fullLabelImage=LabelImageType::New();
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
                    //				LOG<<k<<" label: "<<iterators[k]->Get()<<std::endl;
                    l[k]=iterators[k].Get();
                    ++((iterators[k]));
                }

                //			lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
                lIt.Set(l);
            }
#else          

            typedef typename itk::VectorLinearInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
            //typedef typename itk::VectorNearestNeighborInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
            typedef typename LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
            typedef typename itk::VectorResampleImageFilter< LabelImageType , LabelImageType>	LabelResampleFilterType;
            LabelInterpolatorPointerType labelInterpolator=LabelInterpolatorType::New();
            labelInterpolator->SetInputImage(labelImg);
            //initialise resampler
            
            typename LabelResampleFilterType::Pointer resampler = LabelResampleFilterType::New();
            //resample deformation field to fixed image dimension
            resampler->SetInput( labelImg );
            resampler->SetInterpolator( labelInterpolator );
            resampler->SetOutputOrigin(reference->GetOrigin());
            resampler->SetOutputSpacing ( reference->GetSpacing() );
            resampler->SetOutputDirection ( reference->GetDirection() );
            resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
            resampler->Update();
            fullLabelImage=resampler->GetOutput();
#if 0
            LabelIterator lIt(fullLabelImage,fullLabelImage->GetLargestPossibleRegion());
            lIt.GoToBegin();
            for (;!lIt.IsAtEnd();++lIt){
                LabelType l=lIt.Get();
                lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
            }
#endif
#endif
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

        ImagePointerType deformImage(ImagePointerType image, LabelImagePointerType deformation){
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
                    imageIt.Set((floor(interpolator->EvaluateAtContinuousIndex(idx)+0.5)));

                }else{
                    imageIt.Set(0);
                }
            }
            return deformed;
        }
 


        ConstImagePointerType fixSegmentationImage(ConstImagePointerType segmentationImage){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            hash_map<int, int> map;
            int c=0;
            for (imageIt.GoToBegin();!imageIt.IsAtEnd();++imageIt){
                int seg=floor(imageIt.Get()+0.5);
                if (map.find(seg)==map.end()){
                    map[seg]=c;
                    ++c;
                }
            }
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                imageIt2.Set(map[floor(imageIt.Get()+0.5)]);
            }
            return (ConstImagePointerType)newImage;
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
        double compareSegmentations(ConstImagePointerType seg1, ConstImagePointerType seg2, int nSegmentations){
            double DICE=0;
            if (seg1 &&seg2){
                ConstIteratorType labelIt(seg1,seg1->GetLargestPossibleRegion());
                ConstIteratorType newLabelIt(seg2,seg2->GetLargestPossibleRegion());
                int tp=1,fp=0,fn=0,tn=0;
                for (newLabelIt.GoToBegin(),labelIt.GoToBegin();!labelIt.IsAtEnd();++labelIt,++newLabelIt){
                    //get segmentation label from optimisation
                    int segmentation=(labelIt.Get());
                    int oldSegmentation=(newLabelIt.Get());
                    if (segmentation==nSegmentations-1){
                        if (oldSegmentation==nSegmentations-1){
                            tp+=1;
                        }
                        else{
                            fp+=1;
                        }
                    }
                    else{
                        if (oldSegmentation==nSegmentations-1){
                            fn+=1;
                        }
                        else{
                            tn+=1;
                        }
                    }
                }
                DICE=2.0*tp/(2*tp+fp+fn);
            }
            return DICE;
        }
    }; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
