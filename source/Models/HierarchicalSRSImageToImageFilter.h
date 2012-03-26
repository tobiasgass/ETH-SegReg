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
#include "Classifier.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkBSplineDeformableTransform.h"
#include <itkWarpImageFilter.h>


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
        typedef typename LabelMapperType::LabelImageType LabelImageType;
        typedef typename LabelImageType::Pointer LabelImagePointerType;
        typedef itk::ImageRegionIterator< LabelImageType>       LabelIteratorType;
        typedef VectorLinearInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
        typedef typename  LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
        typedef itk::VectorResampleImageFilter< LabelImageType , LabelImageType>	LabelResampleFilterType;
 
    
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
    
        typedef  typename itk::DisplacementFieldCompositionFilter<LabelImageType,LabelImageType> CompositionFilterType;
    private:
        SRSConfig m_config;
        LabelImagePointerType m_finalDeformation,m_bulkTransform;
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
        void setBulkTransform(LabelImagePointerType transf){
            m_bulkTransform=transf;
            m_useBulkTransform=true;
        }
        LabelImagePointerType affineRegistration(ConstImagePointerType targetImage, ConstImagePointerType atlasImage){
        }

        LabelImagePointerType getFinalDeformation(){
            return m_finalDeformation;
        }
        ImagePointerType getTargetSegmentationEstimate(){
            return m_finalSegmentation;
        }
        virtual void Update(){
            
            bool segment=m_config.pairwiseSegmentationWeight>0 ||  m_config.unarySegmentationWeight>0 ||m_config.pairwiseCoherenceWeight>0;
            bool regist= m_config.pairwiseRegistrationWeight>0||  m_config.unaryRegistrationWeight>0|| m_config.pairwiseCoherenceWeight>0;

            //define input images
            ConstImagePointerType targetImage = this->GetInput(0);
            ConstImagePointerType atlasImage = this->GetInput(1);
            ConstImagePointerType atlasSegmentationImage;
            
            if (D==2){
                //2d segmentations pngs [from matlab] may have screwed up intensities
                atlasSegmentationImage = fixSegmentationImage(this->GetInput(2),m_config.nSegmentations);
                ImageUtils<ImageType>::writeImage("test.png",atlasSegmentationImage);
                
            }else{
                atlasSegmentationImage = (this->GetInput(2));
            }
            ConstImagePointerType targetGradientImage = this->GetInput(3);
            ConstImagePointerType atlasGradientImage=this->GetInput(4);
            
            //results
            ImagePointerType deformedAtlasImage,deformedAtlasSegmentation,segmentationImage;
            LabelImagePointerType fullDeformation,previousFullDeformation;
        
            if (regist){
                if (m_useBulkTransform){
                    previousFullDeformation=m_bulkTransform;
                }else{
                    //allocate memory
                    previousFullDeformation=LabelImageType::New();
                    previousFullDeformation->SetRegions(targetImage->GetLargestPossibleRegion());
                    previousFullDeformation->SetOrigin(targetImage->GetOrigin());
                    previousFullDeformation->SetSpacing(targetImage->GetSpacing());
                    previousFullDeformation->SetDirection(targetImage->GetDirection());
                    if (m_config.verbose) cout<<"allocating full deformation" <<endl;
                    previousFullDeformation->Allocate();
                    Vector<float, D> tmpVox(0.0);
                    previousFullDeformation->FillBuffer(tmpVox);
                }
            }
            //instantiate potentials
            UnaryRegistrationPotentialPointerType unaryRegistrationPot=UnaryRegistrationPotentialType::New();
            UnarySegmentationPotentialPointerType unarySegmentationPot=UnarySegmentationPotentialType::New();
            PairwiseSegmentationPotentialPointerType pairwiseSegmentationPot=PairwiseSegmentationPotentialType::New();
            PairwiseRegistrationPotentialPointerType pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
            PairwiseCoherencePotentialPointerType pairwiseCoherencePot=PairwiseCoherencePotentialType::New();

            //instantiate interpolators
            SegmentationInterpolatorPointerType segmentationInterpolator=SegmentationInterpolatorType::New();
            ImageInterpolatorPointerType atlasInterpolator=ImageInterpolatorType::New();

            if (segment){
                unarySegmentationPot->SetTargetImage(targetImage);
                unarySegmentationPot->SetTargetGradient((ConstImagePointerType)targetGradientImage);
                unarySegmentationPot->SetAtlasImage(atlasImage);
                unarySegmentationPot->SetAtlasGradient((ConstImagePointerType)atlasGradientImage);
                unarySegmentationPot->SetAtlasSegmentation(atlasSegmentationImage);
                unarySegmentationPot->SetGradientScaling(m_config.pairwiseSegmentationWeight);
                //                unarySegmentationPot->SetTissuePrior((ConstImagePointerType)ImageUtils<ImageType>::readImage(m_config.tissuePriorFilename));
                unarySegmentationPot->Init();
                
                pairwiseSegmentationPot->SetTargetImage(targetImage);
                pairwiseSegmentationPot->SetTargetGradient((ConstImagePointerType)targetGradientImage);
                pairwiseSegmentationPot->SetAtlasImage(atlasImage);
                pairwiseSegmentationPot->SetAtlasGradient((ConstImagePointerType)atlasGradientImage);
                pairwiseSegmentationPot->SetAtlasSegmentation(atlasSegmentationImage);
                pairwiseSegmentationPot->Init();
                if (ImageType::ImageDimension==2){
                    //pairwiseSegmentationPot->evalImage(targetImage,(ConstImagePointerType)targetGradientImage);
                }
            }
            LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        
         
            int iterationCount=0; 

            
            int level;
            double scale=-1;

            //start pyramid
            //asm volatile("" ::: "memory");
            LabelImagePointerType deformation;
            ImagePointerType segmentation;
            if (segment && regist){
                pairwiseCoherencePot->SetNumberOfSegmentationLabels(m_config.nSegmentations);
                pairwiseCoherencePot->SetAtlasSegmentation(atlasSegmentationImage);
            }

          
            for (int l=0;l<m_config.nLevels;++l){

                //compute scaling factor for downsampling the images in the registration potential
                double mantisse=(1/m_config.scale);
                int exponent=m_config.nLevels-l;
                if (m_config.downScale||ImageType::ImageDimension==2){
                    exponent--;
                }
                if (m_config.imageLevels>0){
                    exponent=max(0,m_config.imageLevels-l);
                }
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1/reductionFactor;
                //unaryRegistrationPot->SetScale(7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0]);
                cout<<"Scaling : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<endl;

#if 1
                level=m_config.levels[l];
                double labelScalingFactor=1;
                
                //roughly compute downscaling
                scale=1;//1;//7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0];
                
              
                scale=scale<1.0?scale:1.0;
                //full resolution at last level of pyramid enforced
                //            if (l==(m_config.nLevels-1)) scale=1.0;
                std::cout<<"scale :"<<scale<<std::endl;
                //downsample images if wanted
              
                //init graph
                std::cout<<"init graph"<<std::endl;
                GraphModelType graph;
                graph.setConfig(m_config);
                graph.setTargetImage(targetImage);
                graph.setDisplacementFactor(labelScalingFactor);
                graph.initGraph(level);

            
                atlasInterpolator->SetInputImage(atlasImage);
                
                if (segment && regist)
                    segmentationInterpolator->SetInputImage(atlasSegmentationImage);

                if (regist){
                    //setup registration potentials
                    unaryRegistrationPot->SetScale(scaling);
                    unaryRegistrationPot->SetRadius(graph.getSpacing());
                    unaryRegistrationPot->SetTargetImage(targetImage);
                    unaryRegistrationPot->SetAtlasImage(atlasImage);
#if 0
                    unaryRegistrationPot->SetAtlasSegmentation(atlasSegmentationImage);
                    unaryRegistrationPot->SetAlpha(m_config.alpha);
                    unaryRegistrationPot->SetTargetGradient(targetImageGradient);
#endif          
                    unaryRegistrationPot->Init();
            
                    pairwiseRegistrationPot->SetTargetImage(targetImage);
                    pairwiseRegistrationPot->SetSpacing(graph.getPixelSpacing());
                    
                }

                if (segment){
                    //setup segmentation potentials
                    unarySegmentationPot->SetTargetImage(targetImage);
                    unarySegmentationPot->SetTargetGradient(targetGradientImage);
                    unarySegmentationPot->SetGradientScaling(m_config.pairwiseSegmentationWeight);
                }
                if (segment && regist){
                    //setup segreg potentials
                    pairwiseCoherencePot->SetAtlasSegmentationInterpolator(segmentationInterpolator);
                    //pairwiseCoherencePot->SetAtlasInterpolator(atlasInterpolator);
                    pairwiseCoherencePot->SetAtlasImage(atlasImage);
                    pairwiseCoherencePot->SetTargetImage(targetImage);
                    pairwiseCoherencePot->SetAsymmetryWeight(m_config.asymmetry);
                }

                //register images and potentials
                graph.setUnaryRegistrationFunction(unaryRegistrationPot);
                graph.setPairwiseRegistrationFunction(pairwiseRegistrationPot);
                graph.setUnarySegmentationFunction(unarySegmentationPot);
                graph.setPairwiseCoherenceFunction(pairwiseCoherencePot);
                graph.setPairwiseSegmentationFunction(pairwiseSegmentationPot);


                if (regist && !segment){
                    //if we don't do SRS, the deformation needs only be resampled to the image resolution within the unary registration potential
                    previousFullDeformation=bSplineInterpolateLabelImage(previousFullDeformation, (ConstImagePointerType)unaryRegistrationPot->GetTargetImage());
                }else{
                    previousFullDeformation=bSplineInterpolateLabelImage(previousFullDeformation, targetImage);
                }

                if (regist){
                    if (!segment){
                        unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    }else{
                        unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation,scaling);
                    }
                }

                //now scale it according to spacing difference between this and the previous iteration
                SpacingType sp;
                sp.Fill(1.0);
                std::cout<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
                std::cout<<"Current grid size :"<<graph.getGridSize()<<std::endl;
                std::cout<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
#endif
                //typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
              
                for (int i=0;i<m_config.iterationsPerLevel;++i,++iterationCount){
                    std::cout<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";
                    // displacementfactor decreases with iterations
#if 1
                    graph.setDisplacementFactor(labelScalingFactor);
                    
                    //register deformation from previous iteration
                    if (regist){
                        if (!segment){
                            unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                        }else{
                            unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation,scaling);
                        }
                        pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                        if (segment){
                            pairwiseCoherencePot->SetBaseLabelMap(previousFullDeformation);
                        }
                    }
#endif
                    //if (i>0 || l> 0) pairwiseCoherencePot->SetAtlasSegmentation((ConstImagePointerType)deformedAtlasSegmentation);

                    //	ok what now: create graph! solve graph! save result!Z
                    //double linearIncreasingWeight=1.0/(m_config.nLevels-l);
                    //double expIncreasingWeight=exp(-(m_config.nLevels-l-1));
                    //double linearDecreasingWeight=1-linearIncreasingWeight;
                    //double expDecreasingWeight=exp(-l);
                    //#define TRUNC
                    {
#if 1
                        if (ImageType::ImageDimension==2){
#ifdef TRUNC
                            typedef TRWS_SRSMRFSolverTruncQuadrat2D<GraphModelType> MRFSolverType;
#else
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                            //typedef NewFastPDMRFSolver<GraphModelType> MRFSolverType;
#endif

                            MRFSolverType  *mrfSolver= new MRFSolverType(&graph,
                                                                         m_config.unaryRegistrationWeight,
                                                                         m_config.pairwiseRegistrationWeight * (l>0 || i>0 ) ,
                                                                         m_config.unarySegmentationWeight,
                                                                         m_config.pairwiseSegmentationWeight,
                                                                         m_config.pairwiseCoherenceWeight,
                                                                         m_config.verbose);
                            mrfSolver->createGraph();
                            mrfSolver->optimize(m_config.optIter);
                            std::cout<<" ]"<<std::endl;
                            if (regist){
                                deformation=graph.getDeformationImage(mrfSolver->getDeformationLabels());
                            }
                            segmentation=graph.getSegmentationImage(mrfSolver->getSegmentationLabels());
                              
                            delete mrfSolver;

                        }else{
#ifdef TRUNC
                            typedef TRWS_SRSMRFSolverTruncQuadrat3D<GraphModelType> MRFSolverType;
#else
                            typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
#endif
                            MRFSolverType  *mrfSolver= new MRFSolverType(&graph,
                                                                         m_config.unaryRegistrationWeight,
                                                                         m_config.pairwiseRegistrationWeight, 
                                                                         m_config.unarySegmentationWeight,
                                                                         m_config.pairwiseSegmentationWeight,
                                                                         m_config.pairwiseCoherenceWeight,
                                                                         m_config.verbose);
                            mrfSolver->createGraph();
                            mrfSolver->optimize(m_config.optIter);
                            std::cout<<" ]"<<std::endl;
                            deformation=graph.getDeformationImage(mrfSolver->getDeformationLabels());
                            segmentation=graph.getSegmentationImage(mrfSolver->getSegmentationLabels());
                            delete mrfSolver;

                        }

#else
                        //     HeapProfilerStart("segreg") ;
                        MRFSolverType  *mrfSolver= new MRFSolverType();
                        HeapProfilerDump("dump");
                        mrfSolver->init2();
                        delete mrfSolver;
                        HeapProfilerDump("dump");
                        //HeapProfilerStop();
#endif

                    }

                    //initialise interpolator
                    //deformation
                    LabelImagePointerType composedDeformation;

#if 1
                    if (regist){
                        if (segment)
                            fullDeformation=bSplineInterpolateLabelImage(deformation,targetImage);
                        else
                            fullDeformation=bSplineInterpolateLabelImage(deformation,(ConstImagePointerType)unaryRegistrationPot->GetTargetImage());
                    }
                    //fullDeformation=scaleLabelImage(fullDeformation,graph.getDisplacementFactor());
           
                    //apply deformation to atlas image
                    ConstIteratorType targetIt(targetImage,targetImage->GetLargestPossibleRegion());
                    if (regist){

#if 1
                  
                        typename CompositionFilterType::Pointer composer=CompositionFilterType::New();
                        composer->SetInput(1,fullDeformation);
                        composer->SetInput(0,previousFullDeformation);
                        composer->Update();
                        composedDeformation=composer->GetOutput();
                        LabelIteratorType labelIt(composedDeformation,composedDeformation->GetLargestPossibleRegion());
#else
                        //
                        composedDeformation=LabelImageType ::New();
                        composedDeformation->SetRegions(previousFullDeformation->GetLargestPossibleRegion());
                        composedDeformation->Allocate();
                        LabelIteratorType labelIt(composedDeformation,composedDeformation->GetLargestPossibleRegion());
                        LabelIteratorType label2It(previousFullDeformation,previousFullDeformation->GetLargestPossibleRegion());
                        LabelIteratorType label3It(fullDeformation,fullDeformation->GetLargestPossibleRegion());
                        for (label3It.GoToBegin(),label2It.GoToBegin(),labelIt.GoToBegin();!labelIt.IsAtEnd();++label2It,++labelIt,++label3It){
                            LabelType label=label2It.Get()+label3It.Get();
                            typename ImageInterpolatorType::ContinuousIndexType idx=labelIt.GetIndex();
                            //typename LabelImageType::SizeType size=composedDeformation->GetLargestPossibleRegion().GetSize();
                            idx+=LabelMapperType::getDisplacement(label);
                            labelIt.Set(label);
                        }
#endif
                
                        deformedAtlasImage=warpImage(atlasImage,composedDeformation);
                        deformedAtlasSegmentation=deformSegmentationImage(atlasSegmentationImage,composedDeformation);
                        //deformedAtlasSegmentation=warpImage(atlasSegmentationImage,composedDeformation);
                    }
                    
      
                    //pairwiseCoherencePot->SetThreshold(13);
                    //pairwiseCoherencePot->SetThreshold(max(10.0,10*graph.getMaxDisplacementFactor()));
                    pairwiseCoherencePot->SetThreshold(max(1.0,graph.getMaxDisplacementFactor()));//*(m_config.iterationsPerLevel-i)));
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
                            if (segment) ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)segmentation,LabelMapperType::nSegmentations));
                            if (regist) ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)deformedAtlasSegmentation,LabelMapperType::nSegmentations));
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
                                //		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
                                ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
                                //					ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),deformation);

                                //
                            }
                        }
                    }
                    
#endif
                    

                }
                std::cout<<std::endl<<std::endl;
            }


            m_finalDeformation=bSplineInterpolateLabelImage(previousFullDeformation, targetImage);
            m_finalSegmentation=(segmentation);
            delete labelmapper;
	
        }
        LabelImagePointerType bSplineInterpolateLabelImage(LabelImagePointerType labelImg, ConstImagePointerType atlas){
            typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
            LabelImagePointerType fullLabelImage;
#if 1
            const unsigned int SplineOrder = 3;
            typedef typename itk::Image<float,ImageType::ImageDimension> ParamImageType;
            typedef typename itk::ResampleImageFilter<ParamImageType,ParamImageType> ResamplerType;
            typedef typename itk::BSplineResampleImageFunction<ParamImageType,double> FunctionType;
            typedef typename itk::BSplineDecompositionImageFilter<ParamImageType,ParamImageType>			DecompositionType;
            typedef typename itk::ImageRegionIterator<ParamImageType> Iterator;
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
                    function->SetSplineOrder(SplineOrder);
                    upsampler->SetInput( paramsK );
                    upsampler->SetInterpolator( function );
                    upsampler->SetSize(atlas->GetLargestPossibleRegion().GetSize() );
                    upsampler->SetOutputSpacing( atlas->GetSpacing() );
                    upsampler->SetOutputOrigin( atlas->GetOrigin());
                    upsampler->SetOutputDirection( atlas->GetDirection());
#if 1
                    upsampler->Update();
                    newImages[k]=upsampler->GetOutput();
#else
                    typename DecompositionType::Pointer decomposition = DecompositionType::New();
                    decomposition->SetSplineOrder( SplineOrder );
                    decomposition->SetInput( upsampler->GetOutput() );
                    decomposition->Update();
                    newImages[k] = decomposition->GetOutput();
#endif
                
                }
    
            std::vector< Iterator> iterators(ImageType::ImageDimension+1);
            for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
                {
                    iterators[k]=Iterator(newImages[k],newImages[k]->GetLargestPossibleRegion());
                    iterators[k].GoToBegin();
                }
            fullLabelImage=LabelImageType::New();
            fullLabelImage->SetRegions(atlas->GetLargestPossibleRegion());
            fullLabelImage->SetOrigin(atlas->GetOrigin());
            fullLabelImage->SetSpacing(atlas->GetSpacing());
            fullLabelImage->SetDirection(atlas->GetDirection());
            fullLabelImage->Allocate();
            LabelIterator lIt(fullLabelImage,fullLabelImage->GetLargestPossibleRegion());
            lIt.GoToBegin();
            for (;!lIt.IsAtEnd();++lIt){
                LabelType l;
                for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ ){
                    //				std::cout<<k<<" label: "<<iterators[k]->Get()<<std::endl;
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
            //resample deformation field to target image dimension
            resampler->SetInput( labelImg );
            resampler->SetInterpolator( labelInterpolator );
            resampler->SetOutputOrigin(atlas->GetOrigin());
            resampler->SetOutputSpacing ( atlas->GetSpacing() );
            resampler->SetOutputDirection ( atlas->GetDirection() );
            resampler->SetSize ( atlas->GetLargestPossibleRegion().GetSize() );
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

        ImagePointerType deformImageITK(ConstImagePointerType image, LabelImagePointerType deformation){
            //does not work!!!
            //itk bspline parameters seem to be very differently arranged compared to my own 
            exit(1);

            //cast labelimage into itk transform
            typedef typename itk::BSplineDeformableTransform<double,ImageType::ImageDimension,3> TransformType;
            typedef typename TransformType::CoefficientImageArray ParameterType;
            ParameterType transformParameters;
            typedef typename TransformType::ImagePointer ParamImagePointer;
            typedef typename TransformType::ImageType ParamImageType;
            typedef typename itk::ImageRegionIterator<ParamImageType> Iterator;
            typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;

            for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
                {
                    ParamImagePointer paramsK=ParamImageType::New();
                    paramsK->SetRegions(deformation->GetLargestPossibleRegion());
                    paramsK->SetOrigin(deformation->GetOrigin());
                    paramsK->SetSpacing(deformation->GetSpacing());
                    paramsK->SetDirection(deformation->GetDirection());
                    paramsK->Allocate();
                    Iterator itCoarse( paramsK, paramsK->GetLargestPossibleRegion() );
                    LabelIterator itOld(deformation,deformation->GetLargestPossibleRegion());
                    for (itCoarse.GoToBegin(),itOld.GoToBegin();!itCoarse.IsAtEnd();++itOld,++itCoarse){
                        itCoarse.Set((itOld.Get()[ ImageType::ImageDimension - k - 1 ]));
                    }
                    transformParameters[ k ]=paramsK;
                }
            //set parameters
            typename TransformType::Pointer bSplineTransform=TransformType::New();
            bSplineTransform->SetCoefficientImage(transformParameters);
            //setup resampler
            typedef typename itk::ResampleImageFilter<ImageType,ImageType> ResampleFilterType; 
            typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();
            resampler->SetTransform(bSplineTransform);
            resampler->SetInput(image);
            resampler->SetDefaultPixelValue( 0 );
            resampler->SetSize(    image->GetLargestPossibleRegion().GetSize() );
            resampler->SetOutputOrigin(  image->GetOrigin() );
            resampler->SetOutputSpacing( image->GetSpacing() );
            resampler->SetOutputDirection( image->GetDirection() );
            resampler->Update();
            return resampler->GetOutput();
        }
        
        ImagePointerType warpImage(ConstImagePointerType image, LabelImagePointerType deformation){
            typedef typename itk::WarpImageFilter<ImageType,ImageType,LabelImageType>     WarperType;
            typedef typename WarperType::Pointer     WarperPointer;
            WarperPointer warper=WarperType::New();
            warper->SetInput( image);
            warper->SetDeformationField(deformation);
            warper->SetOutputOrigin(  image->GetOrigin() );
            warper->SetOutputSpacing( image->GetSpacing() );
            warper->SetOutputDirection( image->GetDirection() );
            warper->Update();
            return warper->GetOutput();
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
                    imageIt.Set(int(interpolator->EvaluateAtContinuousIndex(idx)));
                }else{
                    imageIt.Set(0);
                }
            }
            return deformed;
        }
        ConstImagePointerType makePngFromLabelImage(ConstImagePointerType segmentationImage, int nSegmentations){
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
                //    cout<<imageIt.Get()*multiplier<<" "<<multiplier<<endl;
                imageIt2.Set(imageIt.Get()*multiplier);
            }
            return (ConstImagePointerType)newImage;
        }
        
        
        ConstImagePointerType fixSegmentationImage(ConstImagePointerType segmentationImage, int nSegmentations){
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
#if 0
            hash_map<int, int> map;
            int c=0;
            for (imageIt.GoToBegin();!imageIt.IsAtEnd();++imageIt){
                int seg=floor(imageIt.Get()+0.5);
                cout<<seg<<endl;
                if (map.find(seg)==map.end()){
                    map[seg]=c;
                    ++c;
                }
            }
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                imageIt2.Set(map[floor(imageIt.Get()+0.5)]);
            }
#else
            nSegmentations=nSegmentations>0?nSegmentations:2;
            int divisor=std::numeric_limits<PixelType>::max()/(nSegmentations-1);
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                imageIt2.Set(floor(1.0*imageIt.Get()/divisor+0.5));
            }
#endif
            return (ConstImagePointerType)newImage;
        }
    }; //class
} //namespace
#endif /* HIERARCHICALSRSIMAGETOIMAGEFILTER_H_ */
