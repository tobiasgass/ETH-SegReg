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
#include "itkImageRegionConstIteratorWithIndex.h"
#include <google/heap-profiler.h>
#include "ChamferDistanceTransform.h"
#include "itkCastImageFilter.h"
#include "Classifier.h"

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
            ConstImagePointerType movingSegmentationImage;

            if (D==2){
                //2d segmentations pngs [from matlab] may have screwed up intensities
                movingSegmentationImage = fixSegmentationImage(this->GetInput(2),m_config.nSegmentations);
                ImageUtils<ImageType>::writeImage("test.png",movingSegmentationImage);

            }else{
                movingSegmentationImage = (this->GetInput(2));
            }
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
        
#if 1
            ImagePointerType movingGradientImage=ImageUtils<ImageType>::readImage(m_config.movingGradientFilename);
            typedef typename UnarySegmentationPotentialType::ClassifierType ClassifierType;
            typename ClassifierType::Pointer  classifier=  ClassifierType::New();
            classifier->setNIntensities(256);
            if (m_config.nSegmentations){
                classifier->setData(movingImage,movingSegmentationImage,(ConstImagePointerType)movingGradientImage);
                //classifier->setData(movingImage,movingSegmentationImage);
                classifier->train();
                //classifier->evalImage(targetImage);
                classifier->evalImage(targetImage,fixedGradientImage);

                typedef SmoothnessClassifierGradient<ImageType> SmoothClassifierType;
                typename SmoothClassifierType::Pointer  smoothClassifier=  SmoothClassifierType::New();
                smoothClassifier->setNIntensities(256);
                smoothClassifier->setData(movingImage,movingSegmentationImage,(ConstImagePointerType)movingGradientImage);
                smoothClassifier->train();
                unarySegmentationPot->SetSmoothnessClassifier(smoothClassifier);   
                
            }
            std::cout<<"returnedFromClassifier"<<std::endl;
            unarySegmentationPot->SetClassifier(classifier);
            
#endif
            LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,m_config.maxDisplacement);
        
         
            int iterationCount=0; 

            
            int level;
            double scale=-1, oldscale=1.0;

            //start pyramid
            //asm volatile("" ::: "memory");
            LabelImagePointerType deformation;
            ImagePointerType segmentation;
            pairwiseSegmentationRegistrationPot->SetDistanceTransform(getDistanceTransform(movingSegmentationImage,m_config.nSegmentations-1));
            pairwiseSegmentationRegistrationPot->SetNumberOfSegmentationLabels(m_config.nSegmentations);
            if (m_config.nSegmentations>2)
                pairwiseSegmentationRegistrationPot->SetBackgroundDistanceTransform(getDistanceTransform(movingSegmentationImage,1));
            typedef typename itk::CastImageFilter<FloatImageType,ImageType> CasterType;
            typename CasterType::Pointer caster=CasterType::New();
          
            caster->SetInput(pairwiseSegmentationRegistrationPot->GetDistanceTransform());
            caster->Update();
            ImagePointerType output=caster->GetOutput();
            if (ImageType::ImageDimension==2){
                ImageUtils<ImageType>::writeImage("dt.png",(output));    
            }
            if (ImageType::ImageDimension==3){
                ImageUtils<ImageType>::writeImage("dt.nii",(output));
            }
            

            for (int l=0;l<m_config.nLevels;++l){

                //compute scaling factor for downsampling the images in the registration potential
                double mantisse=(1/m_config.scale);
                int exponent=m_config.nLevels-l;
                if (m_config.downScale)
                    exponent--;
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1/reductionFactor;
                //unaryRegistrationPot->SetScale(7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0]);
                cout<<"Scaling : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<endl;

#if 1
                level=m_config.levels[l];
                double labelScalingFactor=1;
                double sigma=1;
                
                //roughly compute downscaling
                scale=1;//7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0];

                if (m_config.downScale){
                    scale=scaling;
                    scaling=0.5;
                }
                scale=scale<1.0?scale:1.0;
                //full resolution at last level of pyramid enforced
                //            if (l==(m_config.nLevels-1)) scale=1.0;
                std::cout<<"scale :"<<scale<<std::endl;
                //downsample images if wanted
                ConstImagePointerType downSampledTarget,downSampledReference,downSampledReferenceSegmentation,downSampledTargetSheetness;
                if (scale<1.0){
                    downSampledTarget=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(targetImage,sigma),scale);
                    downSampledReference=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(movingImage,sigma),scale);
                    downSampledReferenceSegmentation=FilterUtils<ImageType>::NNResample((movingSegmentationImage),scale);
                    downSampledTargetSheetness=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(fixedGradientImage,sigma),scale);
                    //downSampledTargetSheetness=FilterUtils<ImageType>::NNResample((fixedGradientImage),scale);
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

                //             typename itk::ImageRegionConstIteratorWithIndex<ImageType> ii(downSampledTarget, downSampledTarget->GetLargestPossibleRegion());
                //             for (ii.GoToBegin();!ii.IsAtEnd();++ii){
                //                 std::cout<<ii.GetIndex()<<" "<<graph.getClosestGraphIndex(ii.GetIndex())<<std::endl;
                //             }
                movingInterpolator->SetInputImage(downSampledReference);

                segmentationInterpolator->SetInputImage(downSampledReferenceSegmentation);

                //setup registration potentials
                unaryRegistrationPot->SetRadius(graph.getSpacing());
                unaryRegistrationPot->SetFixedImage(downSampledTarget);
                unaryRegistrationPot->SetMovingImage(downSampledReference);
                //unaryRegistrationPot->SetAtlasSegmentation(downSampledReferenceSegmentation);
                //unaryRegistrationPot->SetTargetSheetness(downSampledTargetSheetness);
               
                unaryRegistrationPot->SetScale(scaling);
                unaryRegistrationPot->Init();
            
                pairwiseRegistrationPot->SetFixedImage(downSampledTarget);

                //setup segmentation potentials
                unarySegmentationPot->SetFixedImage(downSampledTarget);
                unarySegmentationPot->SetGradientImage(downSampledTargetSheetness);
                unarySegmentationPot->SetGradientScaling(m_config.pairwiseSegmentationWeight);
                //setup segreg potentials
                pairwiseSegmentationRegistrationPot->SetMovingSegmentationInterpolator(segmentationInterpolator);
                pairwiseSegmentationRegistrationPot->SetMovingInterpolator(movingInterpolator);
                pairwiseSegmentationRegistrationPot->SetFixedImage(downSampledTarget);
                pairwiseSegmentationRegistrationPot->SetAsymmetryWeight(m_config.asymmetry);


                //register images and potentials
                graph.setUnaryRegistrationFunction(unaryRegistrationPot);
                graph.setPairwiseRegistrationFunction(pairwiseRegistrationPot);
                graph.setUnarySegmentationFunction(unarySegmentationPot);
                graph.setPairwiseSegmentationRegistrationFunction(pairwiseSegmentationRegistrationPot);
          
            
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
                std::cout<<"Current displacementFactor :"<<graph.getDisplacementFactor()<<std::endl;
                std::cout<<"Current grid size :"<<graph.getGridSize()<<std::endl;
                std::cout<<"Current grid spacing :"<<graph.getSpacing()<<std::endl;
#endif
                typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                for (int i=0;i<m_config.iterationsPerLevel;++i,++iterationCount){
                    std::cout<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";
                    // displacementfactor decreases with iterations
#if 1
                    graph.setDisplacementFactor(labelScalingFactor);

                    //register deformation from previous iteration
                    unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    pairwiseSegmentationRegistrationPot->SetBaseLabelMap(previousFullDeformation);
#endif
                    //	ok what now: create graph! solve graph! save result!Z
                    double linearIncreasingWeight=1.0/(m_config.nLevels-l);
                    double expIncreasingWeight=exp(-(m_config.nLevels-l-1));
                    double linearDecreasingWeight=1-linearIncreasingWeight;
                    double expDecreasingWeight=exp(-l);
                    {
#if 1
                        MRFSolverType  *mrfSolver= new MRFSolverType(&graph,
                                                                     m_config.simWeight,
                                                                     m_config.pairwiseRegistrationWeight, 
                                                                     m_config.rfWeight,
                                                                     m_config.pairwiseSegmentationWeight,
                                                                     m_config.segWeight,
                                                                     m_config.verbose);
                        mrfSolver->createGraph();
                        mrfSolver->optimize(m_config.optIter);
                        std::cout<<" ]"<<std::endl;
                        deformation=graph.getDeformationImage(mrfSolver->getDeformationLabels());
                        segmentation=graph.getSegmentationImage(mrfSolver->getSegmentationLabels());
                        
                        delete mrfSolver;
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
#if 1
                    fullDeformation=bSplineInterpolateLabelImage(deformation,downSampledTarget);
                    //fullDeformation=scaleLabelImage(fullDeformation,graph.getDisplacementFactor());
           
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
                        //typename LabelImageType::SizeType size=composedDeformation->GetLargestPossibleRegion().GetSize();
                        idx+=LabelMapperType::getDisplacement(label);
                        labelIt.Set(label);
                    }
#endif
                
                    deformedImage=deformImage(downSampledReference,composedDeformation);
                    deformedSegmentationImage=deformSegmentationImage(downSampledReferenceSegmentation,composedDeformation);
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
                        ImageUtils<ImageType>::writeImage(deformedFilename.str().c_str(), deformedImage);
                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<m_config.segmentationOutputFilename<<"-l"<<l<<"-i"<<i<<suff;
                        if (ImageType::ImageDimension==2){
                            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)segmentation,LabelMapperType::nSegmentations));
                            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)deformedSegmentationImage,LabelMapperType::nSegmentations));
                        }
                        if (ImageType::ImageDimension==3){
                            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentation);
                            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedSegmentationImage);
                        }
                        //deformation
                        if (m_config.defFilename!=""){
                            ostringstream tmpDeformationFilename;
                            tmpDeformationFilename<<m_config.defFilename<<"-l"<<l<<"-i"<<i<<".mha";
                            //		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
                            ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),previousFullDeformation);
                            //					ImageUtils<LabelImageType>::writeImage(tmpDeformationFilename.str().c_str(),deformation);

                            //
                        }
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
                //            finalSegmentation=FilterUtils<ImageType>::NNResample(segmentation,1/scale);
                finalSegmentation=FilterUtils<ImageType>::NNResample(segmentation,targetImage);
            }
            if (ImageType::ImageDimension==2){
                ImageUtils<ImageType>::writeImage("dt-def.png",deformImage(output,finalDeformation));    
            }
            if (ImageType::ImageDimension==3){
                ImageUtils<ImageType>::writeImage("dt-def.nii",deformImage(output,finalDeformation));
            }
          
            

            ImagePointerType finalDeformedReference=deformImage(movingImage,finalDeformation);
            ImagePointerType finalDeformedReferenceSegmentation=deformSegmentationImage(movingSegmentationImage,finalDeformation);

      
      
            ImageUtils<ImageType>::writeImage(m_config.outputDeformedFilename, finalDeformedReference);
            if (ImageType::ImageDimension==2){
                ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename,makePngFromLabelImage((ConstImagePointerType)finalSegmentation,LabelMapperType::nSegmentations));
                ImageUtils<ImageType>::writeImage(m_config.outputDeformedSegmentationFilename,makePngFromLabelImage((ConstImagePointerType)finalDeformedReferenceSegmentation,LabelMapperType::nSegmentations));
            }
            if (ImageType::ImageDimension==3){
                ImageUtils<ImageType>::writeImage(m_config.segmentationOutputFilename,finalSegmentation);
                ImageUtils<ImageType>::writeImage(m_config.outputDeformedSegmentationFilename,finalDeformedReferenceSegmentation);
            }


            //deformation
            if (m_config.defFilename!=""){
                //		ImageUtils<LabelImageType>::writeImage(defFilename,deformation);
                ImageUtils<LabelImageType>::writeImage(m_config.defFilename,previousFullDeformation);
                //
            }


            delete labelmapper;
            //	}

	
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
        
        FloatImagePointerType getDistanceTransform(ConstImagePointerType segmentationImage, int value){
#if 0
            typedef ChamferDistanceTransform<ImageType, FloatImageType> CDT;
            CDT cdt;
            return cdt.compute(segmentationImage, CDT::MANHATTEN, true);
#else
            typedef typename itk::SignedMaurerDistanceMapImageFilter< ImageType, FloatImageType > DistanceTransformType;
            typename DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                float val=imageIt.Get();
                imageIt2.Set(val==value);
                
            }
            distanceTransform->SetInput(newImage);
            distanceTransform->SquaredDistanceOn ();
            distanceTransform->UseImageSpacingOn();
            distanceTransform->Update();
            return distanceTransform->GetOutput();
#endif
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
