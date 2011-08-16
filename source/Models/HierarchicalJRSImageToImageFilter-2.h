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
#include "SegmentationGraph.h"
#include "BaseLabel.h"
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
#include "MRF-FAST-PD.h"
namespace itk{
    template<class TImage, 
             class TLabelMapper,
             class TUnaryRegistrationPotential, 
             class TUnarySegmentationPotential,
             class TPairwiseRegistrationPotential           >
    class HierarchicalJRSImageToImageFilter2: public HierarchicalJRSImageToImageFilter2<TImage,TLabelMapper,TUnaryRegistrationPotential,TUnarySegmentationPotential,TPairwiseRegistrationPotential>{
    public:
        typedef HierarchicalJRSImageToImageFilter2 Self;
        typedef HierarchicalJRSImageToImageFilter2<TImage,TLabelMapper,TUnaryRegistrationPotential,TUnarySegmentationPotential,TPairwiseRegistrationPotential> SuperClass;
        typedef SmartPointer< Self >        Pointer;
    
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Run-time type information (and related methods). */
        itkTypeMacro(HierarchicalJRSImageToImageFilter2, ImageToImageFilter);
    
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
        typedef PairwisePotentialSegmentationRegistration<LabelMapperType,ImageType> PairwisePotentialSegmentationRegistrationType;

        typedef typename  UnaryRegistrationPotentialType::Pointer UnaryRegistrationPotentialPointerType;
        typedef typename  UnarySegmentationPotentialType::Pointer UnarySegmentationPotentialPointerType;
        typedef typename  UnaryRegistrationPotentialType::RadiusType RadiusType;
        typedef typename  PairwiseRegistrationPotentialType::Pointer PairwiseRegistrationPotentialPointerType;
        

        typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;

        //typedef ITKGraphModel<UnaryPotentialType,LabelMapperType,ImageType> GraphModelType;
        typedef  GraphModel<ImageType, 
                            UnaryRegistrationPotentialType,
                            PairwiseRegistrationPotentialType,
                            UnarySegmentationPotentialType,
                            PairwisePotentialSegmentationRegistrationType,
                            LabelMapperType> GraphModelType;
        typedef  typename itk::DisplacementFieldCompositionFilter<LabelImageType,LabelImageType> CompositionFilterType;
    
    public:


        virtual void Update(){

            const ConstImagePointerType targetImage = this->GetInput(0);
            const ConstImagePointerType movingImage = this->GetInput(1);
            
            ConstImagePointerType movingSegmentationImage;
            if (D==2){
                //2d segmentations pngs [from matlab] may have screwed up intensities
                movingSegmentationImage = fixSegmentationImage(this->GetInput(2));
            }else{
                movingSegmentationImage = (this->GetInput(2));
            }
      
            //results
            ImagePointerType deformedImage,deformedSegmentationImage,segmentationImage, segmentationPrior;
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
            PairwiseRegistrationPotentialPointerType pairwiseRegistrationPot=PairwiseRegistrationPotentialType::New();
         
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
                std::cout<<"scale :"<<scale<<std::endl;
                //downsample images if wanted
                ConstImagePointerType downSampledTarget,downSampledReference,downSampledReferenceSegmentation,downSampledTargetSheetness;
                if (scale<1.0){
                    downSampledTarget=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(targetImage,sigma),scale);
                    downSampledReference=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(movingImage,sigma),scale);
                    downSampledReferenceSegmentation=FilterUtils<ImageType>::NNResample((movingSegmentationImage),scale);
                }
                else{
                    downSampledTarget=targetImage;
                    downSampledReference=movingImage;
                    downSampledReferenceSegmentation=movingSegmentationImage;
                }
                std::cout<<"Downsampled images to: "<<downSampledTarget->GetLargestPossibleRegion().GetSize()<<std::endl;

                //init graph
                std::cout<<"init graph"<<std::endl;
                GraphModelType graph;
                graph.setFixedImage(downSampledTarget);
                graph.setDisplacementFactor(labelScalingFactor);
                graph.initGraph(level);

   
                double mantisse=(1/m_config.scale);
                int exponent=m_config.nLevels-l;
                double reductionFactor=pow(mantisse,exponent);
                double scaling=1/reductionFactor;
                //unaryRegistrationPot->SetScale(7.0*level/targetImage->GetLargestPossibleRegion().GetSize()[0]);
                cout<<"Scaling : "<<scaling<<" "<<mantisse<<" "<<exponent<<" "<<reductionFactor<<endl;
                //setup registration potentials
                unaryRegistrationPot->SetScale(scaling);
                unaryRegistrationPot->SetRadius(graph.getSpacing());
                unaryRegistrationPot->SetFixedImage(downSampledTarget);
                unaryRegistrationPot->SetMovingImage(downSampledReference);
                unaryRegistrationPot->SetAtlasSegmentation(downSampledReferenceSegmentation);
            
                unaryRegistrationPot->SetAlpha(alpha);
              
             
                unaryRegistrationPot->Init();
            
                pairwiseRegistrationPot->SetFixedImage(downSampledTarget);

                //register images and potentials
                graph.setUnaryRegistrationFunction(unaryRegistrationPot);
                graph.setPairwiseRegistrationFunction(pairwiseRegistrationPot);
            
            
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

                typedef TRWS_SRSMRFSolver<GraphModelType> MRFSolverType;
                for (int i=0;i<m_config.iterationsPerLevel;++i,++iterationCount){
                    std::cout<<"Multiresolution optimization at level "<<l<<" in iteration "<<i<<" :[";
                    graph.setDisplacementFactor(labelScalingFactor);

                    //register deformation from previous iteration
                    unaryRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    pairwiseRegistrationPot->SetBaseLabelMap(previousFullDeformation);
                    
                    int iter=0;
                    while (!converged){
                        unaryRegistrationPot->SetSegmentationPrior((ConstImagePointerType)segmentationPrior);
                        LabelImagePointerType composedDeformation;
                        //register
                        {
                             //	ok what now: create graph! solve graph! save result!Z
                            MRFSolverType  *mrfSolver= new MRFSolverType(&graph,
                                                                         m_config.simWeight,//*exp(-i),
                                                                         m_config.pairwiseRegistrationWeight, 
                                                                         0,
                                                                         0,
                                                                         0,
                                                                         m_config.verbose);
                            mrfSolver->createGraph();
                            mrfSolver->optimize(m_config.optIter);
                            std::cout<<" ]"<<std::endl;
                            deformation=graph.getDeformationImage(mrfSolver->getDeformationLabels());
                            
                            delete mrfSolver;
                        
                        
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
                            composedDeformation=composer->GetOutput();
                            deformedImage=deformImage(downSampledReference,composedDeformation);
                            deformedSegmentationImage=deformSegmentationImage(downSampledReferenceSegmentation,composedDeformation);
                        }
                        
                        
                        //segment
                        {
                            segmentationPrior=segmentImage(deformedSegmentationImage,alpha);
                        }
                     
                        //check convergence
                        double dice=compareSegmentations(oldSegmentation,previousSegmentation);
                        double dice2=compareSegmentations(previousSegmentation,previousDeformedReferenceSegmentation);
                        std::cout<<endl<<endl<<"----------------------------------------------"<<endl;
                        std::cout<<D<<" Iteration :"<<iter<<", dice (oldSeg vs. newSeg)="<<dice<<", dice (newSeg vs. newDefSeg)="<<dice2<< std::endl;                   
                        std::cout<<"----------------------------------------------"<<endl<<endl;; 
                        if (iter>=10 || (dice>0.99 && dice2>0.99) ){
                            converged=true;
                        }
                        ++iter;
                    }
                    previousFullDeformation=composedDeformation;
                    labelScalingFactor*=m_config.displacementRescalingFactor;


                }
                std::cout<<std::endl<<std::endl;
            }


            LabelImagePointerType finalDeformation=bSplineInterpolateLabelImage(previousFullDeformation, targetImage);
         
            //if last level of pyramid was not 1:1, the final deformation has to be scaled and the segmentation needs to be resampled
            if (scale!=1.0){
                SpacingType sp;
                sp.Fill(1.0);
                finalDeformation=scaleLabelImage(finalDeformation,sp/scale);

            }
          
            
            
            //save intermediate results
            std::string suff;
            if (ImageType::ImageDimension==2){
                suff=".png";
            }
            if (ImageType::ImageDimension==3){
                suff=".nii";
            }
            
            if (D==3){
                ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),previousSegmentation);
                ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),previousDeformedReferenceSegmentation);
                
            }else{
                ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)previousSegmentation, m_config.nSegmentations));
                ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),makePngFromLabelImage((ConstImagePointerType)previousDeformedReferenceSegmentation ,m_config.nSegmentations));
                
            }
            
        }

#define GC
        ImagePointerType segmentImage(ImagePointerType deformedSegmentation, double alpha=0){
            ImagePointerType segmentation;

            LabelMapperType * labelmapper=new LabelMapperType(m_config.nSegmentations,0);
            if (verbose) cout<<labelmapper->nSegmentations<<endl;
#ifndef GC            
            GraphModelType graph;
#else
            typedef SegmentationGraphModel<ImageType,UnarySegmentationPotentialType,LabelMapperType> SegGraphType;
            SegGraphType graph;
#endif
            const ConstImagePointerType targetImage = this->GetInput(0);
            const ConstImagePointerType fixedGradientImage = this->GetInput(3);

            graph.setFixedImage(targetImage);
            //size of the graph doesn't matter
            graph.initGraph(3);

            UnarySegmentationPotentialPointerType unarySegmentationPot=UnarySegmentationPotentialType::New();

            unarySegmentationPot->SetFixedImage(targetImage);
            unarySegmentationPot->SetGradientImage(fixedGradientImage);
            unarySegmentationPot->SetDeformationPrior((ConstImagePointerType)deformedSegmentation);
            unarySegmentationPot->SetAlpha(alpha);
            graph.setUnarySegmentationFunction(unarySegmentationPot);
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
            //typedef  GC_MRFSolver<SegGraphType> MRFSolverType;
            typedef NewFastPDMRFSolver<SegGraphType> MRFSolverType;
            MRFSolverType * mrfSolver= new MRFSolverType(&graph,
                                    m_config.rfWeight,
                                    m_config.pairwiseSegmentationWeight, 
                                    m_config.verbose);
            
#endif
            mrfSolver->createGraph();
            mrfSolver->optimize(m_config.optIter);
            std::cout<<" ]"<<std::endl;
#ifndef GC
            segmentation=graph.getSegmentationImage(mrfSolver->getSegmentationLabels());
#else
            segmentation=graph.getSegmentationImage(mrfSolver->getLabels());
#endif
            delete mrfSolver;
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
                    imageIt.Set((floor(interpolator->EvaluateAtContinuousIndex(idx)+0.5)));

                }else{
                    imageIt.Set(0);
                }
            }
            return deformed;
        }
 
        FloatImagePointerType getDistanceTransform(ConstImagePointerType segmentationImage){
#if 0
            typedef ChamferDistanceTransform<ImageType, FloatImageType> CDT;
            CDT cdt;
            return cdt.compute(segmentationImage, CDT::MANHATTEN, true);
#else
            typedef typename itk::SignedMaurerDistanceMapImageFilter< ImageType, FloatImageType > DistanceTransformType;
            typename DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
            distanceTransform->SetInput(segmentationImage);
            distanceTransform->SquaredDistanceOff ();
            distanceTransform->UseImageSpacingOn();
            distanceTransform->Update();
            return distanceTransform->GetOutput();
#endif
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
                //    cout<<imageIt.Get()*multiplier<<" "<<multiplier<<endl;
                imageIt2.Set(imageIt.Get()*multiplier);
            }
            return (ConstImagePointerType)newImage;
        }
        double compareSegmentations(ImagePointerType seg1, ImagePointerType seg2){
            double DICE=0;
            if (seg1 &&seg2){
                IteratorType labelIt(seg1,seg1->GetLargestPossibleRegion());
                IteratorType newLabelIt(seg2,seg2->GetLargestPossibleRegion());
                int tp=1,fp=0,fn=0,tn=0;
                for (newLabelIt.GoToBegin(),labelIt.GoToBegin();!labelIt.IsAtEnd();++labelIt,++newLabelIt){
                    //get segmentation label from optimisation
                    int segmentation=(labelIt.Get());
                    int oldSegmentation=(newLabelIt.Get());
                    if (segmentation>0){
                        if (oldSegmentation>0){
                            tp+=1;
                        }
                        else{
                            fp+=1;
                        }
                    }
                    else{
                        if (oldSegmentation>0){
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
