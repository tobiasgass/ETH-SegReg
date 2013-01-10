#pragma once


#include "itkImage.h"
#include "Log.h"
#include "ImageUtils.h"
#include "itkAffineTransform.h"
#include <iostream>
#include "FilterUtils.hpp"
#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"
#include <itkBSplineDeformableTransform.h>
#include <itkWarpImageFilter.h>
#include "itkVectorLinearInterpolateImageFunction.h"
#include <itkVectorNearestNeighborInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkDisplacementFieldCompositionFilter.h"
#include <utility>
#include <itkWarpVectorImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include "itkFixedPointInverseDeformationFieldImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkCenteredTransformInitializer.h"
using namespace std;

template<class ImageType, class CDisplacementPrecision=float>
class TransfUtils {

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename itk::AffineTransform<double,ImageType::ImageDimension> AffineTransformType;
    typedef typename AffineTransformType::Pointer AffineTransformPointerType;
    static const int D=ImageType::ImageDimension;

    typedef  CDisplacementPrecision DisplacementPrecision;
    //typedef double DisplacementPrecision;
    typedef itk::Vector<DisplacementPrecision,D> DisplacementType;
    typedef itk::Image<DisplacementType,D> DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename DeformationFieldType::ConstPointer DeformationFieldConstPointerType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SpacingType SpacingType;
    typedef itk::ContinuousIndex<double,D> ContinuousIndexType;
    typedef typename itk::LinearInterpolateImageFunction<ImageType, double> LinearInterpolatorType;
    typedef typename LinearInterpolatorType::Pointer LinearInterpolatorPointerType;
    typedef typename itk::ResampleImageFilter< ImageType , ImageType>	ResampleFilterType;
    typedef typename ResampleFilterType::Pointer ResampleFilterPointerType;
    typedef itk::NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;
    typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;
    typedef typename itk::FixedPointInverseDeformationFieldImageFilter<DeformationFieldType,DeformationFieldType> InverseDeformationFieldFilterType;
    typedef typename InverseDeformationFieldFilterType::Pointer InverseDeformationFieldFilterPointerType;
    typedef itk::Image<float,D> FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef itk::AddImageFilter<DeformationFieldType,DeformationFieldType,
                                DeformationFieldType>                           AdderType;
    typedef typename AdderType::Pointer                  AdderPointer;

    typedef itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,
                                     DeformationFieldType>                           SubtracterType;
    typedef typename SubtracterType::Pointer                  SubtracterPointer;

    typedef itk::DiscreteGaussianImageFilter<DeformationFieldType,DeformationFieldType>  DiscreteGaussianImageFilterType;
    //typedef itk::SmoothingRecursiveGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
    typedef typename DiscreteGaussianImageFilterType::Pointer DiscreteGaussianImageFilterPointer;

public:
    static  DisplacementType zeroDisp(){
        DisplacementType d;
        d->Fill(0);
        return d;
    }; 
    static DeformationFieldPointerType affineToDisplacementField(AffineTransformPointerType affine, ImagePointerType targetImage){
        DeformationFieldPointerType deformation=DeformationFieldType::New();
        deformation->SetRegions(targetImage->GetLargestPossibleRegion());
        deformation->SetOrigin(targetImage->GetOrigin());
        deformation->SetSpacing(targetImage->GetSpacing());
        deformation->SetDirection(targetImage->GetDirection());
        deformation->Allocate();
        typename AffineTransformType::InverseTransformBasePointer inverse=affine->GetInverseTransform();
        typedef itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
        typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        ImageIteratorType imIt(targetImage,targetImage->GetLargestPossibleRegion());
        DeformationIteratorType defIt(deformation,deformation->GetLargestPossibleRegion());
        for (imIt.GoToBegin(),defIt.GoToBegin();!imIt.IsAtEnd();++defIt, ++imIt){
            
            IndexType idx=imIt.GetIndex();
            PointType p,p2;
            
            targetImage->TransformIndexToPhysicalPoint(idx,p);
            
            p2=affine->TransformPoint(p);
            DisplacementType disp;

#ifdef PIXELTRANSFORM
            ContinuousIndexType idx2;
            targetImage->TransformPhysicalPointToIndex(p,idx2);
            for (int d=0;d<D;++d){
                disp[d]=idx2[d]-idx[d];
            }
#else
            for (int d=0;d<D;++d){
                disp[d]=p2[d]-p[d];
            }
#endif
            defIt.Set(disp);
        }
                               
        return deformation;
        
    }
    static DeformationFieldPointerType computeCenteringTransform(ImagePointerType targetImage, ImagePointerType movingImage){
#if 1
        typedef itk::AffineTransform< double, D > TransformType;
        typedef typename itk::CenteredTransformInitializer< 
            TransformType, 
            ImageType, 
            ImageType >  TransformInitializerType;
  
        
        typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
        typename TransformType::Pointer   transform  = TransformType::New();

        initializer->SetTransform(   transform );
        initializer->SetFixedImage( targetImage);
        initializer->SetMovingImage( movingImage );
        initializer->MomentsOn();
        initializer->InitializeTransform();
        
        return affineToDisplacementField(transform,targetImage);
  
#else
        DeformationFieldPointerType deformation=DeformationFieldType::New();
        deformation->SetRegions(targetImage->GetLargestPossibleRegion());
        deformation->SetOrigin(targetImage->GetOrigin());
        deformation->SetSpacing(targetImage->GetSpacing());
        deformation->SetDirection(targetImage->GetDirection());
        deformation->Allocate();
        ContinuousIndexType centerTargetIndex,centerMovingIndex;
        PointType centerTargetPoint,centerMovingPoint;

        {       //compute center of target image

            const typename ImageType::RegionType & targetRegion =
                targetImage->GetLargestPossibleRegion();
            const typename ImageType::IndexType & targetIndex =
                targetRegion.GetIndex();
            const typename ImageType::SizeType & targetSize =
                targetRegion.GetSize();
            
            
            typedef typename PointType::ValueType CoordRepType;
            typedef typename ContinuousIndexType::ValueType ContinuousIndexValueType;
            
            
            for (  int k = 0; k < D; k++ )
                {
                    centerTargetIndex[k] =
                        static_cast< ContinuousIndexValueType >( targetIndex[k] )
                        + static_cast< ContinuousIndexValueType >( targetSize[k] - 1 ) / 2.0;
                }
            
            targetImage->TransformContinuousIndexToPhysicalPoint( centerTargetIndex, centerTargetPoint);
            
        }
        {        //compute center of moving image

            const typename ImageType::RegionType & movingRegion =
                movingImage->GetLargestPossibleRegion();
            const typename ImageType::IndexType & movingIndex =
                movingRegion.GetIndex();
            const typename ImageType::SizeType & movingSize =
                movingRegion.GetSize();
            
            
            typedef typename PointType::ValueType CoordRepType;
            typedef typename ContinuousIndexType::ValueType ContinuousIndexValueType;
            
            
            for ( int k = 0; k < D; k++ )
                {
                    centerMovingIndex[k] =
                        static_cast< ContinuousIndexValueType >( movingIndex[k] )
                        + static_cast< ContinuousIndexValueType >( movingSize[k] - 1 ) / 2.0;
                }
            
            movingImage->TransformContinuousIndexToPhysicalPoint( centerMovingIndex, centerMovingPoint);
            
        }
        DisplacementType translation= centerMovingPoint-centerTargetPoint;
        deformation->FillBuffer(translation);
        return deformation;
#endif        
    }
    

    static AffineTransformPointerType readAffine(std::string filename){

        itk::TransformFileReader::Pointer reader = itk::TransformFileReader::New();
        reader->SetFileName(filename);
        try{
            reader->Update();
        }catch( itk::ExceptionObject & err ){
            LOG<<"could not read affine transform from " <<filename<<std::endl;
            LOG<<"ERR: "<<err<<std::endl;
            exit(0);
        }
        
        typedef itk::TransformFileReader::TransformListType * TransformListType;
        TransformListType transforms = reader->GetTransformList();
        AffineTransformPointerType affine;
        itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
        if(!strcmp((*it)->GetNameOfClass(),"AffineTransform"))
            {
                affine = static_cast<AffineTransformType*>((*it).GetPointer());
            }
        
        if (!affine){
            LOG<<"Expected affine transform, got "<<(*it)->GetNameOfClass()<<", aborting"<<std::endl;
            exit(0);
        }
        return affine;
    }
    static ImagePointerType affineDeformImage(ImagePointerType input, AffineTransformPointerType affine, ImagePointerType target){
        
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
        resampler->SetInterpolator(interpol);
        resampler->SetTransform(affine);
        resampler->SetOutputOrigin(target->GetOrigin());

		resampler->SetOutputSpacing ( target->GetSpacing() );
		resampler->SetOutputDirection ( target->GetDirection() );
		resampler->SetSize ( target->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, ImagePointerType reference){ 
        return bSplineInterpolateDeformationField(labelImg,(ConstImagePointerType)reference);
    }
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, DeformationFieldPointerType reference){ 
        ImagePointerType ref = createEmptyImage(reference);
        return bSplineInterpolateDeformationField(labelImg,(ConstImagePointerType)ref);
    }
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType reference){ 
        LOGV(2)<<"Extrapolating deformation image"<<std::endl;
        LOGV(3)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
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
                function->SetSplineOrder(SplineOrder);
                upsampler->SetInput( paramsK );
                upsampler->SetInterpolator( function );
                upsampler->SetSize(reference->GetLargestPossibleRegion().GetSize() );
                upsampler->SetOutputSpacing( reference->GetSpacing() );
                upsampler->SetOutputOrigin( reference->GetOrigin());
                upsampler->SetOutputDirection( reference->GetDirection());
                upsampler->Update();
#if 1
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
        fullDeformationField=DeformationFieldType::New();
        fullDeformationField->SetRegions(reference->GetLargestPossibleRegion());
        fullDeformationField->SetOrigin(reference->GetOrigin());
        fullDeformationField->SetSpacing(reference->GetSpacing());
        fullDeformationField->SetDirection(reference->GetDirection());
        fullDeformationField->Allocate();
        LabelIterator lIt(fullDeformationField,fullDeformationField->GetLargestPossibleRegion());
        lIt.GoToBegin();
        for (;!lIt.IsAtEnd();++lIt){
            DisplacementType l;
            for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ ){
                //				LOG<<k<<" label: "<<iterators[k]->Get()<<std::endl;
                l[k]=iterators[k].Get();
                ++((iterators[k]));
            }

            //			lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
            lIt.Set(l);
        }

        LOGV(2)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

    static DeformationFieldPointerType linearInterpolateDeformationField(DeformationFieldPointerType labelImg, ImagePointerType reference){ 
            return linearInterpolateDeformationField(labelImg,(ConstImagePointerType)reference);
    }
  static DeformationFieldPointerType linearInterpolateDeformationField(DeformationFieldPointerType labelImg, DeformationFieldPointerType reference){ 
        ImagePointerType ref = createEmptyImage(reference);
        return linearInterpolateDeformationField(labelImg,(ConstImagePointerType)ref);
    }
    static DeformationFieldPointerType linearInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType reference){ 
        LOGV(2)<<"Linearly intrapolating deformation image"<<std::endl;
        LOGV(3)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
        typedef typename itk::VectorLinearInterpolateImageFunction<DeformationFieldType, double> LabelInterpolatorType;
        //typedef typename itk::VectorNearestNeighborInterpolateImageFunction<DeformationFieldType, double> LabelInterpolatorType;
        typedef typename LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
        typedef typename itk::VectorResampleImageFilter< DeformationFieldType , DeformationFieldType>	LabelResampleFilterType;
        LabelInterpolatorPointerType labelInterpolator=LabelInterpolatorType::New();
        labelInterpolator->SetInputImage(labelImg);
        //initialise resampler
            
        typename LabelResampleFilterType::Pointer resampler = LabelResampleFilterType::New();
        //resample deformation field to target image dimension
        resampler->SetInput( labelImg );
        resampler->SetInterpolator( labelInterpolator );
        resampler->SetOutputOrigin(reference->GetOrigin());
        resampler->SetOutputSpacing ( reference->GetSpacing() );
        resampler->SetOutputDirection ( reference->GetDirection() );
        resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        fullDeformationField=resampler->GetOutput();

        LOGV(2)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

       static DeformationFieldPointerType nearestNeighborInterpolateDeformationField(DeformationFieldPointerType labelImg, ImagePointerType reference){ 
            return nearestNeighborInterpolateDeformationField(labelImg,(ConstImagePointerType)reference);
    }
    static DeformationFieldPointerType nearestNeighborInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType reference){ 
        LOGV(2)<<"NearestNeighborly intrapolating deformation image"<<std::endl;
        LOGV(3)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
        typedef typename itk::VectorNearestNeighborInterpolateImageFunction<DeformationFieldType, double> LabelInterpolatorType;
        //typedef typename itk::VectorNearestNeighborInterpolateImageFunction<DeformationFieldType, double> LabelInterpolatorType;
        typedef typename LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
        typedef typename itk::VectorResampleImageFilter< DeformationFieldType , DeformationFieldType>	LabelResampleFilterType;
        LabelInterpolatorPointerType labelInterpolator=LabelInterpolatorType::New();
        labelInterpolator->SetInputImage(labelImg);
        //initialise resampler
            
        typename LabelResampleFilterType::Pointer resampler = LabelResampleFilterType::New();
        //resample deformation field to target image dimension
        resampler->SetInput( labelImg );
        resampler->SetInterpolator( labelInterpolator );
        resampler->SetOutputOrigin(reference->GetOrigin());
        resampler->SetOutputSpacing ( reference->GetSpacing() );
        resampler->SetOutputDirection ( reference->GetDirection() );
        resampler->SetSize ( reference->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        fullDeformationField=resampler->GetOutput();

        LOGV(2)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }
    static DeformationFieldPointerType scaleDeformationField(DeformationFieldPointerType labelImg, SpacingType scalingFactors){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        LabelIterator lIt(labelImg,labelImg->GetLargestPossibleRegion());
        lIt.GoToBegin();
        for (;!lIt.IsAtEnd();++lIt){
            //lIt.Set(LabelMapperType::scaleDisplacement(lIt.Get(),scalingFactors));
        }
        LOG<<"not implemented"<<endl;
        exit(0);
        return labelImg;
    }
    static      ImagePointerType warpImageWithReference(ConstImagePointerType image, DeformationFieldPointerType deformation, ImagePointerType reference, bool nnInterpol=false){
        typedef typename itk::WarpImageFilter<ImageType,ImageType,DeformationFieldType>     WarperType;
        typedef typename WarperType::Pointer     WarperPointer;
        WarperPointer warper=WarperType::New();
        if (nnInterpol){
            NNInterpolatorPointerType nnInt=NNInterpolatorType::New();
            warper->SetInterpolator(nnInt);
        }
        warper->SetInput( image);
        warper->SetDeformationField(deformation);
        warper->SetOutputOrigin(  reference->GetOrigin() );
        warper->SetOutputSpacing( reference->GetSpacing() );
        warper->SetOutputDirection( reference->GetDirection() );
        warper->Update();
        return warper->GetOutput();
    }

    //#define ITK_WARP
#ifdef ITK_WARP
    static      ImagePointerType warpImage(ConstImagePointerType image, DeformationFieldPointerType deformation,bool nnInterpol=false){
        typedef typename itk::WarpImageFilter<ImageType,ImageType,DeformationFieldType>     WarperType;
        typedef typename WarperType::Pointer     WarperPointer;
        WarperPointer warper=WarperType::New();
        if (nnInterpol){
            NNInterpolatorPointerType nnInt=NNInterpolatorType::New();
            warper->SetInterpolator(nnInt);
        }
        warper->SetInput( image);
        warper->SetDeformationField(deformation);
        warper->SetOutputOrigin(  deformation->GetOrigin() );
        warper->SetOutputSpacing( deformation->GetSpacing() );
        warper->SetOutputDirection( deformation->GetDirection() );
        warper->Update();
        return warper->GetOutput();
    }
#else
    static ImagePointerType warpImage(ImagePointerType image, DeformationFieldPointerType deformation,bool nnInterpol=false){
        return warpImage(ConstImagePointerType(image),deformation,nnInterpol);
    }

    static ImagePointerType warpImage(ConstImagePointerType image, DeformationFieldPointerType deformation,bool nnInterpol=false){
        std::pair<ImagePointerType,ImagePointerType> result;
        result = warpImageWithMask(image,deformation,nnInterpol);
        //ImageUtils<ImageType>::writeImage("mask.nii",result.second);
        return result.first;
    }

    static std::pair<ImagePointerType,ImagePointerType> warpImageWithMask(ImagePointerType image, DeformationFieldPointerType deformation,bool nnInterpol=false){
        return warpImageWithMask(ConstImagePointerType(image),deformation,nnInterpol);
    }
    static std::pair<ImagePointerType,ImagePointerType> warpImageWithMask(ConstImagePointerType image, DeformationFieldPointerType deformation,bool nnInterpol=false){
        //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        LabelIterator deformationIt(deformation,deformation->GetLargestPossibleRegion());

        typedef typename itk::LinearInterpolateImageFunction<ImageType, double> ImageInterpolatorType;
        typename ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New(); 
        NNInterpolatorPointerType nnInt=NNInterpolatorType::New();

        if (nnInterpol){
            nnInt->SetInputImage(image);
        }else{
            interpolator->SetInputImage(image);
        }
        ImagePointerType deformed=ImageType::New();//ImageUtils<ImageType>::createEmpty(image);
        deformed->SetRegions(deformation->GetLargestPossibleRegion());
        deformed->SetOrigin(deformation->GetOrigin());
        deformed->SetSpacing(deformation->GetSpacing());
        deformed->SetDirection(deformation->GetDirection());
        deformed->Allocate();
        ImagePointerType mask=ImageUtils<ImageType>::createEmpty((ConstImagePointerType)deformed);
        ImageIterator imageIt(deformed,deformed->GetLargestPossibleRegion());        
        ImageIterator maskIt(mask,mask->GetLargestPossibleRegion());        
        for (maskIt.GoToBegin(),imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt,++maskIt){
            IndexType index=deformationIt.GetIndex();
            typename ImageInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();

            PointType p;
            deformed->TransformIndexToPhysicalPoint(index,p);
            //            LOG<<VAR(p)<<" "<<VAR(displacement)<<endl;
            p+=displacement;
            image->TransformPhysicalPointToContinuousIndex(p,idx);

            bool inside=true;
            if (nnInterpol){
                if (nnInt->IsInsideBuffer(idx)){
                    imageIt.Set(nnInt->EvaluateAtContinuousIndex(idx));
                }else inside=false;
            }else{
                if (interpolator->IsInsideBuffer(idx)){
                    imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
                }else inside=false;
            }
            if (!inside){
                imageIt.Set(0);
                maskIt.Set(0);
            }else{
                maskIt.Set(1);
            }
        }
        pair<ImagePointerType,ImagePointerType> result=std::make_pair(deformed,mask);
        return result;
    }
#endif
    static ImagePointerType warpSegmentationImage(ImagePointerType image, DeformationFieldPointerType deformation){
        return warpImage(image,deformation,true);
    }

    static ImagePointerType deformImage(ConstImagePointerType image, DeformationFieldPointerType deformation){
        //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
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
            DisplacementType displacement=deformationIt.Get();
#ifdef PIXELTRANSFORM
            idx+=(displacement);
#else
            PointType p;
            image->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            image->TransformPhysicalPointToContinuousIndex(p,idx);
#endif
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

    static      ImagePointerType deformImage(ImagePointerType image, DeformationFieldPointerType deformation){
        //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
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
            DisplacementType displacement=deformationIt.Get();
#ifdef PIXELTRANSFORM
            idx+=(displacement);
#else
            PointType p;
            image->TransformIndexToPhysicalPoint(idx,p);
            p+=displacement;
            image->TransformPhysicalPointToContinuousIndex(p,idx);
#endif
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

    static    ImagePointerType deformImageITK(ConstImagePointerType image, DeformationFieldPointerType deformation){
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
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;

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
 
    static     ImagePointerType deformSegmentationImage(ConstImagePointerType segmentationImage, DeformationFieldPointerType deformation){
        //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
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
            DisplacementType displacement=deformationIt.Get();
#ifdef PIXELTRANSFORM
            idx+=(displacement);
#else
            PointType p;
            segmentationImage->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            segmentationImage->TransformPhysicalPointToContinuousIndex(p,idx);
#endif
            if (interpolator->IsInsideBuffer(idx)){
                imageIt.Set(int(interpolator->EvaluateAtContinuousIndex(idx)));
            }else{
                imageIt.Set(0);
            }
        }
        return deformed;
    }
    static DeformationFieldPointerType createEmpty(DeformationFieldPointerType def){
        DeformationFieldPointerType result=DeformationFieldType::New();
        result->SetRegions(def->GetLargestPossibleRegion());
        result->SetOrigin(def->GetOrigin());
        result->SetSpacing(def->GetSpacing());
        result->SetDirection(def->GetDirection());
        result->Allocate();
        DisplacementType tmpVox(0.0);
        result->FillBuffer(tmpVox);
        return result;
    }
    static DeformationFieldPointerType createEmpty(ImagePointerType def){
        DeformationFieldPointerType result=DeformationFieldType::New();
        result->SetRegions(def->GetLargestPossibleRegion());
        result->SetOrigin(def->GetOrigin());
        result->SetSpacing(def->GetSpacing());
        result->SetDirection(def->GetDirection());
        result->Allocate();
        DisplacementType tmpVox(0.0);
        result->FillBuffer(tmpVox);
        return result;
    }

    static FloatImagePointerType createEmptyFloat(DeformationFieldPointerType def){
        FloatImagePointerType result=FloatImageType::New();
        result->SetRegions(def->GetLargestPossibleRegion());
        result->SetOrigin(def->GetOrigin());
        result->SetSpacing(def->GetSpacing());
        result->SetDirection(def->GetDirection());
        result->Allocate();
        result->FillBuffer(0.0);
        return result;
    }
    static ImagePointerType createEmptyImage(DeformationFieldPointerType def){
        ImagePointerType result=ImageType::New();
        result->SetRegions(def->GetLargestPossibleRegion());
        result->SetOrigin(def->GetOrigin());
        result->SetSpacing(def->GetSpacing());
        result->SetDirection(def->GetDirection());
        result->Allocate();
        result->FillBuffer(0.0);
        return result;
    }
    //#define USE_INRIA
#ifdef USE_INRIA
    static DeformationFieldPointerType composeDeformations(DeformationFieldPointerType def1, DeformationFieldPointerType def2){
        typedef  typename itk::DisplacementFieldCompositionFilter<DeformationFieldType,DeformationFieldType> CompositionFilterType;
        typename CompositionFilterType::Pointer composer=CompositionFilterType::New();
        composer->SetInput(1,def1);
        composer->SetInput(0,def2);
        composer->Update();
     
        return composer->GetOutput();
    }
#else
    static DeformationFieldPointerType composeDeformations(DeformationFieldPointerType rightField, DeformationFieldPointerType leftField){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
      
        // Setup the default interpolator
        typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
            DeformationFieldType,double> DefaultFieldInterpolatorType;

        typename DefaultFieldInterpolatorType::Pointer interpolator=DefaultFieldInterpolatorType::New();
        interpolator->SetInputImage(leftField);
     

        DeformationFieldPointerType warpedLeftField=ImageUtils<DeformationFieldType>::createEmpty((DeformationFieldConstPointerType)rightField);
        
        LabelIterator imageIt(warpedLeftField,warpedLeftField->GetLargestPossibleRegion());
        LabelIterator deformationIt(rightField,rightField->GetLargestPossibleRegion());
        for (imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename DefaultFieldInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();
            PointType p;
            rightField->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            //leftField->TransformPhysicalPointToContinuousIndex(p,idx);
            //imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx)+displacement);
            imageIt.Set(interpolator->Evaluate(p)+displacement);
        }
        
        return warpedLeftField;
    }
#endif
    static double computeDeformationNorm(DeformationFieldPointerType def, double exp=2){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;

        double norm=0.0;
        int count=0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt){
            DisplacementType t=deformationIt.Get();
            for (unsigned int d=0;d<D;++d,++count){
                double tmp=pow(fabs(t[d]),exp);
                norm+=tmp;
            }
        }
        return pow(norm,1.0/exp)/count;
    }
    static double computeDeformationNormMask(DeformationFieldPointerType def, ImagePointerType mask, double exp=2){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        ImageIterator imageIt(mask,def->GetLargestPossibleRegion());
        
        double norm=0.0;
        int count=0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt){
            if (imageIt.Get()){
                DisplacementType t=deformationIt.Get();
                for (unsigned int d=0;d<D;++d,++count){
                    double tmp=pow(fabs(t[d]),exp);
                    norm+=tmp;
                }
            }
        }
        return pow(norm/count,1.0/exp);
    }

    static DeformationFieldPointerType warpDeformation(DeformationFieldPointerType img, DeformationFieldPointerType def){
        typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
            DeformationFieldType,double> DefaultFieldInterpolatorType;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;

        typename DefaultFieldInterpolatorType::Pointer interpolator=DefaultFieldInterpolatorType::New();
        interpolator->SetInputImage(img);
        DeformationFieldPointerType deformedImg=ImageUtils<DeformationFieldType>::createEmpty((DeformationFieldConstPointerType)def);
        LabelIterator imageIt(deformedImg,deformedImg->GetLargestPossibleRegion());
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        for (imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename DefaultFieldInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();
            PointType p;
            def->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            img->TransformPhysicalPointToContinuousIndex(p,idx);
            imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
        }
        return deformedImg;

    }

    static FloatImagePointerType computeLocalDeformationNormWeights(DeformationFieldPointerType def, double sigma=1.0, double * averageNorm=NULL){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<FloatImageType> ImageIterator;

        FloatImagePointerType normImage=createEmptyFloat(def);
        double norm=0.0;
        int count=0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        ImageIterator imageIt(normImage,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt,++count){
            DisplacementType t=deformationIt.Get();
            double localNorm=t.GetSquaredNorm();
            imageIt.Set(exp(-localNorm/sigma));
            norm+=localNorm;
        }
        if (averageNorm!=NULL){
            (*averageNorm)=norm/count;
        }
        return normImage;
    }

    static FloatImagePointerType computeLocalDeformationNorm(DeformationFieldPointerType def, double sigma=1.0, double * averageNorm=NULL){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<FloatImageType> ImageIterator;

        FloatImagePointerType normImage=createEmptyFloat(def);
        double norm=0.0;
        int count=0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        ImageIterator imageIt(normImage,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt,++count){
            DisplacementType t=deformationIt.Get();
            double localNorm=t.GetNorm();
            imageIt.Set(localNorm/sigma);
            norm+=localNorm;
        }
        if (averageNorm!=NULL){
            (*averageNorm)=norm/count;
        }
        return normImage;
    }
    static DeformationFieldPointerType locallyScaleDeformation(DeformationFieldPointerType def, FloatImagePointerType weights){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<FloatImageType> ImageIterator;
        double norm=0.0;
        int count=0;
        DeformationFieldPointerType scaledDeformation=ImageUtils<DeformationFieldType>::createEmpty(def);
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        LabelIterator scaledDeformationIt(scaledDeformation,def->GetLargestPossibleRegion());
        ImageIterator imageIt(weights,def->GetLargestPossibleRegion());
        for (scaledDeformationIt.GoToBegin(),deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt,++scaledDeformationIt){
            DisplacementType t=deformationIt.Get();
            float scalar=imageIt.Get();
            t=t*scalar;
            scaledDeformationIt.Set(t);
        }
        return scaledDeformation;
    }

   

    static DeformationFieldPointerType add(DeformationFieldPointerType d1,DeformationFieldPointerType d2){
        AdderPointer m_Adder=AdderType::New();
        m_Adder->InPlaceOff();
        m_Adder->SetInput1(d1);
        m_Adder->SetInput2(d2);
        m_Adder->Update();
        return m_Adder->GetOutput();        
    }
    static void addInPlace(DeformationFieldPointerType d1,DeformationFieldPointerType d2){
        AdderPointer m_Adder=AdderType::New();
        m_Adder->InPlaceOff();
        m_Adder->SetInput1(d1);
        m_Adder->SetInput2(d2);
        m_Adder->Update();
        d1=m_Adder->GetOutput();        
    }

    static DeformationFieldPointerType subtract(DeformationFieldPointerType d1,DeformationFieldPointerType d2){
        SubtracterPointer m_Subtracter=SubtracterType::New();
        m_Subtracter->InPlaceOff();
        m_Subtracter->SetInput1(d1);
        m_Subtracter->SetInput2(d2);
        m_Subtracter->Update();
        return m_Subtracter->GetOutput();        
    }

    static DeformationFieldPointerType locallyInvertScaleDeformation(DeformationFieldPointerType def, ImagePointerType weights){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        double norm=0.0;
        int count=0;
        DeformationFieldPointerType scaledDeformation=ImageUtils<DeformationFieldType>::createEmpty(def);
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        LabelIterator scaledDeformationIt(scaledDeformation,def->GetLargestPossibleRegion());
        ImageIterator imageIt(weights,def->GetLargestPossibleRegion());
        for (scaledDeformationIt.GoToBegin(),deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt,++scaledDeformationIt){
            DisplacementType t=deformationIt.Get();
            float scalar=imageIt.Get();
            if (scalar != 0.0)
                t=t/scalar;
            scaledDeformationIt.Set(t);
        }
        return scaledDeformation;
    }

    static DeformationFieldPointerType multiplyOutOfPlace(DeformationFieldPointerType def1, DeformationFieldPointerType def2){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        double norm=0.0;
        int count=0;
        DeformationFieldPointerType multDef=ImageUtils<DeformationFieldType>::createEmpty(def1);
        LabelIterator def1It(def1,def1->GetLargestPossibleRegion());
        LabelIterator def2It(def2,def2->GetLargestPossibleRegion());
        LabelIterator multDefIt(multDef,multDef->GetLargestPossibleRegion());
        for (def1It.GoToBegin(),def2It.GoToBegin(),multDefIt.GoToBegin();!def1It.IsAtEnd();++def1It,++def2It,++multDefIt){
            DisplacementType t1=def1It.Get();
            DisplacementType t2=def2It.Get();
            for (int d=0;d<D;++d)
                t1[d]=t1[d]*t2[d];
            multDefIt.Set(t1);
        }
        return multDef;
    }

    static void multiply(DeformationFieldPointerType def1, ImagePointerType weights){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        double norm=0.0;
        int count=0;
        LabelIterator def1It(def1,def1->GetLargestPossibleRegion());
        ImageIterator imgIt(weights,weights->GetLargestPossibleRegion());
        imgIt.GoToBegin();
        for (def1It.GoToBegin();!def1It.IsAtEnd();++def1It,++imgIt){
            DisplacementType t1=def1It.Get();
            PixelType w=imgIt.Get();
            for (int d=0;d<D;++d)
                t1[d]=t1[d]*w;
            def1It.Set(t1);
        }
    }
    static void divide(DeformationFieldPointerType def1, ImagePointerType weights){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        double norm=0.0;
        int count=0;
        LabelIterator def1It(def1,def1->GetLargestPossibleRegion());
        ImageIterator imgIt(weights,weights->GetLargestPossibleRegion());
        imgIt.GoToBegin();
        for (def1It.GoToBegin();!def1It.IsAtEnd();++def1It,++imgIt){
            DisplacementType t1=def1It.Get();
            PixelType w=imgIt.Get();
            for (int d=0;d<D;++d){
                if (w!=0.0)
                    t1[d]=t1[d]/w;
                else
                    t1[d]=0.0;
            }
            def1It.Set(t1);
        }
    }

    static DeformationFieldPointerType multiplyOutOfPlace(DeformationFieldPointerType def1, double scalar){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        double norm=0.0;
        int count=0;
        DeformationFieldPointerType multDef=ImageUtils<DeformationFieldType>::createEmpty(def1);
        LabelIterator def1It(def1,def1->GetLargestPossibleRegion());
        LabelIterator multDefIt(multDef,multDef->GetLargestPossibleRegion());
        for (def1It.GoToBegin(),multDefIt.GoToBegin();!def1It.IsAtEnd();++def1It,++multDefIt){
            DisplacementType t1=def1It.Get()*scalar;
            multDefIt.Set(t1);
        }
        return multDef;
    }

 

    
    static DeformationFieldPointerType localSqrt(DeformationFieldPointerType def1){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType multDef=ImageUtils<DeformationFieldType>::createEmpty(def1);
        LabelIterator def1It(def1,def1->GetLargestPossibleRegion());
        LabelIterator multDefIt(multDef,multDef->GetLargestPossibleRegion());
        for (def1It.GoToBegin(),multDefIt.GoToBegin();!def1It.IsAtEnd();++def1It,++multDefIt){
            DisplacementType t1=def1It.Get();
            for (int d=0;d<D;++d)
                t1[d]=sqrt(t1[d]);
            multDefIt.Set(t1);
        }
        return multDef;
    }
   

    static DeformationFieldPointerType invert(DeformationFieldPointerType def, ImagePointerType ref=NULL){
        InverseDeformationFieldFilterPointerType inverter=InverseDeformationFieldFilterType::New();
        inverter->SetInput(def);
        if (ref.IsNotNull()){
            inverter->SetOutputOrigin(ref->GetOrigin());
            inverter->SetSize(ref->GetLargestPossibleRegion().GetSize());
            inverter->SetOutputSpacing(ref->GetSpacing());
            
        }else{
            inverter->SetOutputOrigin(def->GetOrigin());
            inverter->SetSize(def->GetLargestPossibleRegion().GetSize());
            inverter->SetOutputSpacing(def->GetSpacing());
        }       
        inverter->SetNumberOfIterations(10);
        inverter->Update();
        return inverter->GetOutput();
        
    }

    static DeformationFieldPointerType gaussian(
                                       DeformationFieldPointerType image, float variance
                                       ) {

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();

        filter->SetInput(image);
        //filter->SetSigma(variance);
        filter->SetVariance(variance*variance);
        filter->Update();

        return filter->GetOutput();
    }

    
};
