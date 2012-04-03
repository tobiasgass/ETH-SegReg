#pragma once


#include "itkImage.h"
#include "Log.h"
#include "ImageUtils.h"
#include "itkAffineTransform.h"
#include <iostream>

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
#include <itkVectorResampleImageFilter.h>
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
using namespace std;

template<class ImageType>
class TransfUtils {

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename itk::AffineTransform<double,ImageType::ImageDimension> AffineTransformType;
    typedef typename AffineTransformType::Pointer AffineTransformPointerType;
    static const int D=ImageType::ImageDimension;
    typedef itk::Vector<float,D> DisplacementType;
    typedef itk::Image<DisplacementType,D> DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
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
public:

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
        DeformationFieldPointerType deformation=DeformationFieldType::New();
        deformation->SetRegions(targetImage->GetLargestPossibleRegion());
        deformation->SetOrigin(targetImage->GetOrigin());
        deformation->SetSpacing(targetImage->GetSpacing());
        deformation->SetDirection(targetImage->GetDirection());
        deformation->Allocate();
        ContinuousIndexType centerTargetIndex,centerMovingIndex;
        PointType centerTargetPoint,centerMovingPoint;

        {
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
        {
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
        //compute center of moving image
        PointType movingCenter;
        DisplacementType translation= centerMovingPoint-centerTargetPoint;
        deformation->FillBuffer(translation);
        return deformation;
        
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
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType atlas){ 
        LOGV(2)<<"Extrapolating deformation image"<<std::endl;
        LOGV(3)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<atlas->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
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
        fullDeformationField=DeformationFieldType::New();
        fullDeformationField->SetRegions(atlas->GetLargestPossibleRegion());
        fullDeformationField->SetOrigin(atlas->GetOrigin());
        fullDeformationField->SetSpacing(atlas->GetSpacing());
        fullDeformationField->SetDirection(atlas->GetDirection());
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
#else          

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
        resampler->SetOutputOrigin(atlas->GetOrigin());
        resampler->SetOutputSpacing ( atlas->GetSpacing() );
        resampler->SetOutputDirection ( atlas->GetDirection() );
        resampler->SetSize ( atlas->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        fullDeformationField=resampler->GetOutput();
#if 0
        LabelIterator lIt(fullDeformationField,fullDeformationField->GetLargestPossibleRegion());
        lIt.GoToBegin();
        for (;!lIt.IsAtEnd();++lIt){
            DisplacementType l=lIt.Get();
            lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
        }
#endif
#endif
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
    

};
