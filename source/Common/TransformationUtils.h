#pragma once


#include "itkImage.h"
#include "Log.h"
#include "ImageUtils.h"
#include "itkAffineTransform.h"
#include <iostream>
#include "FilterUtils.hpp"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkTransformFactoryBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineTransform.h"
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
#include <itkVectorResampleImageFilter.h>
#include <itkDisplacementFieldTransform.h>
#include "itkTransformFactoryBase.h"
#include "itkTransformFactory.h"
#include "itkMatrixOffsetTransformBase.h"
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkDisplacementFieldToBSplineImageFilter.h>
#include "itkConstantPadImageFilter.h"
#include "itkTranslationTransform.h"

using namespace std;

template<typename ImageType, typename CDisplacementPrecision=float, typename COutputPrecision=double,typename CFloatPrecision=float>
class TransfUtils {

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename itk::AffineTransform<double,ImageType::ImageDimension> AffineTransformType;
    //typedef typename itk::Transform<double,ImageType::ImageDimension> AffineTransformType;
    typedef typename AffineTransformType::Pointer AffineTransformPointerType;
    static const int D=ImageType::ImageDimension;

    typedef  CDisplacementPrecision DisplacementPrecision;
    //typedef double DisplacementPrecision;
    typedef itk::Vector<DisplacementPrecision,D> DisplacementType;
    typedef itk::Image<DisplacementType,D> DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename DeformationFieldType::ConstPointer DeformationFieldConstPointerType;

    typedef itk::Vector<COutputPrecision,D> OutputDisplacementType;
    typedef itk::Image<OutputDisplacementType,D> OutputDeformationFieldType;
    typedef typename OutputDeformationFieldType::Pointer OutputDeformationFieldPointerType;
    typedef typename OutputDeformationFieldType::ConstPointer OutputDeformationFieldConstPointerType;

    
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
    typedef itk::Image<CFloatPrecision,D> FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef itk::AddImageFilter<DeformationFieldType,DeformationFieldType,
                                DeformationFieldType>                           AdderType;
    typedef typename AdderType::Pointer                  AdderPointer;

    typedef itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,
                                     DeformationFieldType>                           SubtracterType;
    typedef typename SubtracterType::Pointer                  SubtracterPointer;
    //#define DISCRETEGAUSSIAN

#ifdef DISCRETEGAUSSIAN
    typedef itk::DiscreteGaussianImageFilter<DeformationFieldType,DeformationFieldType>  DiscreteGaussianImageFilterType;
#else
    typedef itk::SmoothingRecursiveGaussianImageFilter<DeformationFieldType,DeformationFieldType>  DiscreteGaussianImageFilterType;
#endif
    typedef typename DiscreteGaussianImageFilterType::Pointer DiscreteGaussianImageFilterPointer;
    
    typedef map< string, map <string, DeformationFieldPointerType> > DeformationCacheType;
    typedef map< string, ImagePointerType > ImageCacheType;

    typedef itk::DisplacementFieldTransform<CDisplacementPrecision, D> DisplacementFieldTransformType;

    typedef typename itk::BSplineTransform<CDisplacementPrecision,D,3 >     BSplineTransformType;
    typedef typename BSplineTransformType::Pointer BSplineTransformPointerType;
    
public:
    static  DisplacementType zeroDisp(){
        DisplacementType d;
        d->Fill(0);
        return d;
    }; 

    static OutputDeformationFieldPointerType cast(DeformationFieldPointerType def){
        typedef itk::CastImageFilter <DeformationFieldType,OutputDeformationFieldType> CastImageFilterType;
        typedef typename CastImageFilterType::Pointer CastImageFilterPointer;
        CastImageFilterPointer caster=CastImageFilterType::New();
        caster->SetInput(def);
        caster->Update();
        return caster->GetOutput();
        
    }
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
        typedef itk::AffineTransform<double,ImageType::ImageDimension> TransformType;
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

        typedef itk::MatrixOffsetTransformBase< double, 3, 3 > MatrixOffsetTransformType;
        itk::TransformFactory<MatrixOffsetTransformType>::RegisterTransform();
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
        if(true) //!strcmp((*it)->GetNameOfClass(),"AffineTransform"))
            {
                affine = static_cast<AffineTransformType*>((*it).GetPointer());
            }
        
        if (!affine){
            LOG<<"Expected affine transform, got "<<(*it)->GetNameOfClass()<<", aborting"<<std::endl;
            exit(0);
        }
        LOGV(3)<<VAR(affine)<<endl;
        return affine;
    }

    
    static void writeAffine(std::string filename,AffineTransformPointerType aff){

        typedef itk::MatrixOffsetTransformBase< double, 3, 3 > MatrixOffsetTransformType;
        itk::TransformFactory<MatrixOffsetTransformType>::RegisterTransform();
    
        itk::TransformFileWriter::Pointer writer = itk::TransformFileWriter::New();
        writer->SetFileName(filename);
        writer->SetInput(aff);
        try{
            writer->Update();
        }catch( itk::ExceptionObject & err ){
            LOG<<"could not write affine transform from " <<filename<<std::endl;
            LOG<<"ERR: "<<err<<std::endl;
            exit(0);
        }
    }

    static ImagePointerType affineDeformImage(ImagePointerType input, AffineTransformPointerType affine, ImagePointerType target, bool NN = false){
        
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        NNInterpolatorPointerType interpol2=NNInterpolatorType::New();
        ResampleFilterPointerType resampler=ResampleFilterType::New();
        resampler->SetInput(input);
        if (NN){
            resampler->SetInterpolator(interpol2);
        }else{
            resampler->SetInterpolator(interpol);
        }
        resampler->SetTransform(affine);
        resampler->SetOutputOrigin(target->GetOrigin());

		resampler->SetOutputSpacing ( target->GetSpacing() );
		resampler->SetOutputDirection ( target->GetDirection() );
		resampler->SetSize ( target->GetLargestPossibleRegion().GetSize() );
        resampler->Update();
        return resampler->GetOutput();
    }
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, ImagePointerType reference,bool smooth=false){ 
        return bSplineInterpolateDeformationField(labelImg,(ConstImagePointerType)reference,smooth);
    }
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, DeformationFieldPointerType reference,bool smooth=false){ 
        ImagePointerType ref = createEmptyImage(reference);
        return bSplineInterpolateDeformationField(labelImg,(ConstImagePointerType)ref,smooth);
    }
    static DeformationFieldPointerType bSplineInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType reference,bool smooth=false){ 
        if (labelImg->GetLargestPossibleRegion().GetSize()==reference->GetLargestPossibleRegion().GetSize() && labelImg->GetSpacing()==reference->GetSpacing()  && labelImg->GetOrigin()==reference->GetOrigin()){
            return ImageUtils<DeformationFieldType>::duplicate(labelImg);
        }
        else if ( labelImg->GetLargestPossibleRegion().GetSize()[0]>reference->GetLargestPossibleRegion().GetSize()[0]){
            //downsampling does not need bspline interpolation
            //note that the test for downsampling is pretty crude...
            typedef typename itk::VectorResampleImageFilter<DeformationFieldType,DeformationFieldType> ResamplerType;
            typename ResamplerType::Pointer resampler=ResamplerType::New();
            if (smooth){
                DeformationFieldPointerType smoothedInput = gaussian(labelImg,reference->GetSpacing()-labelImg->GetSpacing());
                //DeformationFieldPointerType smoothedInput = gaussian(labelImg,reference->GetSpacing()*0.5);
                resampler->SetInput(smoothedInput);
                LOGV(4)<<"Downsampling deformation image (with smoothing)"<<std::endl;

            }else{
                resampler->SetInput(labelImg);
                LOGV(4)<<"Downsampling deformation image (without smoothing)"<<std::endl;
            }
            LOGV(3)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;

            resampler->SetSize(reference->GetLargestPossibleRegion().GetSize() );
            resampler->SetOutputSpacing( reference->GetSpacing() );
            resampler->SetOutputOrigin( reference->GetOrigin());
            resampler->SetOutputDirection( reference->GetDirection());
            resampler->Update();
            return resampler->GetOutput();

        }
        LOGV(5)<<"Upsampling deformation image"<<std::endl;
        LOGV(6)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
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

        LOGV(6)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

    static DeformationFieldPointerType linearInterpolateDeformationField(DeformationFieldPointerType labelImg, ImagePointerType reference, bool smooth=false){ 
        return linearInterpolateDeformationField(labelImg,(ConstImagePointerType)reference,smooth);
    }
    static DeformationFieldPointerType linearInterpolateDeformationField(DeformationFieldPointerType labelImg, DeformationFieldPointerType reference,bool smooth=false){ 
        ImagePointerType ref = createEmptyImage(reference);
        return linearInterpolateDeformationField(labelImg,(ConstImagePointerType)ref,smooth);
    }
    static DeformationFieldPointerType linearInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType reference,bool smooth=false){ 
        if (labelImg->GetLargestPossibleRegion().GetSize()==reference->GetLargestPossibleRegion().GetSize() && labelImg->GetSpacing()==reference->GetSpacing() && labelImg->GetOrigin()==reference->GetOrigin()){
            return ImageUtils<DeformationFieldType>::duplicate(labelImg);
        }
        else if ( labelImg->GetLargestPossibleRegion().GetSize()[0]>reference->GetLargestPossibleRegion().GetSize()[0]){
            //note that the test for downsampling is pretty crude...
            typedef typename itk::VectorResampleImageFilter<DeformationFieldType,DeformationFieldType> ResamplerType;
            typename ResamplerType::Pointer resampler=ResamplerType::New();
            if (smooth){
                DeformationFieldPointerType smoothedInput = gaussian(labelImg,reference->GetSpacing()-labelImg->GetSpacing());
                //DeformationFieldPointerType smoothedInput = gaussian(labelImg,reference->GetSpacing()*0.5);
                resampler->SetInput(smoothedInput);
                LOGV(4)<<"Downsampling deformation image (with smoothing)"<<std::endl;

            }else{
                resampler->SetInput(labelImg);
                LOGV(4)<<"Downsampling deformation image (without smoothing)"<<std::endl;
            }
            LOGV(3)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;

            resampler->SetSize(reference->GetLargestPossibleRegion().GetSize() );
            resampler->SetOutputSpacing( reference->GetSpacing() );
            resampler->SetOutputOrigin( reference->GetOrigin());
            resampler->SetOutputDirection( reference->GetDirection());
            resampler->Update();
            return resampler->GetOutput();

        }


        LOGV(5)<<"Linearly intrapolating deformation image"<<std::endl;
        LOGV(6)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
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

        LOGV(6)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

    static DeformationFieldPointerType nearestNeighborInterpolateDeformationField(DeformationFieldPointerType labelImg, ImagePointerType reference){ 
        return nearestNeighborInterpolateDeformationField(labelImg,(ConstImagePointerType)reference);
    }
    static DeformationFieldPointerType nearestNeighborInterpolateDeformationField(DeformationFieldPointerType labelImg, ConstImagePointerType reference){ 
        LOGV(5)<<"NearestNeighborly intrapolating deformation image"<<std::endl;
        LOGV(6)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
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

        LOGV(6)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

    static DeformationFieldPointerType computeBSplineTransformFromDeformationField(DeformationFieldPointerType labelImg,ImagePointerType reference){ 
     
        LOGV(5)<<"computing bspline transform parameters"<<std::endl;
        LOGV(6)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
        const unsigned int SplineOrder = 3;
        typedef typename itk::Image<CDisplacementPrecision,ImageType::ImageDimension> ParamImageType;
        typedef typename itk::ResampleImageFilter<ParamImageType,ParamImageType> ResamplerType;
        typedef typename itk::BSplineResampleImageFunction<ParamImageType,double> FunctionType;
        typedef typename itk::BSplineDecompositionImageFilter<ParamImageType,ParamImageType>			DecompositionType;
        typedef typename itk::ImageRegionIterator<ParamImageType> Iterator;
        std::vector<typename ParamImageType::Pointer> newImages(ImageType::ImageDimension);
        //interpolate deformation
        for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
            {
                //			LOG<<k<<" setup"<<std::endl;
                typename ParamImageType::Pointer paramsK=getComponent(labelImg,k);
                typename DecompositionType::Pointer decomposition = DecompositionType::New();
                typename DecompositionType::Pointer decomposition2 = DecompositionType::New();
                typename ResamplerType::Pointer resampler = ResamplerType::New();
                typename FunctionType::Pointer function = FunctionType::New();
                //first decomposition gets bSpline parameters at full resolution
                decomposition->SetSplineOrder( SplineOrder );
                decomposition->SetInput( paramsK );

                //now get bSpline interpolated deformation field at requested bSpline grid resolution
                function->SetSplineOrder(SplineOrder);
                resampler->SetInput(decomposition->GetOutput());
                resampler->SetInterpolator( function );
                resampler->SetSize(reference->GetLargestPossibleRegion().GetSize() );
                resampler->SetOutputSpacing( reference->GetSpacing() );
                resampler->SetOutputOrigin( reference->GetOrigin());
                resampler->SetOutputDirection( reference->GetDirection());
#if 1              
                //lastly compute Bspline coefficients for said resolution oO
                decomposition2->SetSplineOrder( SplineOrder );
                decomposition2->SetInput( resampler->GetOutput() );
                decomposition2->Update();
                newImages[k] = decomposition2->GetOutput();
#else
                resampler->Update();
                newImages[k] = resampler->GetOutput();
#endif
            }
        std::vector< Iterator> iterators(ImageType::ImageDimension);
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
                l[k]=iterators[k].Get();
                ++((iterators[k]));
            }
            lIt.Set(l);
        }
        LOGV(6)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

    static DeformationFieldPointerType deformationFieldToBSpline(DeformationFieldPointerType labelImg, ImagePointerType reference){
        typedef typename itk::DisplacementFieldToBSplineImageFilter<DeformationFieldType> FilterType;
        typename FilterType::Pointer bspliner=FilterType::New();
        bspliner->SetDisplacementField(labelImg);
        typename FilterType::ArrayType numberOfControlPoints;
        int splineOrder=3;
        for (int d=0;d<D;++d){
            numberOfControlPoints[d]=reference->GetLargestPossibleRegion().GetSize()[d];//+splineOrder-1;
        }
        LOG<<VAR(numberOfControlPoints)<<endl;
        bspliner->SetNumberOfControlPoints(numberOfControlPoints);
        bspliner->SetNumberOfFittingLevels( 1 );
        bspliner->SetSplineOrder( splineOrder );
        bspliner->EnforceStationaryBoundaryOff();
        //bspliner->EnforceStationaryBoundaryOn();
        //        bspliner->SetEnforceStationaryBoundary( false );
        bspliner->EstimateInverseOff();
        try{
            bspliner->Update();
        }catch( itk::ExceptionObject & excp )
            {
                std::cerr << "Exception thrown " << std::endl;
                std::cerr << excp << std::endl;
            }
        DeformationFieldPointerType result=ImageUtils<DeformationFieldType>::duplicateConst(bspliner->GetDisplacementFieldControlPointLattice());
        LOG<<result->GetLargestPossibleRegion().GetSize()<<endl;
        LOG<<result->GetSpacing()<<endl;
#if 0
        result->SetSpacing(reference->GetSpacing());
        result->SetOrigin(reference->GetOrigin());
        result->SetDirection(reference->GetDirection());


        typedef typename itk::ConstantPadImageFilter< DeformationFieldType,DeformationFieldType > PadFilterType;
        typename PadFilterType::Pointer padder=PadFilterType::New();
        typename ImageType::SizeType lower,upper;
        lower.Fill(1);
        upper.Fill(1);
        padder->SetPadLowerBound(lower);
        padder->SetPadUpperBound(upper);
        DisplacementType zer;
        zer.Fill(0);
        padder->SetConstant(zer);
        padder->SetInput(result);
        padder->Update();
        result=padder->GetOutput();
#endif
        return bspliner->GetOutput();;
        
    }



    static DeformationFieldPointerType computeDeformationFieldFromBSplineTransform(DeformationFieldPointerType labelImg,ImagePointerType reference){ 
     
        LOGV(5)<<"computing bspline transform parameters"<<std::endl;
        LOGV(6)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
        typedef typename itk::Image<CDisplacementPrecision,ImageType::ImageDimension> ParamImageType;
        typedef typename ParamImageType::Pointer ParamImagePointerType;
        //interpolate deformation
        fullDeformationField=DeformationFieldType::New();
        fullDeformationField->SetRegions(reference->GetLargestPossibleRegion());
        fullDeformationField->SetOrigin(reference->GetOrigin());
        fullDeformationField->SetSpacing(reference->GetSpacing());
        fullDeformationField->SetDirection(reference->GetDirection());
        fullDeformationField->Allocate();
        typedef typename itk::BSplineTransform<
            CDisplacementPrecision,
            D,
            3 >     DeformableTransformType;
        typename DeformableTransformType::Pointer bSplineTransform=DeformableTransformType::New();

        ParamImagePointerType parameterImages[D];
        for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
            {
                parameterImages[k]=getComponent(labelImg,k);
                
            }
        try{
            bSplineTransform->SetCoefficientImages( parameterImages );
        }catch( itk::ExceptionObject & err ){
            LOG<<err<<endl;
            exit(0);
        }

        LOGV(5)<<"Initialised bspline, now allocating iterator"<<endl;
        LabelIterator lIt(fullDeformationField,fullDeformationField->GetLargestPossibleRegion());
        lIt.GoToBegin();
        LOGV(5)<<"beginning iteration"<<endl;
        for (;!lIt.IsAtEnd();++lIt){        
            LOGV(5)<<"inside iteration"<<endl;
            IndexType idx=lIt.GetIndex();
            LOGV(5)<<idx<<endl;
            PointType originPoint, deformedPoint;
            if (reference->GetLargestPossibleRegion().IsInside(idx)){
                LOGV(5)<<"is inside!"<<endl;
                reference->TransformIndexToPhysicalPoint(idx,originPoint);
                LOGV(5)<<VAR(originPoint)<<endl;
                try{
                    deformedPoint=bSplineTransform->TransformPoint(originPoint);
                }catch( itk::ExceptionObject & err ){
                    LOG<<err<<endl;
                    exit(0);
                }

                LOGV(5)<<VAR(deformedPoint)<<endl;
                DisplacementType deformation=deformedPoint-originPoint;
                lIt.Set(deformation);
            }else{
                LOG<<VAR(idx)<<" not inside reference???"<<std::endl;
            }
        }
        LOGV(6)<<"Finshed extrapolation"<<std::endl;
        return fullDeformationField;
    }

    static BSplineTransformPointerType computeITKBSplineTransformFromDeformationField(DeformationFieldPointerType labelImg,ImagePointerType reference){ 
     
        LOGV(5)<<"computing bspline transform parameters"<<std::endl;
        LOGV(6)<<"From: "<<labelImg->GetLargestPossibleRegion().GetSize()<<" to: "<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
        const unsigned int SplineOrder = 3;
        typedef typename itk::Image<CDisplacementPrecision,ImageType::ImageDimension> ParamImageType;
        typedef typename itk::ResampleImageFilter<ParamImageType,ParamImageType> ResamplerType;
        typedef typename itk::BSplineResampleImageFunction<ParamImageType,double> FunctionType;
        typedef typename itk::BSplineDecompositionImageFilter<ParamImageType,ParamImageType>			DecompositionType;
        typedef typename itk::ImageRegionIterator<ParamImageType> Iterator;
        //std::vector<typename ParamImageType::Pointer> newImages(ImageType::ImageDimension);
        typedef typename ParamImageType::Pointer ParamImagePointerType;
        ParamImagePointerType newImages[D];
        BSplineTransformPointerType bsplineTransform=BSplineTransformType::New();
   
#if 0
        bsplineTransform->SetTransformDomainOrigin(reference->GetOrigin()  );
        
        typename BSplineTransformType::MeshSizeType meshSize;
        typename BSplineTransformType::PhysicalDimensionsType   physicalDimensions;
        for (int d=0;d<D;++d){
            meshSize[d]=reference->GetLargestPossibleRegion().GetSize()[d]-SplineOrder;
            physicalDimensions=(reference->GetLargestPossibleRegion().GetSize()[d]-1)*reference->GetSpacing()[d];

        }

        bsplineTransform->SetTransformDomainPhysicalDimensions( physicalDimensions  );
        bsplineTransform->SetTransformDomainDirection( reference->GetDirection() );
        bsplineTransform->SetTransformDomainMeshSize(meshSize);
#endif
        //interpolate deformation
        for ( unsigned int k = 0; k < ImageType::ImageDimension; k++ )
            {
                //			LOG<<k<<" setup"<<std::endl;
                typename ParamImageType::Pointer paramsK=getComponent(labelImg,k);
                typename DecompositionType::Pointer decomposition = DecompositionType::New();
                typename DecompositionType::Pointer decomposition2 = DecompositionType::New();
                typename ResamplerType::Pointer resampler = ResamplerType::New();
                typename FunctionType::Pointer function = FunctionType::New();
              
#if 0
                //first decomposition gets bSpline parameters at full resolution
                decomposition->SetSplineOrder( SplineOrder );
                decomposition->SetInput( paramsK );
                
                typename ParamImageType::Pointer targetparamsK=bsplineTransform->GetCoefficientImages()[k];

                //now get bSpline interpolated deformation field at requested bSpline grid resolution
                function->SetSplineOrder(SplineOrder);
                resampler->SetInput(decomposition->GetOutput());
                resampler->SetInterpolator( function );
                resampler->SetSize(targetparamsK->GetLargestPossibleRegion().GetSize() );
                resampler->SetOutputSpacing( targetparamsK->GetSpacing() );
                resampler->SetOutputOrigin( targetparamsK->GetOrigin());
                resampler->SetOutputDirection( targetparamsK->GetDirection());
                resampler->Update();
                newImages[k]=resampler->GetOutput();

#else
                typename ParamImageType::Pointer downsampleddeformation=FilterUtils<ParamImageType>::LinearResample(paramsK,FilterUtils<ImageType,ParamImageType>::cast(reference),false);
                decomposition->SetSplineOrder( SplineOrder );
                decomposition->SetInput( downsampleddeformation );
                decomposition->Update();
                newImages[k]=decomposition->GetOutput();

#endif
            }
        LOG<<VAR(newImages[0]->GetLargestPossibleRegion())<<endl;
        bsplineTransform->SetCoefficientImages(newImages);
        typename itk::TransformFileWriterTemplate<CDisplacementPrecision>::Pointer writer = itk::TransformFileWriterTemplate<CDisplacementPrecision>::New();
        writer->SetFileName("bspline.txt");
        writer->SetInput(bsplineTransform);
        writer->Update();
        return bsplineTransform;
    }

    static DeformationFieldPointerType computeDeformationFieldFromITKBSplineTransform(BSplineTransformPointerType bSplineTransform,ImagePointerType reference){ 
     
        LOGV(5)<<"computing bspline transform parameters"<<std::endl;
        LOGV(6)<<reference->GetLargestPossibleRegion().GetSize()<<std::endl;
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        DeformationFieldPointerType fullDeformationField;
       
        //initialise deformation
        fullDeformationField=DeformationFieldType::New();
        fullDeformationField->SetRegions(reference->GetLargestPossibleRegion());
        fullDeformationField->SetOrigin(reference->GetOrigin());
        fullDeformationField->SetSpacing(reference->GetSpacing());
        fullDeformationField->SetDirection(reference->GetDirection());
        fullDeformationField->Allocate();
      
        
        LabelIterator lIt(fullDeformationField,fullDeformationField->GetLargestPossibleRegion());
        lIt.GoToBegin();
        
        for (;!lIt.IsAtEnd();++lIt){
            IndexType idx=lIt.GetIndex();
            PointType originPoint, deformedPoint;
            reference->TransformIndexToPhysicalPoint(idx,originPoint);
            deformedPoint=bSplineTransform->TransformPoint(originPoint);
            DisplacementType deformation=deformedPoint-originPoint;
            lIt.Set(deformation);
        }
        LOGV(6)<<"Finshed extrapolation"<<std::endl;
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
    static      ImagePointerType warpImageWithReference(ImagePointerType image, DeformationFieldPointerType deformation, ImagePointerType reference, bool nnInterpol=false){
        typedef typename itk::WarpImageFilter<ImageType,ImageType,DeformationFieldType>     WarperType;
        typedef typename WarperType::Pointer     WarperPointer;
        WarperPointer warper=WarperType::New();
        if (nnInterpol){
            NNInterpolatorPointerType nnInt=NNInterpolatorType::New();
            warper->SetInterpolator(nnInt);
        }
        warper->SetInput( image);
        warper->SetEdgePaddingValue(FilterUtils<ImageType>::getMin(image));
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
        warper->SetEdgePaddingValue(FilterUtils<ImageType>::getMin(image));
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
        logSetStage("warping image");
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
        PixelType fillVal= FilterUtils<ImageType>::getMin(image);
        for (maskIt.GoToBegin(),imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt,++maskIt){
            IndexType index=deformationIt.GetIndex();
            typename ImageInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();

            PointType p;
            deformed->TransformIndexToPhysicalPoint(index,p);
            //            LOG<<VAR(p)<<" "<<VAR(displacement)<<endl;

            for (int i=0;i<p.Dimension;i++) {
              p[i] += displacement[i];
            }
            
            image->TransformPhysicalPointToContinuousIndex(p,idx);

            bool inside=true;
            if (nnInterpol){
                if (nnInt->IsInsideBuffer(idx)){
                    //imageIt.Set(nnInt->EvaluateAtContinuousIndex(idx));
                    imageIt.Set(nnInt->Evaluate(p));
                }else inside=false;
            }else{
                if (interpolator->IsInsideBuffer(idx)){
                    //imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
                    imageIt.Set(interpolator->Evaluate(p));
                }else inside=false;
            }
            if (!inside){
                imageIt.Set(fillVal);
                maskIt.Set(0);
            }else{
                maskIt.Set(1);
            }
        }
        pair<ImagePointerType,ImagePointerType> result=std::make_pair(deformed,mask);
        LOGV(10)<<VAR(image->GetLargestPossibleRegion().GetSize())<<" "<<deformation->GetLargestPossibleRegion().GetSize()<<" "<<deformed->GetLargestPossibleRegion().GetSize()<<endl;
        logResetStage;
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
 
    static    ImagePointerType deformImage(ImagePointerType movingImage, ImagePointerType fixedImage,BSplineTransformPointerType deformation){
        typedef typename itk::ResampleImageFilter<
            ImageType,ImageType >    ResampleFilterType;
        
        typename ResampleFilterType::Pointer resample = ResampleFilterType::New();
        
        resample->SetTransform( deformation );
        resample->SetInput( movingImage );
        resample->SetSize(    fixedImage->GetLargestPossibleRegion().GetSize() );
        resample->SetOutputOrigin(  fixedImage->GetOrigin() );
        resample->SetOutputSpacing( fixedImage->GetSpacing() );
        resample->SetOutputDirection( fixedImage->GetDirection() );
          
        resample->Update();
          
        return resample->GetOutput();
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
            
            for (int i = 0; i < p.Dimension; i++) {
              p[i] += displacement[i];
            }
            //leftField->TransformPhysicalPointToContinuousIndex(p,idx);
            //imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx)+displacement);
            DisplacementType newVec = interpolator->Evaluate(p);
            for (int i = 0; i < newVec.Dimension; i++) {
              newVec[i] += displacement[i];
            }
            imageIt.Set(newVec);
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
            double tmp2=0.0;
            for (unsigned int d=0;d<D;++d){
                double tmp=pow(fabs(t[d]),exp);
                tmp2+=tmp;
            }
            norm+=pow(tmp2,1.0/exp);
            ++count;
        }
        //return sqrt(norm)/count;
        return (norm)/count;
    }
    
    static double computeDeformationNormMask(DeformationFieldPointerType def, ImagePointerType mask, double exp=2, bool verbose=false){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
        ImageIterator imageIt(mask,def->GetLargestPossibleRegion());
        
        double norm=0.0;
        int count=0;
        double maxErr=-1.0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt){
            DisplacementType t=deformationIt.Get();
            if (verbose) LOGV(8)<<deformationIt.GetIndex()<<" "<<t<<" "<<imageIt.Get()<<endl;
            if (imageIt.Get()){
                double tmp2=0.0;
            
                for (unsigned int d=0;d<D;++d){
                    double tmp=pow(fabs(t[d]),exp);
                    tmp2+=tmp;
                }
                tmp2=pow(tmp2,1.0/exp);
                norm+=tmp2;
                maxErr=tmp2>maxErr?tmp2:maxErr;
                ++count;
            }
        }
        return norm/count;
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

    static FloatImagePointerType computeLocalDeformationNormWeights(DeformationFieldPointerType def, double sigma=1.0, double * averageNorm=NULL, ImagePointerType mask=NULL){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<FloatImageType> ImageIterator;

        FloatImagePointerType normImage=createEmptyFloat(def);
        double norm=0.0;
        int count=0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        ImageIterator imageIt(normImage,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt){
            DisplacementType t=deformationIt.Get();
            double localNorm=t.GetSquaredNorm();
            imageIt.Set(exp(-localNorm/sigma));
            if (!mask.IsNotNull() || mask->GetPixel(deformationIt.GetIndex())){
                norm+=localNorm;
                ++count;
            }
        }
        if (averageNorm!=NULL){
            (*averageNorm)=norm/count;
        }
        return normImage;
    }

    static FloatImagePointerType computeLocalDeformationNorm(DeformationFieldPointerType def, double sigma=1.0, double * averageNorm=NULL, ImagePointerType mask=NULL){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<FloatImageType> ImageIterator;

        FloatImagePointerType normImage=createEmptyFloat(def);
        double norm=0.0;
        int count=0;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        ImageIterator imageIt(normImage,def->GetLargestPossibleRegion());
        for (deformationIt.GoToBegin(),imageIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++imageIt){
            DisplacementType t=deformationIt.Get();
            double localNorm=t.GetNorm();
            imageIt.Set(localNorm/sigma);
            if (!mask.IsNotNull() || mask->GetPixel(deformationIt.GetIndex())){
                norm+=localNorm;
                ++count;
            }
        }
        if (averageNorm!=NULL){
            (*averageNorm)=norm/count;
        }
        return normImage;
    }
    static DeformationFieldPointerType locallyScaleDeformation(DeformationFieldPointerType def, FloatImagePointerType weights){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        typedef typename  itk::ImageRegionIterator<FloatImageType> ImageIterator;
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
        LOG<<"THIS DOES NOT WORK properly!!!"<<endl;
        //exit(0);
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
        inverter->SetNumberOfIterations(500);
        inverter->Update();
        return inverter->GetOutput();
        
    }

    static DeformationFieldPointerType gaussian(
                                                DeformationFieldPointerType image, float variance
                                                ) {

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();
      
        filter->SetInput(image);
#ifdef DISCRETEGAUSSIAN
        if (variance*variance>32)
            filter->SetMaximumKernelWidth(ceil(variance)*ceil(variance));
        filter->SetVariance(variance*variance);
#else
        filter->SetSigma(variance);
#endif
        filter->Update();

        return filter->GetOutput();
    }

    static DeformationFieldPointerType gaussian(
                                                DeformationFieldPointerType image, SpacingType spacing
                                                ) {

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();

#ifdef DISCRETEGAUSSIAN
        double s[D];
        for (int d=0;d<DeformationFieldType::ImageDimension;++d){
            if (spacing[d]==0){
                spacing[d]=1e-5;
            }
            s[d]=(spacing[d]);
        }
        filter->SetVariance(s);
#else
        for (int d=0;d<DeformationFieldType::ImageDimension;++d){
            if (spacing[d]==0){
                spacing[d]=1e-5;
            }
            spacing[d]=sqrt(spacing[d]);
        }
        filter->SetSigmaArray(spacing);
#endif     
        filter->SetInput(image);

        filter->Update();

        return filter->GetOutput();
    }
    static FloatImagePointerType getJacDets(DeformationFieldPointerType def){
        typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DeformationFieldType,CFloatPrecision> DisplacementFieldJacobianDeterminantFilterType;
        typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
        jacobianFilter->SetInput(def);
        jacobianFilter->SetUseImageSpacingOn();
        jacobianFilter->Update();
        return jacobianFilter->GetOutput();
    }

    static double getMinJacDet(DeformationFieldPointerType def){

        return FilterUtils<FloatImageType>::getMin(getJacobian(def));
        
    }
    static FloatImagePointerType getComponent(DeformationFieldPointerType def, int d){
        FloatImagePointerType result=createEmptyFloat(def);
        typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        DeformationIteratorType defIt(def,def->GetLargestPossibleRegion());
        typedef itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
        FloatImageIteratorType resultIt(result,result->GetLargestPossibleRegion());

        for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
            
            resultIt.Set(defIt.Get()[d]);
        }

        return result;
        
    }
    static void setComponent(DeformationFieldPointerType def, FloatImagePointerType comp, int d){

        typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        DeformationIteratorType defIt(def,def->GetLargestPossibleRegion());
        typedef itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
        FloatImageIteratorType resultIt(comp,comp->GetLargestPossibleRegion());

        for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
            
            DisplacementType disp=defIt.Get();
            disp[d]=resultIt.Get();
            defIt.Set(disp);
        }

        
    }


    static double computeError(DeformationCacheType * cache, DeformationCacheType * m_trueDeformations,  std::vector<string> * m_imageIDList, ImagePointerType mask=NULL){
        //compute inconsistency over triplets
        double averageError=0;
        int count=0;
        int m_numImages=m_imageIDList->size();
        double maxErr=-1.0;
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                DeformationFieldPointerType directDeform;
                if (s!=t){
                    string sourceID=(*m_imageIDList)[s];
                    string targetID=(*m_imageIDList)[t];
                    directDeform= (*cache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]];
                    if (directDeform.IsNotNull()){
                      

                        if ((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]].IsNotNull()){
                            DeformationFieldPointerType trueDeform = (*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]];


                            mask=TransfUtils<ImageType>::createEmptyImage(trueDeform);
                            mask->FillBuffer(0);
                            typename ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
                            IndexType offset;
                            double fraction=0.9;
                            for (int d=0;d<D;++d){
                                offset[d]=(1.0-fraction)/2*size[d];
                                size[d]=fraction*size[d];
                            }
                    
                            typename ImageType::RegionType region;
                            region.SetSize(size);
                            region.SetIndex(offset);
                            LOGV(6)<<VAR(region)<<endl;
                            ImageUtils<ImageType>::setRegion(mask,region,1);


                            directDeform=linearInterpolateDeformationField(directDeform,trueDeform,false);
                            //trueDeform=linearInterpolateDeformationField(trueDeform,directDeform,false);
                            DeformationFieldPointerType diff  = TransfUtils<ImageType>::subtract(directDeform,trueDeform);
                        
                            double residual;
                            if (false && mask.IsNotNull())
                                residual= computeDeformationNormMask(diff,mask,2);
                            else
                                residual= computeDeformationNorm(diff,2);
                            LOGV(3)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(residual)<<endl;
                            averageError += residual;
                            maxErr=residual>maxErr?residual:maxErr;
                            count++;
                        }
                    }
                }
            }
        }
        if (!count)
            return -1;
        else
            return averageError/count;
    }

    static double computeInconsistency(DeformationCacheType * cache,  std::vector<string> * m_imageIDList, DeformationCacheType * trueCache, ImageCacheType * masks=NULL){
        //compute inconsistency over triplets
        double averageInconsistency=0;
        int m_numImages=m_imageIDList->size();
        int count=0;
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    DeformationFieldPointerType directDeform;
                    bool estimatedDeform=false;
                    bool skip=false;
                    directDeform= (*cache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]];
                    if (!directDeform.IsNotNull()){
                        if (trueCache && (*trueCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]].IsNotNull()){
                            directDeform= (*trueCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]];
                        }else{
                            skip = true;
                        }
                    }else{
                        estimatedDeform=true;
                    }
                    if (! skip && s!=t){
                        for (int i=0;i<m_numImages;++i){
                            if (i!=t && i !=s){
                                skip=false;
                                DeformationFieldPointerType d0,d1;
                            
                                d0 = (*cache)[(*m_imageIDList)[s]][(*m_imageIDList)[i]];
                                if (d0.IsNotNull()){
                                    estimatedDeform=true;
                                }else{
                                    if (trueCache && (*trueCache)[(*m_imageIDList)[s]][(*m_imageIDList)[i]].IsNotNull()){
                                        d0 = (*trueCache)[(*m_imageIDList)[s]][(*m_imageIDList)[i]];
                                    }else{
                                        skip=true;
                                    }
                                }
                                d1 = (*cache)[(*m_imageIDList)[i]][(*m_imageIDList)[t]];
                                if (d1.IsNotNull()){
                                    estimatedDeform=true;
                                }else{
                                    if (trueCache && (*trueCache)[(*m_imageIDList)[i]][(*m_imageIDList)[t]].IsNotNull()){
                                        d1 = (*trueCache)[(*m_imageIDList)[i]][(*m_imageIDList)[t]];
                                    }else{
                                        skip = true;
                                    }
                                }

                                if (! skip && estimatedDeform){
                                    ImagePointerType mask;
                                    if (masks == NULL){
                                        mask=TransfUtils<ImageType>::createEmptyImage(directDeform);
                                        mask->FillBuffer(0);
                                        typename ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
                                        IndexType offset;
                                        double fraction=0.9;
                                        for (int d=0;d<D;++d){
                                            offset[d]=(1.0-fraction)/2*size[d];
                                            size[d]=fraction*size[d];
                                        }
                                        
                                        typename ImageType::RegionType region;
                                        region.SetSize(size);
                                        region.SetIndex(offset);
                                        LOGV(6)<<VAR(region)<<endl;
                                        ImageUtils<ImageType>::setRegion(mask,region,1);
                                    }else{
                                        mask=(*masks)[(*m_imageIDList)[t]];
                                        mask=FilterUtils<ImageType>::NNResample(mask,createEmptyImage(d1),false);
                                    }
                            
                                    DeformationFieldPointerType indirectDef = TransfUtils<ImageType>::composeDeformations(d1,d0);
                                    indirectDef=TransfUtils<ImageType>::linearInterpolateDeformationField(indirectDef,TransfUtils<ImageType>::createEmptyImage(directDeform));
                                    DeformationFieldPointerType diff  = TransfUtils<ImageType>::subtract(directDeform,indirectDef);
                                    double residual = TransfUtils<ImageType>::computeDeformationNormMask(diff,mask,2,true);
                                    //double residual = TransfUtils<ImageType>::computeDeformationNorm(diff,2);
                                    averageInconsistency += residual;
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (count)
            return averageInconsistency/count;
        else return -1;
        
    }//computeInconsistency
    

    static double computeTRE(string targetLandmarks, string refLandmarks, 
                             DeformationFieldPointerType def,ImagePointerType reference, 
                             ImagePointerType targetMask=NULL, ImagePointerType referenceMask=NULL){
        typedef typename  ImageType::DirectionType DirectionType;
        typedef typename itk::VectorLinearInterpolateImageFunction<DeformationFieldType> DefInterpolatorType;
        typedef typename DefInterpolatorType::ContinuousIndexType CIndexType;
        PointType p;
        p.Fill(0.0);
        typename DefInterpolatorType::Pointer defInterpol=DefInterpolatorType::New();
        defInterpol->SetInputImage(def);
        ifstream ifs(refLandmarks.c_str());
        int i=0;
        double TRE=0.0;
        int count=0;
        vector<PointType> landmarksReference, landmarksTarget;
        DirectionType refDir=reference->GetDirection();
        DirectionType targetDir=def->GetDirection();

        while ( ! ifs.eof() ) {
            PointType point;
            for (int d=0;d<D;++d){
                ifs>>point[d];
                point[d]=point[d]*refDir[d][d];
            }
            //LOG<<point<<endl;
            landmarksReference.push_back(point);
            
        } 
        //std::cout<<"read "<<landmarksReference.size()<<" landmarks"<<std::endl;
        ifstream ifs2(targetLandmarks.c_str());
        i=0;
        for (;i<landmarksReference.size()-1;++i){
            PointType pointTarget;
            for (int d=0;d<D;++d){
                ifs2>>pointTarget[d];
                pointTarget[d]=pointTarget[d]*targetDir[d][d];
            }        
            IndexType indexTarget,indexReference;
            def->TransformPhysicalPointToIndex(pointTarget,indexTarget);
            if (targetMask.IsNotNull()){
                if (!targetMask->GetPixel(indexTarget)){
                    LOGV(3)<<VAR(pointTarget)<<" not in mask at index "<<VAR(indexTarget)<<endl;
                    continue;
                }
            }
            PointType deformedReferencePoint;
            reference->TransformPhysicalPointToIndex(landmarksReference[i],indexReference);
            if (referenceMask.IsNotNull()){
                if (!referenceMask->GetPixel(indexReference)){
                    LOGV(3)<<VAR(landmarksReference[i])<<" not in mask at index "<<VAR(indexReference)<<endl;
                    continue;
                }
            }
          
            CIndexType cindex;
            def->TransformPhysicalPointToContinuousIndex(pointTarget,cindex);
            if (def->GetLargestPossibleRegion().IsInside(cindex)){
                deformedReferencePoint= pointTarget+defInterpol->EvaluateAtContinuousIndex(cindex);
                double localError=(deformedReferencePoint - landmarksReference[i]).GetNorm();
                LOGI(2,std::cout<<"pt"<<i<<": "<<(localError)<<" ");
                TRE+=localError;
                ++count;
            }
        }
        LOGI(2,std::cout<<std::endl);
        return TRE/count;
    }

    static ImagePointerType translateImage(ImagePointerType img, DisplacementType disp, bool NN=false,ImagePointerType target=NULL){

        typedef typename itk::TranslationTransform<double,D> TranslationTransformType;
        typename TranslationTransformType::Pointer transform =
            TranslationTransformType::New();
        typename TranslationTransformType::OutputVectorType translation;
        for (int d=0;d<D;++d){
            translation[d]=disp[d];
        }
        transform->Translate(translation);
        
        typedef typename itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
        resampleFilter->SetTransform(transform);
        if (target.IsNull()) target=img;
        resampleFilter->SetInput(target);
        resampleFilter->SetOutputOrigin(target->GetOrigin());
		resampleFilter->SetOutputSpacing ( target->GetSpacing() );
		resampleFilter->SetOutputDirection ( target->GetDirection() );
		resampleFilter->SetSize ( target->GetLargestPossibleRegion().GetSize() );
        NNInterpolatorPointerType interpol2=NNInterpolatorType::New();
        LinearInterpolatorPointerType interpol=LinearInterpolatorType::New();
        if (NN){
            resampleFilter->SetInterpolator(interpol2);
        }else{
            resampleFilter->SetInterpolator(interpol);
        }
        
        resampleFilter->Update();
        return resampleFilter->GetOutput();

    }
    static FloatImagePointerType computeDirectedGradient(DeformationFieldPointerType def, int d){
        typedef typename  itk::ImageRegionIterator<DeformationFieldType> LabelIterator;
        LabelIterator deformationIt(def,def->GetLargestPossibleRegion());
        FloatImagePointerType result=createEmptyFloat(def);
        typedef itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
        FloatImageIteratorType resIt(result,result->GetLargestPossibleRegion());
        resIt.GoToBegin();
        typename DeformationFieldType::SizeType size=result->GetLargestPossibleRegion().GetSize();
        double space=2*def->GetSpacing()[d];
        for (deformationIt.GoToBegin();!deformationIt.IsAtEnd();++deformationIt,++resIt){
            IndexType idx=deformationIt.GetIndex();
            double gradient=0;
            if (idx[d]>0 && idx[d]<size[d]-1){
                IndexType rIdx=idx,lIdx=idx;
                rIdx[d]+=1;
                lIdx[d]-=1;
                gradient=def->GetPixel(rIdx)[d]-def->GetPixel(lIdx)[d];
                gradient/=space;
            }
            resIt.Set(gradient);
            
        }
        //return sqrt(norm)/count;
        return result;
    }
};
