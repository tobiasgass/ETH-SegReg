#pragma once


#include "itkImage.h"
#include "Log.h"
#include "ImageUtils.h"
#include "itkAffineTransform.h"
#include <iostream>

#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"
#include "itkImageRegionIteratorWithIndex.h"

 
template<class ImageType>
class TransfUtils {

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename itk::AffineTransform<float,ImageType::ImageDimension> AffineTransformType;
    typedef typename AffineTransformType::Pointer AffineTransformPointerType;
    static const int D=ImageType::ImageDimension;
    typedef itk::Vector<float,D> DisplacementType;
    typedef itk::Image<DisplacementType,D> DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typename ImageType::PointType PointType;
    typename ImageType::IndexType IndexType;
    typename ImageType::ContinuousIndexType ContinuousIndexType;
public:

    static DeformationFieldPointerType affineToDisplacementField(AffineTransformPointerType affine, ImagePointerType targetImage){
        DeformationFieldPointerType deformation=DeformationFieldType::New();
        deformation->SetRegions(targetImage->GetLargestPossibleRegion());
        deformation->SetOrigin(targetImage->GetOrigin());
        deformation->SetSpacing(targetImage->GetSpacing());
        deformation->SetDirection(targetImage->GetDirection());
        deformation->Allocate();
        typedef itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
        typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        ImageIteratorType imIt(targetImage,targetImage->GetLargestPossibleRegion());
        DeformationIteratorType defIt(deformation,deformation->GetLargestPossibleRegion());
        for (imIt.GoToBegin(),defIt.GoToBegin();!imIt.IsAtEnd();++defIt, ++imIt){
            
            IndexType idx=imIt.GetIndex();
            PointType p,p2;
            
            targetImage->TransformIndexToPhysicalPoint(idx,p);
            p2=affine->TransformPoint(p);
            ContinuousIndexType idx2;
            targetImage->TransformPhysicalPointToIndex(p,idx2);
            DisplacementType disp;
            for (int d=0;d<D;++d){
                disp[d]=idx2[d]-idx[d];
            }
            defIt.Set(disp);
        }
                               
        return deformation;
        
    }

    

    static AffineTransformPointerType readAffine(string filename){
        // Register default transforms
        itk::TransformFactoryBase::RegisterDefaultTransforms();
        
        itk::TransformFileReader::Pointer reader = itk::TransformFileReader::New();
        reader->SetFileName(filename);
        try{
            reader->Update();
        }catch( itk::ExceptionObject & err ){
            LOG<<"could not read affine transform from " <<filename<<std::endl;
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
};
