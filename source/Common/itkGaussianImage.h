#pragma once


#include <limits.h>
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"

template<class ImageType>
class GaussianEstimatorScalarImage{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;

private:
    ImagePointerType m_mean,m_variance;
    int count;
public:
    void addImage(ImagePointerType img){
        if (!m_mean.IsNotNull()){
            count=1;
            m_mean=ImageUtils<ImageType>::duplicate(img);
            m_variance=ImageUtils<ImageType>::multiplyOutOfPlace(img,img);
        }else{
            m_mean=ImageUtils<ImageType>::add(m_mean,img);
            m_variance=ImageUtils<ImageType>::add(m_mean,ImageUtils<ImageType>::multiplyOutOfPlace(img,img));
            count++;
        }
    }
    void finalize(){
        ImageUtils<ImageType>::multiply(m_mean,1.0/count);
        ImageUtils<ImageType>::multiply(m_variance,1.0/count);
        m_variance=ImageUtils<ImageType>::subtract(m_variance,ImageUtils<ImageType>::multiplyOutOfPlace(m_mean,m_mean));
    }
    ImagePointerType getMean(){return m_mean;}
    ImagePointerType getVariance(){return m_variance;}
    
};//class

template<class ImageType>
class GaussianEstimatorVectorImage{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename TransfUtils<ImageType>::DisplacementType DeformationType;
    typedef typename TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationImageIteratorType;
    static const int D=ImageType::ImageDimension;
private:
    DeformationFieldPointerType m_mean,m_variance;
    int count;
public:
    void addImage(DeformationFieldPointerType img){
        if (!m_mean.IsNotNull()){
            count=1;
            m_mean=ImageUtils<DeformationFieldType>::duplicate(img);
            m_variance=TransfUtils<ImageType>::multiplyOutOfPlace(img,img);
        }else{
            m_mean=TransfUtils<ImageType>::add(m_mean,img);
            m_variance=TransfUtils<ImageType>::add(m_variance,TransfUtils<ImageType>::multiplyOutOfPlace(img,img));
            count++;
        }
    }
    void finalize(){
        m_mean=TransfUtils<ImageType>::multiplyOutOfPlace(m_mean,1.0/count);
        m_variance=TransfUtils<ImageType>::multiplyOutOfPlace(m_variance,1.0/count);
        m_variance=TransfUtils<ImageType>::subtract(m_variance,TransfUtils<ImageType>::multiplyOutOfPlace(m_mean,m_mean));

    }
    DeformationFieldPointerType getMean(){return m_mean;}
    DeformationFieldPointerType getVariance(){return m_variance;}
    DeformationFieldPointerType getStdDev(){return TransfUtils<ImageType>::localSqrt(m_variance);}

    FloatImagePointerType getLikelihood(DeformationFieldPointerType img){
        FloatImagePointerType result=TransfUtils<FloatImageType>::createEmptyFloat(img);
        FloatImageIteratorType resIt(result,result->GetLargestPossibleRegion());
        DeformationImageIteratorType defIt(img,img->GetLargestPossibleRegion());
        DeformationImageIteratorType meanIt(m_mean,img->GetLargestPossibleRegion());
        DeformationImageIteratorType varIt(m_variance,img->GetLargestPossibleRegion());
        resIt.GoToBegin();
        defIt.GoToBegin();
        for (;!resIt.IsAtEnd();++resIt,++defIt,++meanIt,++varIt){
            double sum=0.0;
            double varProd=1.0;
            DeformationType mean,var,x;
            mean=meanIt.Get();
            var=varIt.Get();
            x=defIt.Get();
            for (int d=0;d<D;++d){
                sum+=pow(x[d]-mean[d],2)/var[d];
                varProd*=var[d];
            }
            resIt.Set(1.0/sqrt(pow(2.0*3.1425,D)*varProd) * exp(-0.5*sum));
        }
        return result;

    }
    
};//class
