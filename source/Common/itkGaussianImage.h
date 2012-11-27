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
	typedef typename ImageType::IndexType IndexType;
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
    FloatImagePointerType m_localWeights;
    int count;
    bool m_useVariance;
public:
    GaussianEstimatorVectorImage(){
        m_localWeights=NULL;
        m_useVariance=false;
    }
    void initialize(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        count=1;
        if (weights.IsNotNull()){
            m_localWeights=ImageUtils<FloatImageType>::duplicate(weights);
            img=TransfUtils<ImageType>::locallyScaleDeformation(img,weights);
        }
        m_variance=TransfUtils<ImageType>::multiplyOutOfPlace(img,img);
        m_mean=ImageUtils<DeformationFieldType>::duplicate(img);
    }
    void addImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        
            if (weights.IsNotNull()){
                m_localWeights=FilterUtils<FloatImageType>::add(m_localWeights,weights);
                img=TransfUtils<ImageType>::locallyScaleDeformation(img,weights);
            }
            m_mean=TransfUtils<ImageType>::add(m_mean,img);
            m_variance=TransfUtils<ImageType>::add(m_variance,TransfUtils<ImageType>::multiplyOutOfPlace(img,img));
           
            count++;
       
    }
    void finalize(){
        if (m_localWeights.IsNotNull()){
            TransfUtils<FloatImageType>::divide(m_mean,m_localWeights);
            TransfUtils<FloatImageType>::divide(m_variance,m_localWeights);
        }else{
            m_mean=TransfUtils<ImageType>::multiplyOutOfPlace(m_mean,1.0/count);
            m_variance=TransfUtils<ImageType>::multiplyOutOfPlace(m_variance,1.0/count);
        }
        m_variance=TransfUtils<ImageType>::subtract(m_variance,TransfUtils<ImageType>::multiplyOutOfPlace(m_mean,m_mean));
    }

    DeformationFieldPointerType getMean(){return m_mean;}
    DeformationFieldPointerType getVariance(){return m_variance;}
    DeformationFieldPointerType getStdDev(){return TransfUtils<ImageType>::localSqrt(m_variance);}

    FloatImagePointerType getLikelihood(DeformationFieldPointerType img, double s=1.0){
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
            IndexType idx=varIt.GetIndex();
            for (int d=0;d<D;++d){
                if (m_useVariance)
                    sum+=pow(x[d]-mean[d],2)/var[d];
                else{
                    double tmp=x[d]-mean[d];
                    tmp*=tmp;
                    sum+=tmp;
                }
                varProd*=var[d];
            }
            if (varProd>0.0){
                if (m_useVariance)
                    resIt.Set(1.0/sqrt(pow(2.0*3.1425,D)*varProd) * exp(-0.5*sum));
                else
                    resIt.Set(exp(-0.5*sum/s));
                if (resIt.Get()<0.0){
                    LOGV(2)<<VAR(idx)<<" "<<VAR(resIt.Get())<<endl;
                }
            }
            else
                resIt.Set(0.0);
        }
        return result;

    }
    
};//class
template<class ImageType>
class FastGaussianEstimatorVectorImage{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename TransfUtils<ImageType>::DisplacementType DeformationType;
    typedef typename TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef typename itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationImageIteratorType;
    static const int D=ImageType::ImageDimension;

private:
    DeformationFieldPointerType m_mean,m_variance;
    FloatImagePointerType m_localWeights;
    ImagePointerType m_localCounts;
    int count;
    bool m_useVariance;
public:
    FastGaussianEstimatorVectorImage(){
        m_localWeights=NULL;
        m_useVariance=false;
        m_localCounts=NULL;
    }
    void initialize(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        count=1;
        if (weights.IsNotNull()){
            m_localWeights=ImageUtils<FloatImageType>::duplicate(weights);
            img=TransfUtils<ImageType>::locallyScaleDeformation(img,weights);
        }
        m_variance=TransfUtils<ImageType>::multiplyOutOfPlace(img,img);
        m_mean=ImageUtils<DeformationFieldType>::duplicate(img);
    }
 
    void addImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
       
        DeformationImageIteratorType meanIt(m_mean,m_mean->GetLargestPossibleRegion());
        DeformationImageIteratorType varIt(m_variance,m_mean->GetLargestPossibleRegion());
        DeformationImageIteratorType inputIt(img,m_mean->GetLargestPossibleRegion());
        meanIt.GoToBegin();        varIt.GoToBegin();        inputIt.GoToBegin();
        
        FloatImageIteratorType weightIt,weightAccuIt;
        if (weights.IsNotNull()){
            weightIt=FloatImageIteratorType(weights,weights->GetLargestPossibleRegion());
            weightIt.GoToBegin();
            weightAccuIt=FloatImageIteratorType(m_localWeights,weights->GetLargestPossibleRegion());
            weightAccuIt.GoToBegin();
        }

        for (;!meanIt.IsAtEnd();++meanIt,++varIt,++inputIt){
            DeformationType def=inputIt.Get();
            if (weights.IsNotNull()){
                weightAccuIt.Set(weightAccuIt.Get()+weightIt.Get());
                def=def*weightIt.Get();
                ++weightIt;
                ++weightAccuIt;
            }

            meanIt.Set(meanIt.Get()+def);
            for (int d=0;d<D;++d){
                def[d]*=def[d];
            }
            varIt.Set(varIt.Get()+def);
        }    
        count++;
        
    }
    void finalize(){

        DeformationImageIteratorType meanIt(m_mean,m_mean->GetLargestPossibleRegion());
        DeformationImageIteratorType varIt(m_variance,m_mean->GetLargestPossibleRegion());
        meanIt.GoToBegin();        varIt.GoToBegin();      
        
        FloatImageIteratorType weightIt,weightAccuIt;
        if (m_localWeights.IsNotNull()){
            weightAccuIt=FloatImageIteratorType(m_localWeights,m_localWeights->GetLargestPossibleRegion());
            weightAccuIt.GoToBegin();
        }

        for (;!meanIt.IsAtEnd();++meanIt,++varIt){
            DeformationType meanDef=meanIt.Get();
            DeformationType varDef=varIt.Get();
            float divisor=count;
            if (m_localWeights.IsNotNull()){
                divisor=weightAccuIt.Get();
                ++weightAccuIt;
            }
            if (divisor){
                meanIt.Set(meanDef/divisor);
                meanDef=meanDef/count;
                for (int d=0;d<D;++d){
                    meanDef[d]*=meanDef[d];
                }

                varDef=(varDef/count - meanDef)*count/divisor;
            }else{
                LOGV(3)<<VAR(meanDef)<<" "<<VAR(varDef)<<endl;
                meanDef.Fill(0.0);
                meanIt.Set(meanDef);
                varDef.Fill(0.0);
                
            }
            varIt.Set(varDef);

        }

    }

    DeformationFieldPointerType getMean(){return m_mean;}
    DeformationFieldPointerType getVariance(){return m_variance;}
    DeformationFieldPointerType getStdDev(){return TransfUtils<ImageType>::localSqrt(m_variance);}

    FloatImagePointerType getLikelihood(DeformationFieldPointerType img, double s=1.0){
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
            IndexType idx=varIt.GetIndex();
            for (int d=0;d<D;++d){
                if (m_useVariance){
                    bool ff=false;
                    if (var[d]<0.0){
                        LOG<<"bad. "<<VAR(var[d])<<endl;
                        ff=true;
                    }
                    if (var[d]>0.0){
                        double tmp=x[d]-mean[d];
                        tmp*=tmp;
                        sum+=tmp/var[d];
                        if (ff){
                            LOG<<"wtf "<<VAR(tmp)<<" "<<VAR(sum)<<endl;
                        }
                    }
                    if (var[d]<0.0){
                        LOG<<"bad2. "<<VAR(var[d])<<endl;
                        ff=true;
                    }
                }
                else{
                    double tmp=x[d]-mean[d];
                    tmp*=tmp;
                    sum+=tmp;
                }
                varProd*=var[d];
            }
            if (varProd>0.0){
                if (m_useVariance){
                    double result=1.0/sqrt(pow(2.0*3.1425,D)*varProd) * exp(-0.5*sum);
                    if (result>1.0){
                        LOGV(3)<<VAR(mean)<<" "<<VAR(x)<<" "<<VAR(var)<<endl;
                        result=1.0;
                    }
                    resIt.Set(result);
                }
                else
                    resIt.Set(exp(-0.5*sum/s));
                if (resIt.Get()<0.0){
                    LOGV(2)<<VAR(idx)<<" "<<VAR(resIt.Get())<<endl;
                }
            }
            else{
                resIt.Set(0.0);
                LOGV(2)<<VAR(idx)<<" "<<VAR(mean)<<" "<<VAR(var)<<" " << VAR(x)<<endl;
            }
        }
        return result;

    }
    
};//class

template<class ImageType>
class LocalVectorMeanShift{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename TransfUtils<ImageType>::DisplacementType DeformationType;
    typedef typename TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef typename itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationImageIteratorType;
    static const int D=ImageType::ImageDimension;

private:
    DeformationFieldPointerType m_mean,m_oldMean;
    FloatImagePointerType m_localWeights;
    ImagePointerType m_localCounts;
    int count;
    bool m_useVariance;
    double m_sigma;
public:
    FastGaussianEstimatorVectorImage(){
        m_localWeights=NULL;
        m_useVariance=false;
        m_localCounts=NULL;
    }
    void setSigma(double s){m_sigma=s;}
    void initialize(DeformationFieldPointerType img){
        count=1;
        m_oldMean=ImageUtils<DeformationFieldType>::duplicate(img);
        m_mean=TransfUtils<ImageType>::createEmpty(img);
    }
 
    void addImage(DeformationFieldPointerType img){
       
        DeformationImageIteratorType meanIt(m_mean,m_mean->GetLargestPossibleRegion());
        DeformationImageIteratorType oldIt(m_oldMean,m_oldMean->GetLargestPossibleRegion());
        DeformationImageIteratorType inputIt(img,m_mean->GetLargestPossibleRegion());
        meanIt.GoToBegin();        oldIt.GoToBegin();        inputIt.GoToBegin();
        
        FloatImageIteratorType weightAccuIt;
        weightAccuIt=FloatImageIteratorType(m_localWeights,img->GetLargestPossibleRegion());
        weightAccuIt.GoToBegin();
        

        for (;!meanIt.IsAtEnd();++meanIt,++oldIt,++inputIt,++weightAccuIt){
            DeformationType def=inputIt.Get();
            DeformationType oldMeanDef=oldIt.Get();

            double kernelValue=(def-oldMeanDef).GetSquaredNorm()/m_sigma;
            weightAccuIt.Set(weightAccuIt.Get()+kernelValue);
            def=def*kernelValue;
            meanIt.Set(meanIt.Get()+def);
        }    
        count++;
        
    }
    void finalize(){

        DeformationImageIteratorType meanIt(m_mean,m_mean->GetLargestPossibleRegion());
        meanIt.GoToBegin();       
        
        FloatImageIteratorType weightAccuIt;
        weightAccuIt=FloatImageIteratorType(m_localWeights,m_localWeights->GetLargestPossibleRegion());
        weightAccuIt.GoToBegin();
        

        for (;!meanIt.IsAtEnd();++meanIt,++weightAccuIt){
            DeformationType meanDef=meanIt.Get();
            float divisor;
            divisor=weightAccuIt.Get();
            if (divisor){
                meanIt.Set(meanDef/divisor);
            }else{
                LOGV(3)<<VAR(meanDef)<<" "<<VAR(varDef)<<endl;
                meanDef.Fill(0.0);
                meanIt.Set(meanDef);
            }
        }

        m_oldMean=m_mean;
        m_mean=TransfUtils<ImageType>::createEmpty(m_mean);

    }

    DeformationFieldPointerType getMean(){return m_oldMean;}
  

    
};//class
