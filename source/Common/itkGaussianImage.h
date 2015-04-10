#pragma once


#include <limits.h>
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"

template<typename ImageType,typename FloatPrecision=float>
class GaussianEstimatorScalarImage{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    static const int D=ImageType::ImageDimension;
    typedef itk::Image<FloatPrecision,D> FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;


protected:
    FloatImagePointerType m_mean,m_variance;
    int count;
    bool finalized;
public:
    GaussianEstimatorScalarImage(){
        m_mean=NULL;
        finalized=false;
        count=0;
    }
    virtual void addImage(ImagePointerType img){
        FloatImagePointerType floatImg=FilterUtils<ImageType,FloatImageType>::cast(img);
        if (!m_mean.IsNotNull()){
            count=1;
            m_mean=floatImg;
            m_variance=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(floatImg,floatImg);
        }else{
            floatImg=FilterUtils<FloatImageType>::LinearResample(floatImg,m_mean,false);
            m_mean=FilterUtils<FloatImageType>::add(m_mean,floatImg);
            m_variance=FilterUtils<FloatImageType>::add(m_variance,ImageUtils<FloatImageType>::multiplyImageOutOfPlace(floatImg,floatImg));
            count++;
        }
    }
    virtual void finalize(){
        if (!finalized){
            if (count == 0){
                LOG<<"no images to compute statistics of..." <<endl;
                return;
            }
            LOGV(6)<<VAR(count)<<endl;
            ImageUtils<FloatImageType>::multiplyImage(m_mean,1.0/count);

            FloatImagePointerType squaredMean=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(m_mean,m_mean);
            //            ImageUtils<FloatImageType>::multiplyImage(squaredMean,1.0/count);

            LOGI(6,ImageUtils<FloatImageType>::writeImage("squaredMean.nii",squaredMean));
            LOGI(6,ImageUtils<FloatImageType>::writeImage("preVariance.nii",m_variance));
            
            LOGI(6,ImageUtils<FloatImageType>::writeImage("mean.nii",m_mean));
            if (count>1){
                ImageUtils<FloatImageType>::multiplyImage(m_variance,1.0/(count-1));
            }else{
                LOGV(2)<<"Warning, only one observation in gauss estimator, variance estimator will not be usefull"<<endl;
            }          
            m_variance=FilterUtils<FloatImageType>::substract(m_variance,squaredMean);
          
            LOGI(6,ImageUtils<FloatImageType>::writeImage("variance.nii",m_variance));
            finalized = true; 
        }
    }
    ImagePointerType getMean(){
        if (! finalized){
            LOG<<"ESTIMATOR NOT FINALIZED " << endl;
        }
        return FilterUtils<FloatImageType,ImageType>::cast(m_mean);
    }
    FloatImagePointerType getFloatMean(){
          if (! finalized){
            LOG<<"ESTIMATOR NOT FINALIZED " << endl;
        }
        return m_mean;
    }
    ImagePointerType getVariance(){return FilterUtils<FloatImageType,ImageType>::cast(m_variance);}
    FloatImagePointerType getFloatVariance(){return (m_variance);}

    
};//class

template<typename ImageType>
class MinEstimatorScalarImage: public GaussianEstimatorScalarImage<ImageType>{
public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    virtual void addImage(ImagePointerType img){
        
        if (!this->m_mean.IsNotNull()){
            this->count=1;
            this->m_mean=ImageUtils<ImageType>::duplicate(img);
        }else{
            FilterUtils<ImageType>::localMin(this->m_mean,img);
            this->count++;
        }
    }
 virtual void finalize(){
        if (!this->finalized){
            this->finalized = true; 
        }
 }
};

template<typename ImageType,class FloatPrecision=float>
class GaussianEstimatorVectorImage{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    typedef TransfUtils<ImageType, float,double,FloatPrecision> TransfUtilsType;
    typedef typename TransfUtilsType::DisplacementType DeformationType;
    typedef typename TransfUtilsType::DeformationFieldType DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType,FloatPrecision>::FloatImageType FloatImageType;
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
        count = 0;
    }
    void initialize(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        count=1;
        if (weights.IsNotNull()){
            m_localWeights=ImageUtils<FloatImageType>::duplicate(weights);
            img=TransfUtilsType::locallyScaleDeformation(img,weights);
        }
        m_variance=TransfUtilsType::multiplyOutOfPlace(img,img);
        m_mean=ImageUtils<DeformationFieldType>::duplicate(img);
    }
    void addImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        if (count == 0){
            count=1;
            if (weights.IsNotNull()){
                m_localWeights=ImageUtils<FloatImageType>::duplicate(weights);
                img=TransfUtilsType::locallyScaleDeformation(img,weights);
            }
            m_variance=TransfUtilsType::multiplyOutOfPlace(img,img);
            m_mean=ImageUtils<DeformationFieldType>::duplicate(img);
        }else{
            if (weights.IsNotNull()){
                m_localWeights=FilterUtils<FloatImageType>::add(m_localWeights,weights);
                img=TransfUtilsType::locallyScaleDeformation(img,weights);
            }
            m_mean=TransfUtilsType::add(m_mean,img);
            m_variance=TransfUtilsType::add(m_variance,TransfUtilsType::multiplyOutOfPlace(img,img));
            count++;
        }
       
    }
    void finalize(){
        if (m_localWeights.IsNotNull()){
            TransfUtils<FloatImageType>::divide(m_mean,m_localWeights);
            TransfUtils<FloatImageType>::divide(m_variance,m_localWeights);
        }else{
            m_mean=TransfUtilsType::multiplyOutOfPlace(m_mean,1.0/count);
            m_variance=TransfUtilsType::multiplyOutOfPlace(m_variance,1.0/(count-1));
        }
        m_variance=TransfUtilsType::subtract(m_variance,TransfUtilsType::multiplyOutOfPlace(m_mean,m_mean));
    }

    DeformationFieldPointerType getMean(){return m_mean;}
    DeformationFieldPointerType getVariance(){return m_variance;}
    DeformationFieldPointerType getStdDev(){return TransfUtilsType::localSqrt(m_variance);}

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
template<typename ImageType>
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

protected:
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
  virtual   void initialize(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        count=1;
        if (weights.IsNotNull()){
            m_localWeights=ImageUtils<FloatImageType>::duplicate(weights);
            img=TransfUtils<ImageType>::locallyScaleDeformation(img,weights);
        }
        m_variance=TransfUtils<ImageType>::multiplyOutOfPlace(img,img);
        m_mean=ImageUtils<DeformationFieldType>::duplicate(img);
    }
 
    virtual void addImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
       
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
                if (weightIt==weightIt.Begin()){
                    //LOG<<VAR(def)<<" "<<VAR(weightIt.Get())<<endl;
                    
                }
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
    virtual void finalize(){

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
                //LOGV(3)<<VAR(meanDef)<<" "<<VAR(varDef)<<endl;
                meanDef.Fill(0.0);
                meanIt.Set(meanDef);
                varDef.Fill(0.0);
                
            }
            varIt.Set(varDef);

        }

    }

    virtual DeformationFieldPointerType getMean(){return m_mean;}
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

template<typename ImageType>
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
    DeformationFieldPointerType m_mean,m_oldMean,m_localSigma;;
    FloatImagePointerType m_localWeights;
    ImagePointerType m_localCounts;
    int count;
    bool m_useVariance;
    double m_sigma;
public:
    LocalVectorMeanShift(){
        m_localWeights=NULL;
        m_useVariance=false;
        m_localCounts=NULL;
    }
    void setSigma(double s){m_sigma=s;}
    void initialize(DeformationFieldPointerType img,DeformationFieldPointerType localBandwidths=NULL){
        count=1;
        m_oldMean=ImageUtils<DeformationFieldType>::duplicate(img);
        m_mean=TransfUtils<ImageType>::createEmpty(img);
        m_localWeights=TransfUtils<FloatImageType>::createEmptyImage(img);
        m_localSigma=localBandwidths;
    }
 
    void addImage(DeformationFieldPointerType img){
       
        DeformationImageIteratorType meanIt(m_mean,m_mean->GetLargestPossibleRegion());
        DeformationImageIteratorType oldIt(m_oldMean,m_oldMean->GetLargestPossibleRegion());
        DeformationImageIteratorType inputIt(img,m_mean->GetLargestPossibleRegion());
        meanIt.GoToBegin();        oldIt.GoToBegin();        inputIt.GoToBegin();
        
        FloatImageIteratorType weightAccuIt;
        DeformationImageIteratorType bandWidthIt;
        weightAccuIt=FloatImageIteratorType(m_localWeights,img->GetLargestPossibleRegion());
        weightAccuIt.GoToBegin();
        if (m_localSigma.IsNotNull()){
            bandWidthIt=DeformationImageIteratorType(m_localSigma,m_localSigma->GetLargestPossibleRegion());
        }
        for (;!meanIt.IsAtEnd();++meanIt,++oldIt,++inputIt,++weightAccuIt){
            DeformationType def=inputIt.Get();
            DeformationType oldMeanDef=oldIt.Get();
            DeformationType sigma;
            double divisor=m_sigma;
            if (m_localSigma.IsNotNull()){
                sigma=bandWidthIt.Get();
                ++bandWidthIt;
            }
            double kernelValue;
            for (int d=0;d<D;++d){
                if (m_localSigma.IsNotNull()){
                    if (sigma[d]>0.0){
                        kernelValue+=pow(def[d]-oldMeanDef[d],2.0)/(sigma[d]*m_sigma);
                    }else{
                        kernelValue=100.0;
                        break;
                    }
                }else{
                    kernelValue+=pow(def[d]-oldMeanDef[d],2.0)/divisor;
                }
            }
            kernelValue=exp(-0.5*kernelValue);

            LOGV(5)<<VAR(kernelValue)<<" "<<VAR(def)<<" "<<VAR(oldMeanDef)<<endl;
            weightAccuIt.Set(weightAccuIt.Get()+kernelValue);
            def=def*kernelValue;
            meanIt.Set(meanIt.Get()+def);
        }    
        count++;
        
    }
    bool finalize(double convergenceCriterion=0.1){

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
                LOGV(3)<<VAR(meanDef)<<endl;
                meanDef.Fill(0.0);
                meanIt.Set(meanDef);
            }
        }
        double diffNorm=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(m_oldMean,m_mean),1.0);
        LOGV(2)<<VAR(diffNorm)<<endl;
        m_oldMean=m_mean;
        m_mean=TransfUtils<ImageType>::createEmpty(m_mean);
        m_localWeights=TransfUtils<FloatImageType>::createEmptyImage(m_mean);

        return (diffNorm<convergenceCriterion);
    }

    virtual  DeformationFieldPointerType getMean(){return m_oldMean;}
  

    
};//class
  
template<typename DeformationType>
struct LIST{
    LIST * pNext;
    double           iValue;
    DeformationType def;
};


  template<typename DeformationType>
  LIST<DeformationType> * insert( LIST<DeformationType> * pList, double i,DeformationType def, int maxLength){
 
    LIST<DeformationType> * pHead = pList;
    
    /* trailing pointer for efficient splice */
    struct LIST<DeformationType> * pTrail = pList;
    struct LIST<DeformationType> * pPrev = NULL;

    int c=0;
    /* splice head into sorted list at proper place */
    bool ins=false;
    while (c<maxLength)
        {
            
            /* does head belong here? */
            if ((pTrail == NULL || i  > (pTrail)->iValue ) && ! ins)
                {
                    //allocate new list element
                    struct LIST<DeformationType> * pNew = new LIST<DeformationType>;
                    pNew->iValue= i;
                    pNew->def = def;
                    //let new element point to previous element
                    pNew->pNext = pTrail;
                    
                    //let previous element point to new element
                    if (pPrev!=NULL)
                        pPrev->pNext=pNew;
                    
                    //
                    pPrev=pNew;
                    if (c == 0){
                        //new is new head
                        pHead=pNew;
                    }
                    //added a new element!
                    ++c;
                    ins=true;
                }
            else
                {
                    /* no - continue down the list */
                    pPrev=pTrail;
                    if (ins && !pTrail)
                        break;
                    pTrail =  (pTrail)->pNext;
                    
                }
            ++c;
            if (c>=maxLength){
                if (pTrail){
                    //  delete pTrail;
                }
            }
        }
    return pHead;
}

template<typename ImageType>
class FastNBestGaussianVectorImage: public FastGaussianEstimatorVectorImage<ImageType>{

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
    int m_N;
  
public:    
    typedef LIST<DeformationType> * LISTTYPE;
    typedef typename itk::Image< LISTTYPE ,D> SortedListImageType;
    typedef typename SortedListImageType::Pointer SortedListImagePointerType;
    typedef typename itk::ImageRegionIterator<SortedListImageType> ListImageIteratorType;


private:
    SortedListImagePointerType m_sortedListImage;
    
public:
    FastNBestGaussianVectorImage(){
#if 0
        DeformationType d;
        double  array[5] ={1,3,2,0,5};
        LISTTYPE l=NULL;
        for (int i=0;i<5;++i){
            l=insert(l,array[i],d,20);
        }
        while (l != NULL ){
            LOG<<VAR(l->iValue)<<endl;
            l=l->pNext;
        }
#endif
    }
    ~FastNBestGaussianVectorImage(){
        ListImageIteratorType listIt(m_sortedListImage,m_sortedListImage->GetLargestPossibleRegion());
        listIt.GoToBegin();
        for (;!listIt.IsAtEnd();++listIt){
            LISTTYPE  l=listIt.Get();
            while ( l != NULL ){
                LISTTYPE tmp=l;
                l=l->pNext;
                delete tmp;
            }
        }
    }
    
    virtual void initialize(DeformationFieldPointerType img, FloatImagePointerType weights=NULL){
        this->count=1;
        this->m_mean=ImageUtils<DeformationFieldType>::createEmpty(img);
        m_sortedListImage=SortedListImageType::New();
        m_sortedListImage->SetLargestPossibleRegion(img->GetLargestPossibleRegion());
        m_sortedListImage->SetBufferedRegion(img->GetLargestPossibleRegion());
        m_sortedListImage->Allocate();
        m_sortedListImage->FillBuffer(NULL);
    }
 
    void addImage(DeformationFieldPointerType img,FloatImagePointerType weights, int N=3){
       
        DeformationImageIteratorType inputIt(img,this->m_mean->GetLargestPossibleRegion());
        ListImageIteratorType listIt(m_sortedListImage,m_sortedListImage->GetLargestPossibleRegion());
        inputIt.GoToBegin(); listIt.GoToBegin();
        
        FloatImageIteratorType weightIt;
        weightIt=FloatImageIteratorType(weights,weights->GetLargestPossibleRegion());
        weightIt.GoToBegin();
        
        for (;!inputIt.IsAtEnd();++inputIt,++weightIt,++listIt){
            DeformationType def=inputIt.Get();
            if (weightIt==weightIt.Begin()){
                //LOG<<VAR(def)<<" "<<VAR(weightIt.Get())<<endl;
                
            }
            double w=weightIt.Get();
            LISTTYPE l=listIt.Get();
            listIt.Set(insert(l,w,def,N));
        }    
        m_N=N;
        this->count++;
        
    }
    virtual void finalize(){

        DeformationImageIteratorType meanIt(this->m_mean,this->m_mean->GetLargestPossibleRegion());
        ListImageIteratorType listIt(m_sortedListImage,m_sortedListImage->GetLargestPossibleRegion());
        meanIt.GoToBegin();      listIt.GoToBegin();
     
        for (;!meanIt.IsAtEnd();++meanIt,++listIt){
            
            LISTTYPE  l=listIt.Get();
            DeformationType meanDef;
            meanDef.Fill(0.0);
            double wAccumulator=0.0;
            int c=0;
            while (c<m_N && l != NULL ){
                //  cout<<"("<<l->iValue<<","<<l->def<<") ";
                meanDef=meanDef+(l->def*l->iValue);
                wAccumulator+=l->iValue;
                l=l->pNext;
                ++c;
            }
            //LOG<<VAR(wAccumulator)<<" "<<VAR(meanDef)<<endl;
            //
         
            
            if (wAccumulator > 0.0){
                meanDef=meanDef/wAccumulator;
                meanIt.Set(meanDef);
                //LOG<<VAR(meanDef)<<endl;
            }

         

        }

    }
    virtual DeformationFieldPointerType getMean(){return this->m_mean;}

   
};//class

