/*
   * Potentials.h
   *
   *  Created on: Nov 24, 2010
   *      Author: gasst
   */

#ifndef _SEGMENTATIONPAIRWISEPOTENTIALS_H_
#define _SEGMENTATIONPAIRWISEPOTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include <itkStatisticsImageFilter.h>

namespace itk{



    template<class TImage>
    class PairwisePotentialSegmentation: public itk::Object{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentation            Self;
        typedef itk::Object Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        SizeType m_fixedSize;
        typedef typename itk::StatisticsImageFilter< ImageType > StatisticsFilterType;
    protected:
        ConstImagePointerType m_fixedImage, m_sheetnessImage;
        SpacingType m_displacementFactor;
        //LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        double m_gradientSigma, m_Sigma;
        double m_gradientScaling;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentation, Object);

        PairwisePotentialSegmentation(){
            this->m_haveLabelMap=false;
        }
        virtual void freeMemory(){
        }
        virtual void Init(){
            assert(this->m_fixedImage);
            assert(this->m_sheetnessImage);
	  
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(this->m_sheetnessImage);
            filter->Update();
            this->m_gradientSigma=filter->GetSigma();
            this->m_gradientSigma*=this->m_gradientSigma;
            std::cout<<"Gradient variance: "<<m_gradientSigma<<std::endl;
           
            filter->SetInput(this->m_fixedImage);
            filter->Update();
            this->m_Sigma=filter->GetSigma();
            this->m_Sigma*=this->m_Sigma;	  
        }
        void SetGradientScaling(double s){m_gradientScaling=s;}
        void SetFixedImage(ConstImagePointerType fixedImage){
            this->m_fixedImage=fixedImage;
            this->m_fixedSize=this->m_fixedImage->GetLargestPossibleRegion().GetSize();

        }
        void SetFixedGradient(ConstImagePointerType sheetnessImage){
            this->m_sheetnessImage=sheetnessImage;
            
            
        }
        
        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            if (label1!=label2){  
                int s1=this->m_sheetnessImage->GetPixel(idx1);
                int s2=this->m_sheetnessImage->GetPixel(idx2);
                double edgeWeight=fabs(s1-s2);
                edgeWeight*=edgeWeight;
                int i1=this->m_fixedImage->GetPixel(idx1);
                int i2=this->m_fixedImage->GetPixel(idx2);
                double intensityDiff=(i1-i2)*(i1-i2);
                edgeWeight=(s1 < s2) ? 1.0 : exp( - 40* (edgeWeight/this->m_gradientSigma) );
                return edgeWeight;
            }else{
                return 0;
            }
        }
    };//class

    template<class TImage, class TSmoothnessClassifier>
    class PairwisePotentialSegmentationClassifier: public PairwisePotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationClassifier            Self;
        typedef PairwisePotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef TSmoothnessClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
    private:
        ClassifierPointerType m_classifier;
        ConstImagePointerType m_referenceSegmentation, m_referenceGradient, m_referenceImage;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationClassifier, Object);
        virtual void Init(){
            assert(this->m_fixedImage);
            assert(this->m_sheetnessImage);
            assert(this->m_referenceSegmentation);
            assert(this->m_referenceGradient);
            assert(this->m_referenceImage);
            m_classifier=ClassifierType::New();
            m_classifier->setNIntensities(256);
            m_classifier->setData((ConstImagePointerType)m_referenceSegmentation,m_referenceImage,(ConstImagePointerType)m_referenceGradient);
            m_classifier->train();
        }
        virtual void Init(string filename){
            assert(false);
            assert(this->m_fixedImage);
            assert(this->m_sheetnessImage);
            m_classifier=ClassifierType::New();
            
            //m_classifier->LoadProbs(filename);
        }
        virtual void SetReferenceSegmentation(ConstImagePointerType im){
            m_referenceSegmentation=im;
        }
        virtual void SetReferenceGradient(ConstImagePointerType im){
            m_referenceGradient=im;
        }
        virtual void SetReferenceImage(ConstImagePointerType im){
            m_referenceImage=im;
        }
        ClassifierPointerType GetClassifier(){return m_classifier;}
        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            int s1=this->m_sheetnessImage->GetPixel(idx1);
            int s2=this->m_sheetnessImage->GetPixel(idx2);
            double sheetnessDiff=fabs(s1-s2);
            int i1=this->m_fixedImage->GetPixel(idx1);
            int i2=this->m_fixedImage->GetPixel(idx2);
            double intensityDiff=fabs(i1-i2);
            double prob=m_classifier->px_l(intensityDiff,label1==label2,sheetnessDiff);
            //std::cout<<intensityDiff<<" "<<sheetnessDiff<<" "<<prob<<" "<<-log(prob)<<endl;
            return -log(prob);
        }
      
    };//class
}//namespace
#endif /* POTENTIALS_H_ */
