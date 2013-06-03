#include "Log.h"
/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SEGMENTATIONPOTENTIALS_H_
#define _SEGMENTATIONPOTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include <itkStatisticsImageFilter.h>
#include "Potential-SegmentationRegistration-Pairwise.h"

namespace itk{



    template<class TImage>
    class UnaryPotentialSegmentation: public itk::Object{
    public:
        //itk declarations
        typedef UnaryPotentialSegmentation            Self;
        typedef itk::Object Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        SizeType m_targetSize;
        typedef typename itk::StatisticsImageFilter< ImageType > StatisticsFilterType;
    protected:
        ImageConstPointerType m_targetImage, m_targetGradient,m_atlasImage, m_atlasGradient;
        ImageConstPointerType m_scaledTargetImage, m_scaledTargetGradient;
        ImageConstPointerType m_atlasSegmentation;
        SpacingType m_displacementFactor;
        //LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        double m_gradientSigma, m_Sigma;
        double m_gradientScaling;
        ImageConstPointerType m_tissuePrior;
        bool m_useTissuePrior;
        int m_nSegmentationLabels;
    public:
        
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentation, Object);
        void SetTissuePrior(ImageConstPointerType img){m_tissuePrior=img;}
        void SetUseTissuePrior(bool b){
            m_useTissuePrior=b;
        }
        UnaryPotentialSegmentation(){
            this->m_haveLabelMap=false;
        }
        void SetNSegmentationLabels(int n){m_nSegmentationLabels=n;}
        virtual void Init(){}
        virtual void freeMemory(){
        }
        void SetGradientScaling(double s){m_gradientScaling=s;}
        void SetTargetImage(ImageConstPointerType targetImage){
            this->m_targetImage=targetImage;
            this->m_targetSize=this->m_targetImage->GetLargestPossibleRegion().GetSize();
        }
        void SetTargetGradient(ImageConstPointerType targetGradient){
            this->m_targetGradient=targetGradient;
            
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(this->m_targetGradient);
            filter->Update();
            this->m_gradientSigma=filter->GetSigma();
            this->m_gradientSigma*=this->m_gradientSigma;
            LOGV(4)<<"Target image gradient variance: "<<m_gradientSigma<<std::endl;
            filter->SetInput(this->m_targetImage);
            filter->Update();
            this->m_Sigma=filter->GetSigma();
            this->m_Sigma*=this->m_Sigma;
                                    
        }
        void ResamplePotentials(double segmentationScalingFactor){
            if (segmentationScalingFactor<1.0){
                //only use gaussian smoothing if downsampling
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(m_targetImage,segmentationScalingFactor,false,false);
                m_scaledTargetGradient=FilterUtils<ImageType>::LinearResample(m_targetGradient,segmentationScalingFactor,false,false);
            }else{
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(m_targetImage,segmentationScalingFactor,true,false);
                m_scaledTargetGradient=FilterUtils<ImageType>::LinearResample(m_targetGradient,segmentationScalingFactor,true,false);
            }
            
        }
        virtual void SetAtlasSegmentation(ImageConstPointerType im){
            m_atlasSegmentation=im;
        }
        virtual void SetAtlasGradient(ImageConstPointerType im){
            m_atlasGradient=im;
        }
        virtual void SetAtlasImage(ImageConstPointerType im){
            m_atlasImage=im;
        }
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            int s= this->m_targetGradient->GetPixel(targetIndex);
            double imageIntensity=this->m_targetImage->GetPixel(targetIndex);
            double segmentationProb=1;
            int tissue=-500;
            int bone=300;
            if (segmentationLabel>0) {
                if (imageIntensity < tissue)
                    segmentationProb = fabs(tissue-imageIntensity);
                else if (imageIntensity <bone)
                    segmentationProb = 0.69;
                else
                    segmentationProb = 0;
            }else{
                if (imageIntensity > bone && s>0)
                    segmentationProb = fabs(imageIntensity-bone);
                else if (imageIntensity > tissue)
                    segmentationProb = 0.69;
                else
                    segmentationProb = 0;
            }
            return segmentationProb;
        }

        virtual double getWeight(IndexType idx1, IndexType idx2){
            int s1=this->m_targetGradient->GetPixel(idx1);
            int s2=this->m_targetGradient->GetPixel(idx2);
            double edgeWeight=fabs(s1-s2);
            edgeWeight*=edgeWeight;

            //int i1=this->m_targetImage->GetPixel(idx1);
            //int i2=this->m_targetImage->GetPixel(idx2);
            //double intensityDiff=(i1-i2)*(i1-i2);
            edgeWeight=(s1 < s2) ? 1.0 : exp( - 40* (edgeWeight/this->m_gradientSigma) );
            //edgeWeight=(s1 < s2) ? 1.0 : exp( - 0.05* edgeWeight );
            //edgeWeight= exp( - 0.5 * 0.5*(edgeWeight/this->m_gradientSigma) +intensityDiff/this->m_Sigma);
            //edgeWeight= 0.5 * 0.5*(edgeWeight/this->m_gradientSigma +intensityDiff/this->m_Sigma);
            //edgeWeight= 0.5 * (edgeWeight/this->m_gradientSigma);
            //edgeWeight= 1;//0.5 * intensityDiff/this->m_Sigma;
            return edgeWeight;
        }
    };//class


   
    template<class TImage>
    class UnaryPotentialSegmentationArtificial: public UnaryPotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationArtificial            Self;
        typedef  UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::IndexType IndexType;

    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationArtificial, Object);
        
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double imageIntensity=this->m_targetImage->GetPixel(targetIndex);
            double segmentationProb=1;
            switch (segmentationLabel) {
            case 1  :
                segmentationProb = (imageIntensity < 128 ) ? 1 : 0;
                break;
            case 0:
                segmentationProb = ( imageIntensity > 128)  ? 1 : 0;
                break;
            default:
                assert(false);
            }
            //        LOG<<targetIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            return segmentationProb;
        }

      
    };//class
    template<class TImage>
    class UnaryPotentialSegmentationArtificial2: public  UnaryPotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationArtificial2            Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationArtificial2, Object);
        
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double imageIntensity=this->m_targetImage->GetPixel(targetIndex);
            double segmentationProb=1;
            if (segmentationLabel>=1) {
                segmentationProb = (imageIntensity < 85 || imageIntensity>170  ) ? 1 : 0;
            }
            else{
                segmentationProb =  (imageIntensity > 85 && imageIntensity<170  )  ? 1 : 0;
            }
            //        LOG<<targetIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            return segmentationProb;
        }

    };//class

 

    template<class TImage>
    class UnaryPotentialSegmentationProb: public UnaryPotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationProb            Self;
        typedef  UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationProb, Object);
        
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double imageIntensity=1.0*this->m_targetImage->GetPixel(targetIndex)/255;
            double segmentationProb=1;
            if (segmentationLabel>0) {
                segmentationProb = 1-imageIntensity;//(imageIntensity < 0.7 ) ? 1 : 0;
            }else{
                segmentationProb = imageIntensity;//( imageIntensity > 0.4) ? 1 : 0;
            }
            //   LOG<<targetIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            
            return segmentationProb;
        }

    };//class

    template<class TImage>
    class UnaryPotentialSegmentationUnsignedBone: public UnaryPotentialSegmentation<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationUnsignedBone            Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
    public:
        
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationUnsignedBone, Object);

        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            int s=this->m_targetGradient->GetPixel(targetIndex);
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;
            double imageIntensity=this->m_targetImage->GetPixel(targetIndex);
            double segmentationProb=1;
            if (segmentationLabel>0) {
                if (imageIntensity < tissue)
                    segmentationProb =fabs(imageIntensity-tissue);
                else if (imageIntensity < bone) 
                    segmentationProb = 0.69; //log (0.5);
                else
                    segmentationProb = 0.00000001;
            }else{
                if ((imageIntensity >  bone)  && s>128)
                    segmentationProb = fabs(imageIntensity-bone);
                else if (imageIntensity >tissue)
                    segmentationProb =0.69 ;
                else
                    segmentationProb = 0.00000001;
                
                //            if (segmentationLabel>0) {
                //                segmentationProb = (imageIntensity < (-500+1000)*255.0/2000 ) ? 1 : 0;
                //            }else{
                //                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000 ) && ( s > 128 ) ? 1 : 0;
            }
            return segmentationProb;
        }

        virtual double getWeight(IndexType idx1, IndexType idx2){
            int s1=this->m_targetGradient->GetPixel(idx1);
            int s2=this->m_targetGradient->GetPixel(idx2);
            double edgeWeight=fabs(s1-s2);
            //edgeWeight*=edgeWeight;

            int i1=this->m_targetImage->GetPixel(idx1);
            int i2=this->m_targetImage->GetPixel(idx2);
            double intensityDiff=(i1-i2)*(i1-i2);
            edgeWeight=(s1 < s2) ? 0.99999999 : exp( - 2*(edgeWeight) );
            //edgeWeight=(s1 < s2) ? 1.0 : exp( - 0.05* edgeWeight );
            //edgeWeight= exp( - 0.5 * 0.5*(edgeWeight/this->m_gradientSigma) +intensityDiff/this->m_Sigma);
            //edgeWeight= 0.5 * 0.5*(edgeWeight/this->m_gradientSigma +intensityDiff/this->m_Sigma);
            //edgeWeight= 0.5 * (edgeWeight/this->m_gradientSigma);
            //edgeWeight= 1;//0.5 * intensityDiff/this->m_Sigma;
#if 0
            edgeWeight=1-edgeWeight;
            if (edgeWeight<=0) edgeWeight=0.00001;
            edgeWeight=-log(edgeWeight);
#endif
            return edgeWeight;
        }
    };

    template<class TImage, class TClassifier>
    class UnaryPotentialSegmentationClassifier: public UnaryPotentialSegmentation<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationClassifier            Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef TClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
        
    protected:
        ImageConstPointerType m_deformationPrior;
        double m_alpha;
        ClassifierPointerType m_classifier;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationClassifier, Object);
          
        virtual void Init(){
            
            m_classifier=  ClassifierType::New();
            m_classifier->setNIntensities(3500);
            m_classifier->setData(this->m_atlasImage,this->m_atlasSegmentation,(ImageConstPointerType)this->m_atlasGradient);
            //m_classiifier->setData(movingImage,movingSegmentationImage);
#if 1
            m_classifier->train();
            m_classifier->saveProbs("test.probs");
#else
            m_classifier->loadProbs("test.probs");
#endif

            m_classifier->evalImage(this->m_targetImage,this->m_targetGradient);
        }
        
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double imageIntensity=this->m_targetImage->GetPixel(targetIndex);
            
            if (false && segmentationLabel){
                bool tissuePrior = (this->m_tissuePrior->GetPixel(targetIndex))>0;
                if (tissuePrior)
                    return 100;
            }

            
            int s= this->m_targetGradient->GetPixel(targetIndex);

            //prob of inverse segmentation label
            //double prob=m_classifier->px_l(imageIntensity,s,(segmentationLabel));
           
            //LOG<<imageIntensity<<" "<<s<<" "<<segmentationLabel<<" "<<prob<<" "<< -log(prob) <<std::endl ;
            //penalize only if prob <0.6
            //double prob=m_classifier->px_l(imageIntensity,(segmentationLabel>0));
            double prob=m_classifier->px_l(imageIntensity,(segmentationLabel>0),s);
            if (prob<=0) prob=0.00000000001;
            //if (segmentationLabel && prob<0.5) prob=0.5;
            return -log(prob);
            //LOG<<segmentationLabel<<" "<<imageIntensity<<" "<<prob<<endl;
            //return (1-prob);

        }

        virtual double getWeight(IndexType idx1, IndexType idx2){
            assert(false);
            return -1;
        }
      
    };

  

    template<class TImage, class TClassifier>
    class UnaryPotentialSegmentationClassifierWithPrior: 
        public UnaryPotentialSegmentationClassifier<TImage,TClassifier>  
    {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationClassifierWithPrior            Self;
        typedef UnaryPotentialSegmentationClassifier<TImage,TClassifier> UnarySuperclass;


        
        
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef TClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;

        typedef PairwisePotentialSegmentationRegistration<TImage> SRSPotentialType;
        typedef typename SRSPotentialType::Pointer SRSPotentialPointerType;
        
    protected:
        double m_alpha;
        SRSPotentialPointerType m_srsPotential;
        itk::Vector<float,ImageType::ImageDimension> zeroDisplacement;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationClassifier, Object);

        UnaryPotentialSegmentationClassifierWithPrior(){
            zeroDisplacement.Fill(0.0);
        }
        void SetAlpha(double alpha){this->m_alpha=alpha;}     
        void SetSRSPotential(SRSPotentialPointerType pot){m_srsPotential=pot;}
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double origPotential=UnarySuperclass::getPotential(targetIndex,segmentationLabel);
            
            double priorPotential=m_srsPotential->getPotential(targetIndex,targetIndex,zeroDisplacement,segmentationLabel);
            return origPotential+m_alpha*priorPotential;
        }
    };//class

    
    template<class TImage>
    class UnaryPotentialSegmentationUnsignedBoneMarcel: public UnaryPotentialSegmentation<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationUnsignedBoneMarcel           Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename TImage::ConstPointer ImageConstPointerType;
    public:
        
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationUnsignedBone, Object);
        
        
        UnaryPotentialSegmentationUnsignedBoneMarcel(){
            this->m_useTissuePrior=false;//true;
        }
      
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            int s=this->m_scaledTargetGradient->GetPixel(targetIndex);
            int bone=(200+1000);//*255.0/2000;
            int tissue=900;//(-500+1000);//*255.0/2000;
            double imageIntensity=this->m_scaledTargetImage->GetPixel(targetIndex);
            double totalCost=1;
            bool tissuePrior=false;
            //#define USEPRIOR
            if (this->m_useTissuePrior){
                tissuePrior = (this->m_tissuePrior->GetPixel(targetIndex))>0;
            }

            switch (segmentationLabel) {
            case 0:
#ifdef USEPRIOR
                //if (this->m_tissuePrior->GetPixel(targetIndex)>0.8) return 100;
                totalCost = ( ((  imageIntensity > bone) && ( s > 0 ) ) )? 1 : 0;
#else
                totalCost = ( imageIntensity > bone) && ( s > 0 ) ? 1 : 0;
                //LOG<<totalCost<<" "<<imageIntensity<<endl;
#endif
                break;
            default  :
#ifdef USEPRIOR
                //if (this->m_tissuePrior->GetPixel(targetIndex)<0.1) return 100;
                totalCost = ( bonePrior || imageIntensity < tissue) ? 1 : 0;
#else
                totalCost = (imageIntensity < tissue) ? 1 : 0;
#endif
                break;
            }
            //            LOG<<imageIntensity<<" "<<segmentationLabel<<" "<<totalCost<<" "<<bone<<" "<<tissue<<" "<<s<<std::endl;
            return totalCost;
        }
    };//class


    template<class TImage>
    class UnaryPotentialSegmentationBoneMarcel: 
        public UnaryPotentialSegmentationUnsignedBoneMarcel<TImage>  
    {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationBoneMarcel            Self;
        typedef UnaryPotentialSegmentationUnsignedBoneMarcel<TImage> UnarySuperclass;
        
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef PairwisePotentialSegmentationRegistration<TImage> SRSPotentialType;
        typedef typename SRSPotentialType::Pointer SRSPotentialPointerType;
        
    protected:
        double m_alpha;
        SRSPotentialPointerType m_srsPotential;
        itk::Vector<float,ImageType::ImageDimension> zeroDisplacement;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationBoneMarcel, Object);
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            int s=this->m_scaledTargetGradient->GetPixel(targetIndex);
            int bone=(300);
            int tissue=(-500);
            double imageIntensity=this->m_scaledTargetImage->GetPixel(targetIndex);
            double totalCost=1;
            bool tissuePrior=false;
            if (this->m_useTissuePrior){
                tissuePrior = (this->m_tissuePrior->GetPixel(targetIndex))>0;
            }
            
            switch (segmentationLabel) {
            case 0:
                
                totalCost = ( imageIntensity > bone) && ( s > 0 ) ? 1 : 0;
                
                break;
            default  :
                totalCost = (tissuePrior || imageIntensity < tissue) ? 1 : 0;
                break;
            }
            return totalCost;
        }
    };//class

    
    template<class TImage>
    class UnaryPotentialSegmentationMarcelWithPrior: 
        public UnaryPotentialSegmentationUnsignedBoneMarcel<TImage>  
    {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationMarcelWithPrior            Self;
        typedef UnaryPotentialSegmentationUnsignedBoneMarcel<TImage> UnarySuperclass;
        
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef PairwisePotentialSegmentationRegistration<TImage> SRSPotentialType;
        typedef typename SRSPotentialType::Pointer SRSPotentialPointerType;
        
    protected:
        double m_alpha;
        SRSPotentialPointerType m_srsPotential;
        itk::Vector<float,ImageType::ImageDimension> zeroDisplacement;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationMarcelWithPrior, Object);

        UnaryPotentialSegmentationMarcelWithPrior(){
            zeroDisplacement.Fill(0.0);
            this->SetUseTissuePrior(false);
        }
        void SetAlpha(double alpha){this->m_alpha=alpha;}     
        void SetSRSPotential(SRSPotentialPointerType pot){m_srsPotential=pot;}
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double origPotential=UnarySuperclass::getPotential(targetIndex,segmentationLabel);
            double priorPotential=m_srsPotential->getPotential(targetIndex,targetIndex,zeroDisplacement,segmentationLabel);
            return origPotential+m_alpha*priorPotential;
        }
    };//class

    template<class TImage, class TClassifier>
    class UnaryPotentialNewSegmentationClassifier: public UnaryPotentialSegmentation<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialNewSegmentationClassifier            Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef TClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
        typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        
    protected:
        ClassifierPointerType m_classifier;
        std::vector<FloatImagePointerType> m_probabilityImages,m_resampledProbImages;
        bool m_trainOnTargetROI;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialNewSegmentationClassifier, Object);
          
        virtual void Init(){
            m_trainOnTargetROI=true;
            LOG<<VAR(m_trainOnTargetROI)<<std::endl;
            m_classifier=  ClassifierType::New();
            m_classifier->setNSegmentationLabels(2);
            std::vector<ImageConstPointerType> atlas;
            if (m_trainOnTargetROI){
                this->m_atlasImage=FilterUtils<ImageType>::NNResample(this->m_atlasImage,this->m_targetImage,false);
                this->m_atlasSegmentation=FilterUtils<ImageType>::NNResample(this->m_atlasSegmentation,this->m_targetImage,false);
            }
            atlas.push_back(this->m_atlasImage);
            //atlas.push_back(this->m_atlasGradient);
            m_classifier->setData(atlas,this->m_atlasSegmentation);
            m_classifier->train();
            std::vector<ImageConstPointerType> target;
            target.push_back(this->m_targetImage);
            //            target.push_back(this->m_targetGradient);
            m_probabilityImages=m_classifier->evalImage(target);
        }
        void ResamplePotentials(double scale){
            m_resampledProbImages= std::vector<FloatImagePointerType>(m_probabilityImages.size());
            for (int i=0;i<m_probabilityImages.size();++i){
                m_resampledProbImages[i]=FilterUtils<FloatImageType>::LinearResample(m_probabilityImages[i],scale,true);
            }
        }
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double prob= m_resampledProbImages[segmentationLabel>0]->GetPixel(targetIndex);
            //return prob;
            if (prob<=0) prob=0.00000000001;
            return -log(prob);
        }
    };


    template<class TImage, class TClassifier>
    class UnaryPotentialNewSegmentationMultilabelClassifier: public UnaryPotentialSegmentation<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialNewSegmentationMultilabelClassifier            Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef TClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
        typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        
    protected:
        ClassifierPointerType m_classifier;
        std::vector<FloatImagePointerType> m_probabilityImages,m_resampledProbImages;
        bool m_trainOnTargetROI;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialNewSegmentationMultilabelClassifier, Object);
          
        virtual void Init(){
            m_trainOnTargetROI=true;
            LOG<<VAR(m_trainOnTargetROI)<<std::endl;
            m_classifier=  ClassifierType::New();
            m_classifier->setNSegmentationLabels(max(2,this->m_nSegmentationLabels));
            std::vector<ImageConstPointerType> atlas;
            if (m_trainOnTargetROI){
                this->m_atlasImage=FilterUtils<ImageType>::NNResample(this->m_atlasImage,this->m_targetImage,false);
                this->m_atlasSegmentation=FilterUtils<ImageType>::NNResample(this->m_atlasSegmentation,this->m_targetImage,false);
            }
            atlas.push_back(this->m_atlasImage);
            //atlas.push_back(this->m_atlasGradient);
            m_classifier->setData(atlas,this->m_atlasSegmentation);
            m_classifier->train();
            std::vector<ImageConstPointerType> target;
            target.push_back(this->m_targetImage);
            //            target.push_back(this->m_targetGradient);
            m_probabilityImages=m_classifier->evalImage(target);
        }
        void ResamplePotentials(double scale){
            m_resampledProbImages= std::vector<FloatImagePointerType>(m_probabilityImages.size());
            for (int i=0;i<m_probabilityImages.size();++i){
                m_resampledProbImages[i]=FilterUtils<FloatImageType>::LinearResample(m_probabilityImages[i],scale,true);
            }
        }
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){
            double prob= m_resampledProbImages[segmentationLabel]->GetPixel(targetIndex);
            return prob;
            if (prob<=0) prob=0.00000000001;
            return -log(prob);
        }
    };

      template<class TImage, class TClassifier>
    class UnaryPotentialNewSegmentationMultilabelClassifierNoCaching: public UnaryPotentialSegmentation<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialNewSegmentationMultilabelClassifierNoCaching            Self;
        typedef UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;

        typedef TClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
        typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        
    protected:
        ClassifierPointerType m_classifier;
        std::vector<FloatImagePointerType> m_probabilityImages,m_resampledProbImages;
        bool m_trainOnTargetROI;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialNewSegmentationMultilabelClassifier, Object);
          
        virtual void Init(){
            m_trainOnTargetROI=true;
            LOG<<VAR(m_trainOnTargetROI)<<std::endl;
            m_classifier=  ClassifierType::New();
            m_classifier->setNSegmentationLabels(max(2,this->m_nSegmentationLabels));
            std::vector<ImageConstPointerType> atlas;
            if (m_trainOnTargetROI){
                this->m_atlasImage=FilterUtils<ImageType>::NNResample(this->m_atlasImage,this->m_targetImage,false);
                this->m_atlasSegmentation=FilterUtils<ImageType>::NNResample(this->m_atlasSegmentation,this->m_targetImage,false);
            }
            atlas.push_back(this->m_atlasImage);
            //atlas.push_back(this->m_atlasGradient);
            m_classifier->setData(atlas,this->m_atlasSegmentation);
            m_classifier->train();
            std::vector<ImageConstPointerType> target;
            target.push_back(this->m_targetImage);
            //            target.push_back(this->m_targetGradient);
            LOGI(10,m_probabilityImages=m_classifier->evalImage(target));

        }
        void ResamplePotentials(double scale){

            this->m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(this->m_targetImage,scale,true);
            this->m_scaledTargetGradient=FilterUtils<ImageType>::LinearResample(this->m_targetGradient,scale,true);
            
        }
        virtual double getPotential(IndexType targetIndex, int segmentationLabel){

            double i1=this->m_scaledTargetImage->GetPixel(targetIndex);
            double i2=this->m_scaledTargetGradient->GetPixel(targetIndex);
            double prob= m_classifier->getProbability(segmentationLabel,i1,i2);
            if (prob<=0) 
                return 10000.0;
            else{
                prob=min(1.0,prob);
                return -log(prob);
            }
        }
    };

}//namespace
#endif /* POTENTIALS_H_ */
