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
        double m_gradientSigma;
        double m_gradientScaling;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentation, Object);

        UnaryPotentialSegmentation(){
            this->m_haveLabelMap=false;
        }
        virtual void freeMemory(){
        }
        void SetGradientScaling(double s){m_gradientScaling=s;}
        void SetFixedImage(ConstImagePointerType fixedImage){
            this->m_fixedImage=fixedImage;
            this->m_fixedSize=this->m_fixedImage->GetLargestPossibleRegion().GetSize();
        }
        void SetGradientImage(ConstImagePointerType sheetnessImage){
            this->m_sheetnessImage=sheetnessImage;
            
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(this->m_sheetnessImage);
            filter->Update();
            this->m_gradientSigma=filter->GetSigma();
            this->m_gradientSigma*=this->m_gradientSigma;
            std::cout<<"Gradient std deviation: "<<m_gradientSigma<<std::endl;
        }
        
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            int s= this->m_sheetnessImage->GetPixel(fixedIndex);
            double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
            double segmentationProb=1;
            int tissue=-500;
            int bone=300;
            if (segmentationLabel>0) {
                //segmentationProb = exp(imageIntensity+500);//(imageIntensity < -500 ) ? 1 : 0;
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
                //segmentationProb = exp(imageIntensity-300);// ( imageIntensity > 300) && ( s > 0 ) ? 1 : 0;
            }
            //        std::cout<<fixedIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            return segmentationProb;
        }

        virtual double getWeight(IndexType idx1, IndexType idx2){
            int s1=this->m_sheetnessImage->GetPixel(idx1);
            int s2=this->m_sheetnessImage->GetPixel(idx2);
            double edgeWeight=fabs(s1-s2);
            edgeWeight*=edgeWeight;
            //edgeWeight=(s1 < s2) ? 1.0 : exp( - 20* (edgeWeight/this->m_gradientSigma) );
            edgeWeight= exp( - 20 * (edgeWeight/this->m_gradientSigma) );
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
        
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
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
            //        std::cout<<fixedIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
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
        
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
            double segmentationProb=1;
            if (segmentationLabel>=1) {
                segmentationProb = (imageIntensity < 85 || imageIntensity>170  ) ? 1 : 0;
            }
            else{
                segmentationProb =  (imageIntensity > 85 && imageIntensity<170  )  ? 1 : 0;
            }
            //        std::cout<<fixedIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            return segmentationProb;
        }

    };//class

    template<class TImage>
    class UnaryPotentialSegmentationWithRegistrationPrior: public UnaryPotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationWithRegistrationPrior            Self;
        typedef  UnaryPotentialSegmentation<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
    protected:
        ConstImagePointerType m_deformationPrior;
        double m_alpha;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentation, Object);
        void SetDeformationPrior(ConstImagePointerType deformedSegmentation){
            this->m_deformationPrior=deformedSegmentation;
        }
          
        void SetAlpha(double alpha){this->m_alpha=alpha;}
        
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            int s= this->m_sheetnessImage->GetPixel(fixedIndex);
            double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
            double segmentationProb=1;
#if 0
            if (segmentationLabel>=1) {
                segmentationProb = (imageIntensity < 85 || imageIntensity>170  ) ? 1 : 0;
            }
            else{
                segmentationProb =  (imageIntensity > 85 && imageIntensity<170  )  ? 1 : 0;
            }
#else
            if (segmentationLabel>0) {
                segmentationProb = (imageIntensity < -500 ) ? 1 : 0;
            }else{
                segmentationProb = ( imageIntensity > 300) && ( s > 0 ) ? 1 : 0;
            }
#endif
            int deformedSegmentationPenalty=0;
            if (this->m_deformationPrior)
                deformedSegmentationPenalty=(segmentationLabel!=this->m_deformationPrior->GetPixel(fixedIndex));
            //std::cout<<fixedIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<" "<<this->m_alpha*deformedSegmentationPenalty<<std::endl;
            return segmentationProb+this->m_alpha*deformedSegmentationPenalty;
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
        
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            double imageIntensity=1.0*this->m_fixedImage->GetPixel(fixedIndex)/255;
            double segmentationProb=1;
            if (segmentationLabel>0) {
                segmentationProb = 1-imageIntensity;//(imageIntensity < 0.7 ) ? 1 : 0;
            }else{
                segmentationProb = imageIntensity;//( imageIntensity > 0.4) ? 1 : 0;
            }
            //   std::cout<<fixedIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            
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

        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            int s=this->m_sheetnessImage->GetPixel(fixedIndex);
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;
            double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
            double segmentationProb=1;
            if (segmentationLabel>0) {
                if (imageIntensity < tissue)
                    segmentationProb =fabs(imageIntensity-tissue);
                else if (imageIntensity < bone) 
                    segmentationProb = 0.69; //log (0.5);
                else
                    segmentationProb = 0;
            }else{
                if ((imageIntensity >  bone)  && s>128)
                    segmentationProb = fabs(imageIntensity-bone);
                else if (imageIntensity >tissue)
                    segmentationProb =0.69 ;
                else
                    segmentationProb = 0;
                
                //            if (segmentationLabel>0) {
                //                segmentationProb = (imageIntensity < (-500+1000)*255.0/2000 ) ? 1 : 0;
                //            }else{
                //                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000 ) && ( s > 128 ) ? 1 : 0;
            }
            return segmentationProb;
        }
    };

    template<class TImage>
    class UnaryPotentialSegmentationUnsignedBoneWithPrior: public UnaryPotentialSegmentationUnsignedBone<TImage> {
    public:
        //itk declarations
        typedef UnaryPotentialSegmentationUnsignedBoneWithPrior            Self;
        typedef UnaryPotentialSegmentationUnsignedBone<TImage> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        
        typedef TImage ImageType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
    protected:
        ConstImagePointerType m_deformationPrior;
        double m_alpha;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentationUnsignedBoneWithPrior, Object);
          

        void SetDeformationPrior(ConstImagePointerType deformedSegmentation){
            this->m_deformationPrior=deformedSegmentation;
        }
          
        void SetAlpha(double alpha){this->m_alpha=alpha;}     
          
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            double origPotential=Superclass::getPotential(fixedIndex,segmentationLabel);
            double priorPotential=0;
            if (this->m_deformationPrior)
                priorPotential=(segmentationLabel!=this->m_deformationPrior->GetPixel(fixedIndex));
            return origPotential+m_alpha*priorPotential;
        }
    };
}//namespace
#endif /* POTENTIALS_H_ */
