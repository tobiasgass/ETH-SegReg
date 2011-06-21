/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _REGISTRATIONPAIRWISEPOTENTIAL_H_
#define _REGISTRATIONPAIRWISEPOTENTIAL_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "itkVector.h"
#include "itkLinearInterpolateImageFunction.h"

namespace itk{



    template<class TLabelMapper,class TImage>
    class PairwisePotentialRegistration{
    public:
        //itk declarations
        typedef PairwisePotentialRegistration            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        SizeType m_fixedSize,m_movingSize;
    protected:
        ImagePointerType m_fixedImage, m_movingImage;
        InterpolatorPointerType m_movingInterpolator;
        SpacingType m_displacementFactor;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotential, Object);

        PairwisePotentialRegistration(){
            m_displacementFactor=1.0;
            m_haveLabelMap=false;
        }
        virtual void freeMemory(){
        }
        void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;m_haveLabelMap=true;}
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
        void SetMovingInterpolator(InterpolatorPointerType movingImage){
            m_movingInterpolator=movingImage;
        }
    	void SetMovingImage(ImagePointerType movingImage){
            m_movingImage=movingImage;
            m_movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
        }
        void SetFixedImage(ImagePointerType fixedImage){
            m_fixedImage=fixedImage;
            m_fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
        }
        void SetDisplacementFactor(const SpacingType & f){m_displacementFactor=f;}
        
        virtual double getPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement1, LabelType displacement2){
            double result=0;
            ContinuousIndexType idx2(fixedIndex);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;
            idx2+= disp;
            itk::Vector<float,ImageType::ImageDimension> baseDisp=m_baseLabelMap->GetPixel(fixedIndex);
            idx2+=baseDisp;
            if (m_movingInterpolator->IsInsideBuffer(idx2)){
                result=fabs(this->m_fixedImage->GetPixel(fixedIndex)-m_movingInterpolator->EvaluateAtContinuousIndex(idx2));
            }else{
                result=999999;
            }
            return result;
        }
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
