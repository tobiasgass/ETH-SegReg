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
    class PairwisePotentialRegistration : public itk::Object{
    public:
        //itk declarations
        typedef PairwisePotentialRegistration            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
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
        ConstImagePointerType m_fixedImage, m_movingImage;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotential, Object);

        PairwisePotentialRegistration(){
            m_haveLabelMap=false;
        }
        virtual void freeMemory(){
        }
        void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;m_haveLabelMap=true;}
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
        void SetFixedImage(ConstImagePointerType fixedImage){
            m_fixedImage=fixedImage;
            m_fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
        }
        
        virtual double getPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement1, LabelType displacement2){
            assert(m_haveLabelMap);
            double result=0;
            //            std::cout<<fixedIndex1<<" "<<fixedIndex2<<" "<<displacement1<<" "<<displacement2<<std::endl;
            return result;
        }
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
