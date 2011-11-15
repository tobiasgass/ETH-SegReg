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
        static const unsigned int D=ImageType::ImageDimension;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename itk::Vector<double,ImageType::ImageDimension> SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        SizeType m_fixedSize,m_movingSize;
    protected:
        ConstImagePointerType m_fixedImage, m_movingImage;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        SpacingType m_gridSpacing;
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
        void SetSpacing(SpacingType  sp) {m_gridSpacing=sp;}
        virtual double getPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement1, LabelType displacement2){
            assert(m_haveLabelMap);
            double result=0;
            

			LabelType oldl1=m_baseLabelMap->GetPixel((fixedIndex1));
			LabelType oldl2=m_baseLabelMap->GetPixel((fixedIndex2));
			double d1,d2;
			int delta;
			displacement1+=oldl1;
			displacement2+=oldl2;

			for (unsigned int d=0;d<D;++d){

				d1=displacement1[d];
				d2=displacement2[d];
				delta=(fixedIndex2[d]-fixedIndex1[d]);
                
				double axisPositionDifference=1.0*(d2-d1);//(m_gridSpacing[d]);

				result+=(axisPositionDifference)*(axisPositionDifference);
			}

			//			if (false){
            
            return (result);
        }
    };//class
    template<class TLabelMapper,class TImage>
    class PairwisePotentialRegistrationSigmoid : public PairwisePotentialRegistration<TLabelMapper,TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialRegistrationSigmoid            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const unsigned int D=ImageType::ImageDimension;
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
  
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotentialSigmoid, Object);

   
        
        virtual double getPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement1, LabelType displacement2){
            assert(this->m_haveLabelMap);
            double result=0;
            

			LabelType oldl1=this->m_baseLabelMap->GetPixel((fixedIndex1));
			LabelType oldl2=this->m_baseLabelMap->GetPixel((fixedIndex2));
			double d1,d2;
			double delta;
			displacement1+=oldl1;
			displacement2+=oldl2;
			for (unsigned int d=0;d<D;++d){

				d1=displacement1[d];
				d2=displacement2[d];
				delta=(fixedIndex2[d]-fixedIndex1[d]);
                if (delta>0){
                    double axisPositionDifference=((1.0*fabs(d2-d1)-delta/3)/(delta/5));//(this->m_spacing[d]);
                    result+=(axisPositionDifference);
                }
			}

			//			if (false){
            //cout<<displacement1-displacement2<<" "<<1.0/(1+exp(-result))<<endl;
            return 1.0/(1+exp(-result));
        }
    };//class
}//namespace
#endif /* POTENTIALS_H_ */
