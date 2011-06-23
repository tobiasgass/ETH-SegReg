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
    protected:
        ConstImagePointerType m_fixedImage, m_sheetnessImage;
        SpacingType m_displacementFactor;
        //LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialSegmentation, Object);

        UnaryPotentialSegmentation(){
            m_haveLabelMap=false;
        }
        virtual void freeMemory(){
        }
        //        void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;m_haveLabelMap=true;}
        //LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}

        void SetFixedImage(ConstImagePointerType fixedImage){
            m_fixedImage=fixedImage;
            m_fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
        }
        void SetGradientImage(ConstImagePointerType sheetnessImage){
            m_sheetnessImage=sheetnessImage;
        }
        
        virtual double getPotential(IndexType fixedIndex, int segmentationLabel){
            int s= m_sheetnessImage->GetPixel(fixedIndex);
            double imageIntensity=m_fixedImage->GetPixel(fixedIndex);
            double segmentationProb=1;
            switch (segmentationLabel) {
            case 1  :
                segmentationProb = (imageIntensity < -500 ) ? 1 : 0;
                break;
            case 0:
                segmentationProb = ( imageIntensity > 400) && ( s > 0 ) ? 1 : 0;
                break;
            default:
                assert(false);
            }
            //        std::cout<<fixedIndex<<" "<<segmentationLabel<<" " << imageIntensity <<" "<<segmentationProb<<std::endl;
            return segmentationProb;
        }

        virtual double getWeight(IndexType idx1, IndexType idx2){
            int s1=m_sheetnessImage->GetPixel(idx1);
            int s2=m_sheetnessImage->GetPixel(idx2);
            double edgeWeight=fabs(s1-s2);
            edgeWeight=(s1 < s2) ? 1.0 : exp ( - 0.05 * edgeWeight);
            //edgeWeight+=1;
            return edgeWeight;
        }
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
