/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SEGMENTATIONREGISTRATIONPAIRWISEPOTENTIAL_H_
#define _SEGMENTATIONREGISTRATIONPAIRWISEPOTENTIAL_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "itkVector.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

namespace itk{



    template<class TLabelMapper,class TImage>
    class PairwisePotentialSegmentationRegistration : public itk::Object{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationRegistration            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
        typedef LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

        typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
        typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        SizeType m_fixedSize,m_movingSize;
    protected:
        ConstImagePointerType m_fixedImage, m_movingImage;
        SegmentationInterpolatorPointerType m_movingSegmentationInterpolator;
        FloatImageInterpolatorPointerType m_movingDistanceTransformInterpolator;
        ImageInterpolatorPointerType  m_movingInterpolator;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        double m_asymm;
        FloatImagePointerType m_distanceTransform;
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationRegistration, Object);

        PairwisePotentialSegmentationRegistration(){
            m_haveLabelMap=false;
            m_asymm=1;
        }
        virtual void freeMemory(){
        }
        void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;m_haveLabelMap=true;}
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
        void SetMovingInterpolator(ImageInterpolatorPointerType movingImage){
            m_movingInterpolator=movingImage;
        }
        void SetMovingSegmentationInterpolator(SegmentationInterpolatorPointerType movingSegmentation){
            m_movingSegmentationInterpolator=movingSegmentation;
        }
    	void SetMovingImage(ConstImagePointerType movingImage){
            m_movingImage=movingImage;
            m_movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
        }
        void SetFixedImage(ConstImagePointerType fixedImage){
            m_fixedImage=fixedImage;
            m_fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
        }
        void SetAsymmetryWeight(double as){
            m_asymm=1-as;
        }
        void SetDistanceTransform(FloatImagePointerType dt){
            m_distanceTransform=dt;
            m_movingDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            m_movingDistanceTransformInterpolator->SetInputImage(dt);
        }
        FloatImagePointerType GetDistanceTransform(){return  m_distanceTransform;}

#if 0        //edge from  segmentation to Registration
        virtual double getPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2(fixedIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;
            idx2+= disp;
            itk::Vector<float,ImageType::ImageDimension> baseDisp=m_baseLabelMap->GetPixel(fixedIndex2);
            idx2+=baseDisp;
            int deformedAtlasSegmentation=-1;
            double distanceToDeformedSegmentation;
            if (!m_movingInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            }
            deformedAtlasSegmentation=floor(m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)+0.5);
            distanceToDeformedSegmentation= fabs(m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2));
            result=0;
            if (segmentationLabel){
                if (deformedAtlasSegmentation!=segmentationLabel){
                    result=m_asymm;//*(1+distanceToDeformedSegmentation);
                }
            }else{
                if (deformedAtlasSegmentation){
                    result=1;//distanceToDeformedSegmentation;
                }
            }
          
            //                result=1-(segmentationLabel==deformedAtlasSegmentation);
            //            cout<<fixedIndex1<<" "<<fixedIndex2<<" "<<displacement<<" "<<segmentationLabel<<" "<<deformedAtlasSegmentation<<" "<<idx2<<" "<<result<<endl;
            return result;
        }
#endif
        //edge from registration to segmentation
        virtual double getBackwardPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2(fixedIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;
            idx2+= disp;
            itk::Vector<float,ImageType::ImageDimension> baseDisp=m_baseLabelMap->GetPixel(fixedIndex2);
            idx2+=baseDisp;
            int deformedAtlasSegmentation=-1;
            double distanceToDeformedSegmentation;
            if (!m_movingInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            }
            deformedAtlasSegmentation=floor(m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)+0.5);
            distanceToDeformedSegmentation= m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
            //std::cout<<deformedAtlasSegmentation<<std::endl;
            result=0;
            if (deformedAtlasSegmentation>0){
                if (deformedAtlasSegmentation!=segmentationLabel){
                    result=1;//*fabs(distanceToDeformedSegmentation);
                    //result=fabs(distanceToDeformedSegmentation);
                }
            }else{
                if (segmentationLabel==2){
                    result=fabs(distanceToDeformedSegmentation);
                }else if (segmentationLabel){
                    result=1;//+fabs(distanceToDeformedSegmentation);
                    //result=fabs(distanceToDeformedSegmentation);
                }
            }
           
            //cout<<fixedIndex1<<" "<<fixedIndex2<<" "<<displacement<<" "<<segmentationLabel<<" "<<deformedAtlasSegmentation<<" "<<idx2<<" "<<result<<endl;
            return result;
        }
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
