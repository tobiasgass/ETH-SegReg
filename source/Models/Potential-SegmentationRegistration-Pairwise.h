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
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;


    protected:
        ConstImagePointerType m_fixedImage, m_movingImage;
        SegmentationInterpolatorPointerType m_movingSegmentationInterpolator;
        FloatImageInterpolatorPointerType m_movingDistanceTransformInterpolator, m_movingBackgroundDistanceTransformInterpolator;
        ImageInterpolatorPointerType  m_movingInterpolator;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        double m_asymm;
        FloatImagePointerType m_distanceTransform;
        double sigma1, sigma2, mean1, mean2, m_threshold;
        int m_nSegmentationLabels;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationRegistration, Object);

        PairwisePotentialSegmentationRegistration(){
            m_haveLabelMap=false;
            m_asymm=1;
            m_threshold=9999999999.0;
        }
        virtual void freeMemory(){
        }
        void SetNumberOfSegmentationLabels(int n){m_nSegmentationLabels=n;}
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
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(dt);
            filter->Update();
            sigma1=filter->GetSigma();
            mean1=filter->GetMean();
            cout<<"distance transform main segmentation sigma :"<<sigma1<<endl;

        }
        void SetBackgroundDistanceTransform(FloatImagePointerType dt){
            m_movingBackgroundDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            m_movingBackgroundDistanceTransformInterpolator->SetInputImage(dt);
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(dt);
            filter->Update();
            sigma2=filter->GetSigma();
            mean2=fabs(filter->GetMean());
            cout<<"distance transform background segmentation sigma :"<<sigma2<<endl;
        }
        FloatImagePointerType GetDistanceTransform(){return  m_distanceTransform;}
        
        void SetThreshold(double t){m_threshold=t;}

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
        inline virtual  double getBackwardPotential(IndexType fixedIndex1, IndexType fixedIndex2,LabelType displacement, int segmentationLabel){
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
            //deformedAtlasSegmentation=(m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2))>0;
            deformedAtlasSegmentation=int(m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2));
            //            ContinuousIndexType idx3(fixedIndex2);

            //cout<<m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx3)<<" "<<m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)<<" "<<(m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0)<<" "<<int(m_movingSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0)<<endl;

     
#if 0
mean1            if (deformedAtlasSegmentation!=segmentationLabel)
                result=1;
            else
                result=0;
#else
            result=0.0;
#endif           
#if 1
            if (deformedAtlasSegmentation>0){
                
               if (segmentationLabel== 0 && deformedAtlasSegmentation == m_nSegmentationLabels - 1){
                    //distanceToDeformedSegmentation= 1;
                    distanceToDeformedSegmentation=m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    result=fabs(distanceToDeformedSegmentation)/((sigma1+sigma2)/2);//1;

                }else if (segmentationLabel == 0 && deformedAtlasSegmentation ){
                    //distanceToDeformedSegmentation= 1;//m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    distanceToDeformedSegmentation= m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    result=fabs(distanceToDeformedSegmentation)/((sigma1+sigma2)/2);//2;

                }
                

               else if (deformedAtlasSegmentation!=segmentationLabel){
                   result=fabs(m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2))/((sigma1+sigma2)/2)+fabs(m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2))/((sigma1+sigma2)/2);
                }


            }else{
                if (segmentationLabel== m_nSegmentationLabels - 1){
                    //distanceToDeformedSegmentation= 1;
                    distanceToDeformedSegmentation=fabs(m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2));
#if 1       
                    if (distanceToDeformedSegmentation>m_threshold)
                        result=99999999999;
                    else
#endif
                        result=(distanceToDeformedSegmentation)/((sigma1+sigma2)/2);//1;

                }else if (segmentationLabel ){
                    //distanceToDeformedSegmentation= 1;//m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    distanceToDeformedSegmentation= fabs(m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2));
                    distanceToDeformedSegmentation=min(m_threshold,distanceToDeformedSegmentation);
                    result=(distanceToDeformedSegmentation)/((sigma1+sigma2)/2);//2;

                }
            }
#else
            if (deformedAtlasSegmentation>0){
                
               if (segmentationLabel== 0 && deformedAtlasSegmentation == m_nSegmentationLabels - 1){
                    //distanceToDeformedSegmentation= 1;
                    distanceToDeformedSegmentation=m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    result=fabs(distanceToDeformedSegmentation)/((sigma1));//1;

                }else if (segmentationLabel == 0 && deformedAtlasSegmentation ){
                    //distanceToDeformedSegmentation= 1;//m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    distanceToDeformedSegmentation= m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    result=fabs(distanceToDeformedSegmentation)/((sigma2));//2;

                }
               else if (deformedAtlasSegmentation!=segmentationLabel){
                   result=max(fabs(m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2))/((sigma1)),fabs(m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2))/((sigma2)));
                }


            }else{
                if (segmentationLabel== m_nSegmentationLabels - 1){
                    //distanceToDeformedSegmentation= 1;
                    distanceToDeformedSegmentation=m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    result=fabs(distanceToDeformedSegmentation)/((sigma1));//1;

                }else if (segmentationLabel ){
                    //distanceToDeformedSegmentation= 1;//m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    distanceToDeformedSegmentation= m_movingBackgroundDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);
                    result=fabs(distanceToDeformedSegmentation)/((sigma2));//2;

                }
            }
#endif

            return result;
        }
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
