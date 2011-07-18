/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _REGISTRATIONUNARYPOTENTIAL_H_
#define _REGISTRATIONUNARYPOTENTIAL_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "itkVector.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkConstNeighborhoodIterator.h"
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>

namespace itk{



    template<class TLabelMapper,class TImage>
    class UnaryPotentialRegistrationNCC : public itk::Object{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCC            Self;
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

        typedef typename LabelMapperType::LabelImageType LabelImageType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;

        SizeType m_fixedSize,m_movingSize;
    protected:
        ConstImagePointerType m_fixedImage, m_movingImage;
        ConstImagePointerType m_scaledFixedImage, m_scaledMovingImage;
        InterpolatorPointerType m_movingInterpolator;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        bool radiusSet;
        RadiusType m_radius;
        ImageNeighborhoodIteratorType nIt;
        double m_scale;
        SizeType m_scaleITK,m_invertedScaleITK;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialNCC, Object);

        UnaryPotentialRegistrationNCC(){
            m_haveLabelMap=false;
            radiusSet=false;
            m_fixedImage=NULL;
            m_movingImage=NULL;
            m_scale=1.0;
            m_scaleITK.Fill(1.0);
        }
        ~UnaryPotentialRegistrationNCC(){
            //delete nIt;
        }
        void Init(){
            assert(m_fixedImage);
            assert(m_movingImage);
            if (m_scale!=1.0){
                m_scaledFixedImage=FilterUtils<ImageType>::LinearResample((m_fixedImage),m_scale);
                m_scaledMovingImage=FilterUtils<ImageType>::LinearResample((m_movingImage),m_scale);
            }else{
                m_scaledFixedImage=m_fixedImage;
                m_scaledMovingImage=m_movingImage;
            }
            assert(radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_radius[d]*=m_scale;
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            nIt=ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            m_movingInterpolator=InterpolatorType::New();
            m_movingInterpolator->SetInputImage(m_scaledMovingImage);
        }
        
        virtual void freeMemory(){
        }
        void SetScale(double s){
            m_scale=s;
            m_scaleITK.Fill(s); 
            m_invertedScaleITK.Fill(1.0/s);
        }
        void SetRadius(SpacingType sp){
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_radius[d]=sp[d];
            }
            radiusSet=true;
        }
        void SetRadius(RadiusType sp){
            m_radius=sp;
            radiusSet=true;
        }
#define RESAMPLEDLABELMAP
        void SetBaseLabelMap(LabelImagePointerType blm){
            m_baseLabelMap=blm;m_haveLabelMap=true;
#ifdef RESAMPLEDLABELMAP
            if (m_scale!=1.0){
                typedef typename itk::VectorLinearInterpolateImageFunction<LabelImageType> InterpolatorType;
                typename InterpolatorType::Pointer interpol=InterpolatorType::New();
                typedef typename itk::VectorResampleImageFilter<LabelImageType,LabelImageType> ResampleFilterType;
                typename ResampleFilterType::Pointer resampler=ResampleFilterType::New();
                resampler->SetInput(blm);
                resampler->SetInterpolator(interpol);
                typename LabelImageType::SpacingType spacing,inputSpacing;
                typename LabelImageType::SizeType size,inputSize;
                typename LabelImageType::PointType origin,inputOrigin;
                inputOrigin=blm->GetOrigin();
                inputSize=blm->GetLargestPossibleRegion().GetSize();
                inputSpacing=blm->GetSpacing();
                for (uint d=0;d<LabelImageType::ImageDimension;++d){
                    size[d]=int(inputSize[d]*m_scale);
                    spacing[d]=inputSpacing[d]*(1.0*inputSize[d]/size[d]);
                    origin[d]=inputOrigin[d]+0.5*spacing[d]/inputSpacing[d];
                }
                resampler->SetOutputOrigin(origin);
                resampler->SetOutputSpacing ( spacing );
                resampler->SetOutputDirection ( blm->GetDirection() );
                resampler->SetSize ( size );
                resampler->Update();
                m_baseLabelMap=resampler->GetOutput();
            }
#endif
        }
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
      
    	void SetMovingImage(ConstImagePointerType movingImage){
            m_movingImage=movingImage;
            m_movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
        }
        void SetFixedImage(ConstImagePointerType fixedImage){
            m_fixedImage=fixedImage;
            m_fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
           
        }

        virtual double getPotential(IndexType fixedIndex, LabelType disp){
            double result=0;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<" "<<m_scale<<"\t";
#ifdef RESAMPLEDLABELMAP
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=m_scale;
            }
            LabelType baseDisp=m_baseLabelMap->GetPixel(fixedIndex);
#else
            LabelType baseDisp=m_baseLabelMap->GetPixel(fixedIndex);
            fixedIndex=fixedIndex*m_scaleITK;
#endif
            baseDisp*=m_scale;
            disp*=m_scale;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(fixedIndex);
            nIt.SetLocation(fixedIndex);
            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            for (unsigned int i=0;i<nIt.Size();++i){
                    bool inBounds;
                    double f=nIt.GetPixel(i,inBounds);
                    if (inBounds){
                        IndexType neighborIndex=nIt.GetIndex(i);
                        IndexType scaledNI=neighborIndex*m_invertedScaleITK;
                        //this should be weighted somehow
                        ContinuousIndexType idx2(neighborIndex);
                        double weight=1.0;
#ifdef RESAMPLEDLABELMAP
                        idx2+=disp+this->m_baseLabelMap->GetPixel(neighborIndex)*m_scale;
#else
                        idx2+=disp+this->m_baseLabelMap->GetPixel(scaledNI)*m_scale;
#endif
                        //                    cout<<fixedIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
                        double m;
                        if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
                            continue;
                            m=0;
#if 0
                            for (int d=0;d<ImageType::ImageDimension;++d){
                                if (idx2[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
                                    idx2[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
                                }
                                else if (idx2[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
                                    idx2[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
                                }
                            }
#endif
                        }else{
                            m=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
                        }
                        //       cout<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                        sff+=f*f;
                        smm+=m*m;
                        sfm+=f*m;
                        sf+=f;
                        sm+=m;
                        count+=1;
                    }

                }
                     //cout<<"doneit"<<endl;
                     if (count){
                         //                cout<<" "<<sff<<" "<<sfm<<" "<<smm<<" "<<sf<<" "<<sm<<" "<<count<<" "<<sff- ( sf * sf / count )<<endl;

                         sff -= ( sf * sf / count );
                         smm -= ( sm * sm / count );
                         sfm -= ( sf * sm / count );
                         //                cout<<" "<<sff<<" "<<sfm<<" "<<smm<<" "<<sf<<" "<<sm<<" "<<count<<" "<<sff- ( sf * sf / count )<<endl;
                         double result;
                         if (smm*sff>0){
                             result=1-(1.0*sfm/sqrt(smm*sff)/2);
                             //  result=(1-fabs(1.0*sfm/sqrt(smm*sff)));
                             //  result=(1-(1.0*sfm/sqrt(smm*sff)+1.0)/2);

                         }
                         else if (sfm>0)result=0;
                         else result=1;
                     }
                     //no correlation whatsoever
                     else result=0.5;
                 //result=result>0.5?0.5:result;
                 return result;
                 }
        };//class

        template<class TLabelMapper,class TImage>
        class UnaryPotentialRegistrationSAD : public UnaryPotentialRegistrationNCC<TLabelMapper, TImage>{
        public:
            //itk declarations
            typedef UnaryPotentialRegistrationSAD           Self;
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
            typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
            typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;

        public:
            /** Method for creation through the object factory. */
            itkNewMacro(Self);
            /** Standard part of every itk Object. */
            itkTypeMacro(RegistrationUnaryPotentialSAD, Object);

            UnaryPotentialRegistrationSAD(){}
        
            virtual double getPotential(IndexType fixedIndex, LabelType disp){
                double result=0;
                itk::Vector<float,ImageType::ImageDimension> baseDisp=this->m_baseLabelMap->GetPixel(fixedIndex);
                this->nIt.SetLocation(fixedIndex);
                double count=0;
                double sum=0.0;
                for (unsigned int i=0;i<this->nIt.Size();++i){
                    bool inBounds;
                    double f=this->nIt.GetPixel(i,inBounds);
                    if (inBounds){
                        IndexType neighborIndex=this->nIt.GetIndex(i);
                        //this should be weighted somehow
                        ContinuousIndexType idx2(neighborIndex);
                        double weight=1.0;
                        idx2+=disp+this->m_baseLabelMap->GetPixel(neighborIndex);
                        //                    cout<<fixedIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
                        double m;
                        if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
                            continue;
                            m=0;
                        }else{
                            m=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
                        }
                        result+=fabs(m-f);
                        count+=1;
                    }

                }

                if (count)
                    return result/count;
                else
                    return 999999999;
            }
        };//class
    }//namespace
#endif /* POTENTIALS_H_ */
