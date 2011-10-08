
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
#include "Potential-SegmentationRegistration-Pairwise.h"
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
            this->m_scale=s;
            this->m_scaleITK.Fill(s); 
            this->m_invertedScaleITK.Fill(1.0/s);
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
        void SetBaseLabelMap(LabelImagePointerType blm){
            m_baseLabelMap=blm;m_haveLabelMap=true;
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

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=m_scale;
            }
            LabelType baseDisp=m_baseLabelMap->GetPixel(fixedIndex);
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
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
                    //this should be weighted somehow
                    ContinuousIndexType idx2(neighborIndex);
                    //double weight=1.0;

                    idx2+=disp+this->m_baseLabelMap->GetPixel(neighborIndex)*m_scale;

                    //cout<<fixedIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
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
                    //cout<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;
                }

            }
            if (count){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    //result=(1-1.0*sfm/sqrt(smm*sff))/2;
                    result=((1+1.0*sfm/sqrt(smm*sff))/2);
                    result=result>0?result:0.00000001;
                    result=-log(result);
                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                }
            }
            //no correlation whatsoever
            else result=-log(0.5);
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

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=this->m_scale;
            }
            LabelType baseDisp=this->m_baseLabelMap->GetPixel(fixedIndex);
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
            baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(fixedIndex);
            this->nIt.SetLocation(fixedIndex);
            double count=0;
            //double sum=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double f=this->nIt.GetPixel(i,inBounds);
                if (inBounds){
                    IndexType neighborIndex=this->nIt.GetIndex(i);
                    //this should be weighted somehow
                    ContinuousIndexType idx2(neighborIndex);
                    //double weight=1.0;

                    idx2+=disp+this->m_baseLabelMap->GetPixel(neighborIndex)*this->m_scale;
                    
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
    
    template<class TLabelMapper,class TImage>
    class UnaryPotentialRegistrationNCCWithSegmentationPrior : public UnaryPotentialRegistrationNCC<TLabelMapper, TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCCWithSegmentationPrior           Self;
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
        typedef NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;

        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        typedef PairwisePotentialSegmentationRegistration<TImage> SRSPotentialType;
        typedef typename SRSPotentialType::Pointer SRSPotentialPointerType;
        
    private:
        ConstImagePointerType m_segmentationPrior, m_atlasSegmentation;
        double m_alpha,m_beta;
        NNInterpolatorPointerType m_segPriorInterpolator,m_atlasSegmentationInterpolator;
        SRSPotentialPointerType m_srsPotential;
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(UnaryPotentialRegistrationNCCWithSegmentationPrior, Object);

      
        
        void SetSegmentationPrior(ConstImagePointerType prior){
            if (prior){
                if (this->m_scale!=1.0){
                    m_segmentationPrior=FilterUtils<ImageType>::NNResample((prior),this->m_scale);
                }else{
                    m_segmentationPrior=prior;  
            
                }
            }
        }
     
        void SetSRSPotential(SRSPotentialPointerType pot){m_srsPotential=pot;}
        void SetAlpha(double alpha){m_alpha=alpha;}
        void SetBeta(double beta){m_beta=beta;}
        
        virtual double getPotential(IndexType fixedIndex, LabelType disp){
            double result=0;
          
            LabelType trueDisplacement=disp;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=this->m_scale;
            }
            LabelType baseDisp=this->m_baseLabelMap->GetPixel(fixedIndex);
            //disp+=baseDisp;
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
            baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(fixedIndex);
            this->nIt.SetLocation(fixedIndex);
            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            double segmentationPenalty=0.0,distanceSum=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double f=this->nIt.GetPixel(i,inBounds);
                if (inBounds){
                    IndexType neighborIndex=this->nIt.GetIndex(i);
                    //this should be weighted somehow
                    ContinuousIndexType idx2(neighborIndex);
                    //double weight=1.0;
                    LabelType baseDisplacement=this->m_baseLabelMap->GetPixel(neighborIndex);
                    LabelType finalDisplacement=disp+baseDisplacement*this->m_scale;
                    
                    idx2+=finalDisplacement;

                    //cout<<fixedIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
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
                    //cout<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;
                    if (this->m_segmentationPrior && this->m_alpha>0){
                        double weight=1.0;
                        IndexType trueIndex=neighborIndex;
                        for (unsigned int d=0;d<D;++d){
                            weight*=1.0-fabs((1.0*fixedIndex[d]-neighborIndex[d])/(this->m_radius[d]));
                            trueIndex[d]/=this->m_scale;
                        }
                        int segmentationPriorLabel=(this->m_segmentationPrior->GetPixel(neighborIndex));
                        //double penalty=weight*this->m_srsPotential->getPotential(neighborIndex,neighborIndex,disp,segmentationPriorLabel);
                        double penalty=weight*this->m_srsPotential->getPotential(trueIndex,trueIndex,trueDisplacement+baseDisplacement,segmentationPriorLabel);
                        segmentationPenalty+=penalty;
                        //cout<<fixedIndex<<" "<<neighborIndex<<" "<<weight<<" "<<segmentationPriorLabel<<" "<<penalty<<endl;
                        distanceSum+=weight;
                    }
                }

            }
            if (count){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    //result=(1-1.0*sfm/sqrt(smm*sff))/2;
                    result=((1+1.0*sfm/sqrt(smm*sff))/2);
                    result=result>0?result:0.00000001;
                    result=-log(result);
                    if (distanceSum){
                        result=result+this->m_alpha*segmentationPenalty/distanceSum;
                    }
                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                }
                result=m_beta*result+this->m_alpha*segmentationPenalty/distanceSum;

            }
            //no correlation whatsoever
            else result=-log(0.5);
            //result=result>0.5?0.5:result;
            return result;
        }

    };//class
    template<class TLabelMapper,class TImage>
    class UnaryPotentialRegistrationNCCWithBonePrior : public UnaryPotentialRegistrationNCC<TLabelMapper, TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCCWithBonePrior           Self;
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
        typedef NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;

        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
    private:
        ConstImagePointerType m_targetSheetness, m_atlasSegmentation;
        double m_alpha;
        NNInterpolatorPointerType m_segPriorInterpolator,m_atlasSegmentationInterpolator;
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialWithBonePrior, Object);

        UnaryPotentialRegistrationNCCWithBonePrior(){}
        
      
        void SetAtlasSegmentation(ConstImagePointerType atlas){
            if (this->m_scale!=1.0){
                m_atlasSegmentation=FilterUtils<ImageType>::LinearResample((atlas),this->m_scale);
            }else{
                m_atlasSegmentation=atlas;
            
            }
            m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            m_atlasSegmentationInterpolator->SetInputImage(m_atlasSegmentation);
        }
        void SetAlpha(double alpha){m_alpha=alpha;}
        
        int getSegmentationCost(int deformedSegmentationLabel, double imageIntensity, int s){
            
            int segmentationProb;
            if (deformedSegmentationLabel>0) {
                segmentationProb = (imageIntensity < (-500+1000)*255.0/2000 ) ? 1 : 0;
            }else{
                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000  &&s>0 ) ? 1 : 0;
            }
            return segmentationProb;

        }
          
        void SetTargetSheetness(ConstImagePointerType img){
            if (this->m_scale!=1.0){
                m_targetSheetness=FilterUtils<ImageType>::LinearResample((img),this->m_scale);

            }else{
                m_targetSheetness=img;
            }
        }
        virtual double getPotential(IndexType fixedIndex, LabelType disp){
            double result=0;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=this->m_scale;
            }
            LabelType baseDisp=this->m_baseLabelMap->GetPixel(fixedIndex);
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
            baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(fixedIndex);
            this->nIt.SetLocation(fixedIndex);
            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            double segmentationPenalty=0.0,distanceSum=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double f=this->nIt.GetPixel(i,inBounds);
                if (inBounds){
                    IndexType neighborIndex=this->nIt.GetIndex(i);
                    //this should be weighted somehow
                    ContinuousIndexType idx2(neighborIndex);
                    //double weight=1.0;

                    idx2+=disp+this->m_baseLabelMap->GetPixel(neighborIndex)*this->m_scale;

                    //cout<<fixedIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
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
                    //cout<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;
                    double weight=1.0;
                    for (unsigned int d=0;d<D;++d){
                        weight*=1.0-fabs((1.0*fixedIndex[d]-neighborIndex[d])/(this->m_radius[d]));
                    }
                    weight=1.0;
                    if (this->m_alpha){
                        segmentationPenalty+=weight*getSegmentationCost(this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0,f, m_targetSheetness->GetPixel(neighborIndex));
                        distanceSum+=weight;
                    }
                }

            }
            double thresh=0.4;
            if (count){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    //result=(1-1.0*sfm/sqrt(smm*sff))/2;
                    //result>thresh?thresh:result;
                    //result=-log((1.0*sfm/sqrt(smm*sff)+1)/2);
                    result=((1+1.0*sfm/sqrt(smm*sff))/2);
                    result=result>0?result:0.00000001;
                    result=-log(result);
                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                }
                if (distanceSum)
                    result=result+this->m_alpha*segmentationPenalty/distanceSum;
            }
            //no correlation whatsoever (-log(0.5))
            else result=0.693147;
            //result=result>0.5?0.5:result;
            return result;
        }
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
