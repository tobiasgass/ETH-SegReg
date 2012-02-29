
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
#include "itkTranslationTransform.h"
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
        RadiusType m_radius, m_scaledRadius;
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
        virtual void Init(){
            assert(m_fixedImage);
            assert(m_movingImage);
            if (m_scale!=1.0){
                m_scaledFixedImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_fixedImage,100),m_scale);
                m_scaledMovingImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_movingImage,100),m_scale);
            }else{
                m_scaledFixedImage=m_fixedImage;
                m_scaledMovingImage=m_movingImage;
            }
            assert(radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_scaledRadius[d]=m_radius[d]*m_scale;
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            cout<<" Radius " << m_radius << " scale "<< m_scale << "scaledRadius "<< m_scaledRadius << endl;
            nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
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
                m_scaledRadius[d]=this->m_scale*sp[d];
            
            }
            radiusSet=true;
        }
        void SetRadius(RadiusType sp){
            m_radius=sp;
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_scaledRadius[d]=this->m_scale*sp[d];
            }
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
      
    	virtual void SetMovingImage(ConstImagePointerType movingImage){
            m_movingImage=movingImage;
            if (m_scale!=1.0){
                m_scaledMovingImage=FilterUtils<ImageType>::LinearResample((m_movingImage),m_scale);
            }else{
                m_scaledMovingImage=m_movingImage;
            }
            m_movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
            m_movingInterpolator=InterpolatorType::New();
            m_movingInterpolator->SetInputImage(m_scaledMovingImage);
        }
        void SetFixedImage(ConstImagePointerType fixedImage){
            m_fixedImage=fixedImage;
            m_fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
            if (m_scale!=1.0){
                m_scaledFixedImage=FilterUtils<ImageType>::LinearResample((m_fixedImage),m_scale);
            }else{
                m_scaledFixedImage=m_fixedImage;
            }
            cout << this->m_scaledRadius <<endl;
            nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());

        }

        virtual double getPotential(IndexType fixedIndex, LabelType disp){
            double result=0;
            IndexType idx1=fixedIndex;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=m_scale;
                if (fixedIndex[d]>=(int)m_scaledMovingImage->GetLargestPossibleRegion().GetSize()[d]) fixedIndex[d]--;
            }
          
            // LabelType baseDisp=m_baseLabelMap->GetPixel(fixedIndex);
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
            //baseDisp*=m_scale;
            disp*=m_scale;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(fixedIndex);
            nIt.SetLocation(fixedIndex);
            double count=0, totalCount=0;
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
                    totalCount+=1.0;
                    if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
#if 0
                        continue;
                        m=0;
                        
#else
                        for (int d=0;d<ImageType::ImageDimension;++d){
                            double d1=idx2[d]-this->m_movingInterpolator->GetEndContinuousIndex()[d];
                            if (d1>0){
                                idx2[d]-=2*d1;
                            }
                            else {
                                double d2=this->m_movingInterpolator->GetStartContinuousIndex()[d]-idx2[d];
                                if (d2<0){                                    
                                    idx2[d]-=2*d2;
                                }
                            }
                        }
                        m=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
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
          
            if (1.0*count/totalCount<0.0001)
                result=0;//100000000;//-log(0.0000000000000000001);{
            else{
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    //result=(1-1.0*sfm/sqrt(smm*sff))/2;
                    result=((1+1.0*sfm/sqrt(sff*smm))/2);
                    result=result>0.00000001?result:0.00000001;
                    result=-log(result);
                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                }
            }
            return result;
        }
        virtual double GetOverlapRatio(IndexType fixedIndex){
            double result=0;
            IndexType idx1=fixedIndex;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=m_scale;
                if (fixedIndex[d]>=(int)m_scaledMovingImage->GetLargestPossibleRegion().GetSize()[d]) fixedIndex[d]--;
            }
          
            nIt.SetLocation(fixedIndex);
            double count=0;
            int totalCount=0;
            for (unsigned int i=0;i<nIt.Size();++i){
                bool inBounds;
                nIt.GetPixel(i,inBounds);
                if (inBounds){
                    IndexType neighborIndex=nIt.GetIndex(i);
                    //this should be weighted somehow
                    ContinuousIndexType idx2(neighborIndex);
                    //double weight=1.0;
                    idx2+=this->m_baseLabelMap->GetPixel(neighborIndex)*m_scale;
                    if (this->m_movingInterpolator->IsInsideBuffer(idx2)){
                        count+=1;
                    }
                    totalCount++;
                }

            }
#if 0
            //overlap ratio
            if (totalCount){
                result=count/totalCount;
            }else{result=0;}
#else
            //patch size ratio [decreases weight for border and corner patches, pro bably in {1,0.5,0.25}
            result=1.0*totalCount/nIt.Size();
#endif
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
                        //continue;
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
                        distanceSum+=1;//weight;
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
                    result=m_beta*(-1.0)*log(result);
                    if (distanceSum){
                        result+=this->m_alpha*segmentationPenalty/distanceSum;
                        
                    }
                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                }
              

            }
            //no correlation whatsoever
            else result=-log(0.0000000000000000001);
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
        ConstImagePointerType m_targetSheetness, m_scaledTargetSheetness, m_atlasSegmentation, m_scaledAtlasSegmentation;
        double m_alpha;
        NNInterpolatorPointerType m_segPriorInterpolator,m_atlasSegmentationInterpolator;
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialWithBonePrior, Object);

        UnaryPotentialRegistrationNCCWithBonePrior(){}
        
      
        void SetAtlasSegmentation(ConstImagePointerType atlas){
            m_atlasSegmentation=atlas;
            
            if (this->m_scale!=1.0){
                m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((atlas),this->m_scale);
            }else{
                m_scaledAtlasSegmentation=atlas;
            
            }
            m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            m_atlasSegmentationInterpolator->SetInputImage(m_scaledAtlasSegmentation);
        }
        void SetAlpha(double alpha){m_alpha=alpha;}
        
        double getSegmentationCost(int deformedSegmentationLabel, double imageIntensity, int s){
            
            int segmentationProb;
            int tissue=(-500+1000)*255.0/2000;
            if (deformedSegmentationLabel>0) {
                segmentationProb = (imageIntensity < tissue) ? 1:0;
            }else{
                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000  &&s>0 ) ? 1 : 0;
            }
            return segmentationProb;

        }
          
        void SetTargetSheetness(ConstImagePointerType img){
            m_targetSheetness=img;
            if (this->m_scale!=1.0){
                m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((img),this->m_scale);

            }else{
                m_scaledTargetSheetness=img;
            }
        }
        virtual void Init(){
            assert(this->m_fixedImage);
            assert(this->m_movingImage);
            assert(this->m_targetSheetness);
            assert(this->m_atlasSegmentation);
            if (this->m_scale!=1.0){
                this->m_scaledFixedImage=FilterUtils<ImageType>::LinearResample((this->m_fixedImage),this->m_scale);
                this->m_scaledMovingImage=FilterUtils<ImageType>::LinearResample((this->m_movingImage),this->m_scale);
                this->m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((m_atlasSegmentation),this->m_scale);
                this->m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((m_targetSheetness),this->m_scale);
            }
            assert(this->radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=this->m_radius[d]*this->m_scale;
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            this->m_movingInterpolator=InterpolatorType::New();
            this->m_movingInterpolator->SetInputImage(this->m_scaledMovingImage);
            this->m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(this->m_scaledAtlasSegmentation);
        }
        virtual double getPotential(IndexType fixedIndex, LabelType disp){
            double result=0;
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=this->m_scale;
            }
            //LabelType baseDisp=this->m_baseLabelMap->GetPixel(fixedIndex);
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
            //baseDisp*=this->m_scale;
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
                        assert(!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2));
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
                        weight*=1.0-fabs((1.0*fixedIndex[d]-neighborIndex[d])/(this->m_scaledRadius[d]));
                    }
                    if (this->m_alpha){
#if 0
                        if (f>=bone){
                            int seg=this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0.5;
                            if ( !seg){
                                segmentationPenalty+=weight;
                            }
                        }else if ( f<tissue){
                            int seg=this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0.5;
                            if (seg){
                                segmentationPenalty+=weight;
                            }
                        }
                        distanceSum+=weight;   
#else
                        bool movingTissue=m<tissue;
                        bool movingBone=m>bone;
                        bool fixedTissue=f<tissue;
                        bool fixedBone=f>bone;
                        
                        distanceSum+=weight;
                        segmentationPenalty+=weight*( (movingTissue==fixedBone) || (movingBone==fixedTissue));
                        
#endif
                    }
                }

            }
            if (count){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
#if 0//log NCC
                    result=((1+1.0*sfm/sqrt(smm*sff))/2);
                    result=result>0?result:0.00000001;
                    result=-log(result);
#else
                    result=(1-1.0*sfm/sqrt(smm*sff))/2;
#endif                    
      
                    //result>thresh?thresh:result;
                    //result=-log((1.0*sfm/sqrt(smm*sff)+1)/2);
                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                }
                // cout<<fixedIndex<<" "<<segmentationPenalty<<" "<<distanceSum<<endl;
                if (distanceSum){
                    result=(1-this->m_alpha)*result+this->m_alpha*segmentationPenalty/distanceSum;
                }
            }
            //no correlation whatsoever (-log(0.5))
            else result=-log(0.0000000000000000001);//0.693147;
            //result=result>0.5?0.5:result;
            return result;
        }
    };//class
    template<class TLabelMapper,class TImage>
    class UnaryPotentialRegistrationNCCWithDistanceBonePrior : public UnaryPotentialRegistrationNCC<TLabelMapper, TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCCWithDistanceBonePrior           Self;
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
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
        typedef LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;
    private:
        ConstImagePointerType m_targetSheetness, m_scaledTargetSheetness, m_atlasSegmentation, m_scaledAtlasSegmentation;
        double m_alpha;
        NNInterpolatorPointerType m_segPriorInterpolator,m_atlasSegmentationInterpolator;
        FloatImageInterpolatorPointerType m_movingDistanceTransformInterpolator;
        double sigma1, sigma2, mean1, mean2, m_threshold;
        FloatImagePointerType  m_distanceTransform;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialWithBonePrior, Object);

        UnaryPotentialRegistrationNCCWithDistanceBonePrior(){}
        FloatImagePointerType getDistanceTransform(ConstImagePointerType segmentationImage){
            typedef typename itk::SignedMaurerDistanceMapImageFilter< ImageType, FloatImageType > DistanceTransformType;
            typename DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2){
                float val=imageIt.Get();
                imageIt2.Set(val>0);
                
            }
            //distanceTransform->InsideIsPositiveOn();
            distanceTransform->SetInput(newImage);
            distanceTransform->SquaredDistanceOn ();
            distanceTransform->UseImageSpacingOn();
            distanceTransform->Update();
            typedef typename  itk::ImageRegionIterator<FloatImageType> FloatImageIterator;

            FloatImagePointerType positiveDM=distanceTransform->GetOutput();
            FloatImageIterator imageIt3(positiveDM,positiveDM->GetLargestPossibleRegion());        
            for (imageIt3.GoToBegin();!imageIt3.IsAtEnd();++imageIt3){
                imageIt3.Set(fabs(imageIt3.Get()));
            }
            return  positiveDM;
        }
      
        void SetAtlasSegmentation(ConstImagePointerType atlas){
            m_atlasSegmentation=atlas;
            
            if (this->m_scale!=1.0){
                m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((atlas),this->m_scale);
            }else{
                m_scaledAtlasSegmentation=atlas;
            
            }
            m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            m_atlasSegmentationInterpolator->SetInputImage(m_scaledAtlasSegmentation);
            FloatImagePointerType dt1=getDistanceTransform(m_scaledAtlasSegmentation);
            m_movingDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            m_movingDistanceTransformInterpolator->SetInputImage(dt1);
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(dt1);
            filter->Update();
            sigma1=filter->GetSigma();
            mean1=filter->GetMean();
            m_distanceTransform=dt1;
        }
        void SetAlpha(double alpha){m_alpha=alpha;}
        
        double getSegmentationCost(int deformedSegmentationLabel, double imageIntensity, int s){
            
            int segmentationProb;
            int tissue=(-500+1000)*255.0/2000;
            if (deformedSegmentationLabel>0) {
                segmentationProb = (imageIntensity < tissue) ? 1:0;
            }else{
                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000  &&s>0 ) ? 1 : 0;
            }
            return segmentationProb;

        }
          
        void SetTargetSheetness(ConstImagePointerType img){
            m_targetSheetness=img;
            if (this->m_scale!=1.0){
                m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((img),this->m_scale);

            }else{
                m_scaledTargetSheetness=img;
            }
        }
        virtual void Init(){
            assert(this->m_fixedImage);
            assert(this->m_movingImage);
            assert(this->m_targetSheetness);
            assert(this->m_atlasSegmentation);
            if (this->m_scale!=1.0){
                this->m_scaledFixedImage=FilterUtils<ImageType>::LinearResample((this->m_fixedImage),this->m_scale);
                this->m_scaledMovingImage=FilterUtils<ImageType>::LinearResample((this->m_movingImage),this->m_scale);
                this->m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((m_atlasSegmentation),this->m_scale);
                this->m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((m_targetSheetness),this->m_scale);
            }
            assert(this->radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=this->m_radius[d]*this->m_scale;
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledFixedImage, this->m_scaledFixedImage->GetLargestPossibleRegion());
            this->m_movingInterpolator=InterpolatorType::New();
            this->m_movingInterpolator->SetInputImage(this->m_scaledMovingImage);
            this->m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(this->m_scaledAtlasSegmentation);
        }
        virtual double getPotential(IndexType fixedIndex, LabelType disp){
            double result=0;
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=this->m_scale;
            }
            //LabelType baseDisp=this->m_baseLabelMap->GetPixel(fixedIndex);
            //std::cout<<baseDisp<<" "<<disp<<std::endl;
            //baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            std::cout<<fixedIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(fixedIndex);
            this->nIt.SetLocation(fixedIndex);
            int count=0, totalCount=0;
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
                    totalCount++;
                    if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
                        assert(!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2));
                        continue;
                        //m=-50;
                        
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

                    if (false &&this->m_alpha){
                        double weight=1.0;
                        for (unsigned int d=0;d<D;++d){
                            weight*=1.0-fabs((1.0*fixedIndex[d]-neighborIndex[d])/(this->m_scaledRadius[d]));
                        }

                        if (f>=bone){
                            int seg=this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0.5;
                            if ( !seg){
                                double distance=m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2)/sigma1;
                                segmentationPenalty+=weight*distance;
                            }
                        }else if ( f<tissue){
                            int seg=this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0.5;
                            if (seg){
                                //double distance=m_movingDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2)/sigma1;
                                //segmentationPenalty+=weight*distance;
                                segmentationPenalty+=weight;
                            }
                        }
                        distanceSum+=weight;   
                    }
                }

            }
            if (count>1){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    double NCC=sfm/sqrt(smm*sff);
                    //NCC*=1.0*count/totalCount;
                    //NCC*=1.0*totalCount/count;
#if 1
                    result=((1.0+NCC)/2);
                    result=result>0?result:0.00000001;
                    result=-log(result);
#else
                    result=(1-NCC)/2;
#endif
                    //result*=1.0*count/totalCount;
                    //result*=1.0*totalCount/count;

                }
                else {
                    if (sfm>0) result=0;
                    else result=1;
                    //cout<<"AUTOCORRELATION ZERO "<<count<<endl;
                }
                // cout<<fixedIndex<<" "<<segmentationPenalty<<" "<<distanceSum<<endl;
                if (distanceSum){
                    result=result+this->m_alpha*segmentationPenalty/distanceSum;
                }
                //cout<<"result "<<result<<" penalty factor:"<<1+this->m_alpha*(1.0*totalCount-count)/(totalCount)<<" countDiff:"<<totalCount-count<<endl;
                result=result*(1+this->m_alpha*(1.0*totalCount-count)/(totalCount+1));//+this->m_alpha*(totalCount-count)/(totalCount+1);
            }
            //no correlation whatsoever (-log(0.5))
            else result=0;//10000000;//100;//-log(0.0000000000000000001);//0.693147;
            //result=result>0.5?0.5:result;
            return result;
        }
    };//class

    template<class TLabelMapper,class TImage>
    class FastUnaryPotentialRegistrationNCC: public UnaryPotentialRegistrationNCC<TLabelMapper,TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationNCC            Self;
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
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

    protected:
        ImagePointerType m_shiftedMovingImage;
        ImageNeighborhoodIteratorType m_movingNeighborhoodIterator;
        LabelType m_currentDisp;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialNCC, Object);

        virtual void shiftMovingImage(LabelType displacement){
            LabelType scaledLabel=displacement*this->m_scale;
            typename TranslationTransformType::Pointer transform =
                TranslationTransformType::New();
            typename TranslationTransformType::OutputVectorType translation;
            for (int d=0;d<ImageType::ImageDimension;++d){
                //translation[d] =scaledLabel[d];
                translation[d] =displacement[d];
            }
            transform->Translate(translation);
            typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
            resampleFilter->SetTransform(transform.GetPointer());
            resampleFilter->SetDefaultPixelValue(0);
            //translate moving image
            resampleFilter->SetOutputParametersFromImage(this->m_scaledMovingImage);
            resampleFilter->SetInput(this->m_scaledMovingImage);
            resampleFilter->Update();
#if 0
            if (this->m_scale!=1.0){
                m_shiftedMovingImage=FilterUtils<ImageType>::LinearResample((ConstImagePointerType)(resampleFilter->GetOutput()),this->m_scale);
            }else{
                m_shiftedMovingImage=resampleFilter->GetOutput();
            }
#endif
            
            m_shiftedMovingImage=resampleFilter->GetOutput();
            m_movingNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_radius,this->m_shiftedMovingImage, this-> m_shiftedMovingImage->GetLargestPossibleRegion());
            m_currentDisp=scaledLabel;
          
        }
        virtual double getLocalPotential(IndexType fixedIndex){

            double result;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                fixedIndex[d]*=this->m_scale;
                if (fixedIndex[d]>=(int)m_shiftedMovingImage->GetLargestPossibleRegion().GetSize()[d]) fixedIndex[d]--;

            }
            //  if(!( m_shiftedMovingImage->GetLargestPossibleRegion().IsInside(fixedIndex) && this->m_fixedImage->GetLargestPossibleRegion().IsInside(fixedIndex))){
            //cout<<fixedIndex<<" "<<m_shiftedMovingImage->GetLargestPossibleRegion().GetSize()<<" "<<this->m_scaledMovingImage->GetLargestPossibleRegion().GetSize()<<endl;
            //  }
            this->nIt.SetLocation(fixedIndex);
            m_movingNeighborhoodIterator.SetLocation(fixedIndex);

            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_movingNeighborhoodIterator.GetPixel(i,inBounds);
                if (inBounds){
                    double f=this->nIt.GetPixel(i);
                    
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
            else result=-log(0.0000000000000000001);
            //result=result>0.5?0.5:result;
            return result;
        }
    };//FastUnaryPotentialRegistrationNCC
}//namespace
#endif /* POTENTIALS_H_ */
