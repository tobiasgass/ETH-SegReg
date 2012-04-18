#include "Log.h"

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
#include "TransformationUtils.h"

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
        typedef typename ImageType::PointType PointType;

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

        SizeType m_targetSize,m_atlasSize;
    protected:
        ConstImagePointerType m_targetImage, m_atlasImage;
        ConstImagePointerType m_scaledTargetImage, m_scaledAtlasImage;
        InterpolatorPointerType m_atlasInterpolator;
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
            m_targetImage=NULL;
            m_atlasImage=NULL;
            m_scale=1.0;
            m_scaleITK.Fill(1.0);
        }
        ~UnaryPotentialRegistrationNCC(){
            //delete nIt;
        }
        virtual void Init(){
            assert(m_targetImage);
            assert(m_atlasImage);
            if (m_scale!=1.0){
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_targetImage,1),m_scale);
                m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_atlasImage,1),m_scale);
            }else{
                m_scaledTargetImage=m_targetImage;
                m_scaledAtlasImage=m_atlasImage;
            }
            if (!radiusSet){
                LOG<<"Radius must be set before calling registrationUnaryPotential.Init()"<<endl;
                exit(0);
            }
                
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_scaledRadius[d]= m_radius[d]/m_scaledTargetImage->GetSpacing()[d]*m_targetImage->GetSpacing()[d];
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            LOGV(2)<<"Registration unary patch radius " << m_radius << " scale "<< m_scale << " scaledRadius "<< m_scaledRadius << endl;
            nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            m_atlasInterpolator=InterpolatorType::New();
            m_atlasInterpolator->SetInputImage(m_scaledAtlasImage);
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
                m_radius[d]=sp[d]/m_targetImage->GetSpacing()[d];
            
            }
            radiusSet=true;
        }
#if 0
        void SetRadius(RadiusType sp){
            m_radius=sp;
            radiusSet=true;
        }
#endif
        void SetBaseLabelMap(LabelImagePointerType blm, double scale=1.0){
            m_baseLabelMap=blm;m_haveLabelMap=true;
            if (scale!=1.0){
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
                    size[d]=int(inputSize[d]*scale);
                    spacing[d]=inputSpacing[d]*(1.0*(inputSize[d]-1)/(size[d]-1));
                    origin[d]=inputOrigin[d];//+0.5*spacing[d]/inputSpacing[d];
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
      
    	virtual void SetAtlasImage(ConstImagePointerType atlasImage){
            m_atlasImage=atlasImage;
            m_atlasSize=m_atlasImage->GetLargestPossibleRegion().GetSize();
       
        }
        void SetTargetImage(ConstImagePointerType targetImage){
            m_targetImage=targetImage;
            m_targetSize=m_targetImage->GetLargestPossibleRegion().GetSize();

        }
        ConstImagePointerType GetTargetImage(){
            return m_scaledTargetImage;
        }
        virtual double getPotential(IndexType targetIndex, LabelType disp){
            double result=0;
            IndexType idx1=targetIndex;
            PointType pos;
            //m_targetImage->TransformIndexToPhysicalPoint(idx1,pos);
            //m_scaledTargetImage->TransformPhysicalPointToIndex(pos,idx1);
            for (int d=0;d<ImageType::ImageDimension;++d){
                idx1[d]=m_scale*idx1[d];
            }
                
#ifdef PIXELTRANSFORM
            disp*=m_scale;
#endif
            nIt.SetLocation(idx1);
            double count=0, totalCount=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            for (unsigned int i=0;i<nIt.Size();++i){
                bool inBounds;
                double f=nIt.GetPixel(i,inBounds);
                if (inBounds){
                    IndexType neighborIndex=nIt.GetIndex(i);
#ifdef PIXELTRANSFORM
                    //this should be weighted somehow
                    ContinuousIndexType idx2(neighborIndex);
                    //double weight=1.0;
                    idx2+=disp+this->m_baseLabelMap->GetPixel(neighborIndex)*m_scale;
#else
                          
                    PointType p;
                    m_scaledTargetImage->TransformIndexToPhysicalPoint(neighborIndex,p);
                    p +=disp+this->m_baseLabelMap->GetPixel(neighborIndex);
                    ContinuousIndexType idx2;
                    m_scaledAtlasImage->TransformPhysicalPointToContinuousIndex(p,idx2);
                    
#endif
                    double m;
                    totalCount+=1.0;
                    if (!this->m_atlasInterpolator->IsInsideBuffer(idx2)){
#if 1
                        //continue;
                        m=0;
                        
#else
                        for (int d=0;d<ImageType::ImageDimension;++d){
                            double d1=idx2[d]-this->m_atlasInterpolator->GetEndContinuousIndex()[d];
                            if (d1>0){
                                idx2[d]-=2*d1;
                            }
                            else {
                                double d2=this->m_atlasInterpolator->GetStartContinuousIndex()[d]-idx2[d];
                                if (d2<0){                                    
                                    idx2[d]-=2*d2;
                                }
                            }
                        }
                        m=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
#endif
                    }else{
                        m=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
                    }
                    //LOG<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;
                }

            }
            if (!totalCount){
                LOG<<"this should never happen, neighborhood of pixel "<<idx1<<" was empty." <<endl;
                LOG<<m_scaledTargetImage->GetLargestPossibleRegion().GetSize()<<endl;
                LOG<<m_scaledRadius<<endl;
                LOG<<m_scaledRadius<<endl;
                LOG<<nIt.Size()<<endl;
                exit(0);
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
        virtual double GetOverlapRatio(IndexType targetIndex){
            double result=0;
            IndexType idx1=targetIndex;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=m_scale;
                if (targetIndex[d]>=(int)m_scaledAtlasImage->GetLargestPossibleRegion().GetSize()[d]) targetIndex[d]--;
            }
          
            nIt.SetLocation(targetIndex);
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
                    if (this->m_atlasInterpolator->IsInsideBuffer(idx2)){
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
        typedef typename ImageType::PointType PointType;

    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialSAD, Object);

        UnaryPotentialRegistrationSAD(){}
        
        virtual double getPotential(IndexType targetIndex, LabelType disp){
            double result=0;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
                if (targetIndex[d]>=(int)this->m_scaledAtlasImage->GetLargestPossibleRegion().GetSize()[d]) targetIndex[d]--;
            }
         
            
            //            LOG<<targetIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(targetIndex);
            this->nIt.SetLocation(targetIndex);
            double count=0;
            //double sum=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double f=this->nIt.GetPixel(i,inBounds);
                if (inBounds){
                    IndexType neighborIndex=this->nIt.GetIndex(i);
                    //this should be weighted somehow
                    //double weight=1.0;
                    PointType p;
                    this->m_scaledTargetImage->TransformIndexToPhysicalPoint(neighborIndex,p);
                    p +=disp+this->m_baseLabelMap->GetPixel(neighborIndex);
                    ContinuousIndexType idx2;
                    this->m_scaledAtlasImage->TransformPhysicalPointToContinuousIndex(p,idx2);
                    double m;
                    if (!this->m_atlasInterpolator->IsInsideBuffer(idx2)){
                        //continue;
                        m=0;
                    }else{
                        m=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
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
        
        virtual double getPotential(IndexType targetIndex, LabelType disp){
            double result=0;
          
            LabelType trueDisplacement=disp;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
            }
            LabelType baseDisp=this->m_baseLabelMap->GetPixel(targetIndex);
            //disp+=baseDisp;
            //LOG<<baseDisp<<" "<<disp<<std::endl;
            baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            LOG<<targetIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(targetIndex);
            this->nIt.SetLocation(targetIndex);
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

                    //LOG<<targetIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
                    double m;
                    if (!this->m_atlasInterpolator->IsInsideBuffer(idx2)){
                        continue;
                        m=0;
                        
#if 0
                        for (int d=0;d<ImageType::ImageDimension;++d){
                            if (idx2[d]>=this->m_atlasInterpolator->GetEndContinuousIndex()[d]){
                                idx2[d]=this->m_atlasInterpolator->GetEndContinuousIndex()[d]-0.5;
                            }
                            else if (idx2[d]<this->m_atlasInterpolator->GetStartContinuousIndex()[d]){
                                idx2[d]=this->m_atlasInterpolator->GetStartContinuousIndex()[d]+0.5;
                            }
                        }
#endif
                    }else{
                        m=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
                    }
                    //LOG<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
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
                            weight*=1.0-fabs((1.0*targetIndex[d]-neighborIndex[d])/(this->m_radius[d]));
                            trueIndex[d]/=this->m_scale;
                        }
                        int segmentationPriorLabel=(this->m_segmentationPrior->GetPixel(neighborIndex));
                        //double penalty=weight*this->m_srsPotential->getPotential(neighborIndex,neighborIndex,disp,segmentationPriorLabel);
                        double penalty=weight*this->m_srsPotential->getPotential(trueIndex,trueIndex,trueDisplacement+baseDisplacement,segmentationPriorLabel);
                        segmentationPenalty+=penalty;
                        //LOG<<targetIndex<<" "<<neighborIndex<<" "<<weight<<" "<<segmentationPriorLabel<<" "<<penalty<<endl;
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
            assert(this->m_targetImage);
            assert(this->m_atlasImage);
            assert(this->m_targetSheetness);
            assert(this->m_atlasSegmentation);
            if (this->m_scale!=1.0){
                this->m_scaledTargetImage=FilterUtils<ImageType>::LinearResample((this->m_targetImage),this->m_scale);
                this->m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample((this->m_atlasImage),this->m_scale);
                this->m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((m_atlasSegmentation),this->m_scale);
                this->m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((m_targetSheetness),this->m_scale);
            }
            assert(this->radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=this->m_radius[d]*this->m_scale;
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->m_atlasInterpolator=InterpolatorType::New();
            this->m_atlasInterpolator->SetInputImage(this->m_scaledAtlasImage);
            this->m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(this->m_scaledAtlasSegmentation);
        }
        virtual double getPotential(IndexType targetIndex, LabelType disp){
            double result=0;
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
            }
            //LabelType baseDisp=this->m_baseLabelMap->GetPixel(targetIndex);
            //LOG<<baseDisp<<" "<<disp<<std::endl;
            //baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            LOG<<targetIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(targetIndex);
            this->nIt.SetLocation(targetIndex);
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

                    //LOG<<targetIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
                    double m;
                    if (!this->m_atlasInterpolator->IsInsideBuffer(idx2)){
                        assert(!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2));
                        continue;
                        m=0;
                        
#if 0
                        for (int d=0;d<ImageType::ImageDimension;++d){
                            if (idx2[d]>=this->m_atlasInterpolator->GetEndContinuousIndex()[d]){
                                idx2[d]=this->m_atlasInterpolator->GetEndContinuousIndex()[d]-0.5;
                            }
                            else if (idx2[d]<this->m_atlasInterpolator->GetStartContinuousIndex()[d]){
                                idx2[d]=this->m_atlasInterpolator->GetStartContinuousIndex()[d]+0.5;
                            }
                        }
#endif
                    }else{
                        m=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
                    }
                    //LOG<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;
                    double weight=1.0;
                    for (unsigned int d=0;d<D;++d){
                        weight*=1.0-fabs((1.0*targetIndex[d]-neighborIndex[d])/(this->m_scaledRadius[d]));
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
                        bool atlasTissue=m<tissue;
                        bool atlasBone=m>bone;
                        bool targetTissue=f<tissue;
                        bool targetBone=f>bone;
                        
                        distanceSum+=weight;
                        segmentationPenalty+=weight*( (atlasTissue==targetBone) || (atlasBone==targetTissue));
                        
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
                // LOG<<targetIndex<<" "<<segmentationPenalty<<" "<<distanceSum<<endl;
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
        FloatImageInterpolatorPointerType m_atlasDistanceTransformInterpolator;
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
            m_atlasDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            m_atlasDistanceTransformInterpolator->SetInputImage(dt1);
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
            assert(this->m_targetImage);
            assert(this->m_atlasImage);
            assert(this->m_targetSheetness);
            assert(this->m_atlasSegmentation);
            if (this->m_scale!=1.0){
                this->m_scaledTargetImage=FilterUtils<ImageType>::LinearResample((this->m_targetImage),this->m_scale);
                this->m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample((this->m_atlasImage),this->m_scale);
                this->m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((m_atlasSegmentation),this->m_scale);
                this->m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((m_targetSheetness),this->m_scale);
            }
            assert(this->radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=this->m_radius[d]*this->m_scale;
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->m_atlasInterpolator=InterpolatorType::New();
            this->m_atlasInterpolator->SetInputImage(this->m_scaledAtlasImage);
            this->m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(this->m_scaledAtlasSegmentation);
        }
        virtual double getPotential(IndexType targetIndex, LabelType disp){
            double result=0;
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
            }
            //LabelType baseDisp=this->m_baseLabelMap->GetPixel(targetIndex);
            //LOG<<baseDisp<<" "<<disp<<std::endl;
            //baseDisp*=this->m_scale;
            disp*=this->m_scale;
            //            LOG<<targetIndex<<"\t "<<disp<<"\t "<<std::endl;
            //          nIt->SetLocation(targetIndex);
            this->nIt.SetLocation(targetIndex);
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

                    //LOG<<targetIndex<<" "<<disp<<" "<<idx2<<" "<<endl;
                    double m;
                    totalCount++;
                    if (!this->m_atlasInterpolator->IsInsideBuffer(idx2)){
                        assert(!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2));
                        continue;
                        //m=-50;
                        
#if 0
                        for (int d=0;d<ImageType::ImageDimension;++d){
                            if (idx2[d]>=this->m_atlasInterpolator->GetEndContinuousIndex()[d]){
                                idx2[d]=this->m_atlasInterpolator->GetEndContinuousIndex()[d]-0.5;
                            }
                            else if (idx2[d]<this->m_atlasInterpolator->GetStartContinuousIndex()[d]){
                                idx2[d]=this->m_atlasInterpolator->GetStartContinuousIndex()[d]+0.5;
                            }
                        }
#endif
                    }else{
                        m=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
                    }
                    //LOG<<f<<" "<<m<<" "<<sff<<" "<<sfm<<" "<<sf<<" "<<sm<<endl;
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;

                    if (false &&this->m_alpha){
                        double weight=1.0;
                        for (unsigned int d=0;d<D;++d){
                            weight*=1.0-fabs((1.0*targetIndex[d]-neighborIndex[d])/(this->m_scaledRadius[d]));
                        }

                        if (f>=bone){
                            int seg=this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0.5;
                            if ( !seg){
                                double distance=m_atlasDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2)/sigma1;
                                segmentationPenalty+=weight*distance;
                            }
                        }else if ( f<tissue){
                            int seg=this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0.5;
                            if (seg){
                                //double distance=m_atlasDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2)/sigma1;
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
                    //LOG<<"AUTOCORRELATION ZERO "<<count<<endl;
                }
                // LOG<<targetIndex<<" "<<segmentationPenalty<<" "<<distanceSum<<endl;
                if (distanceSum){
                    result=result+this->m_alpha*segmentationPenalty/distanceSum;
                }
                //LOG<<"result "<<result<<" penalty factor:"<<1+this->m_alpha*(1.0*totalCount-count)/(totalCount)<<" countDiff:"<<totalCount-count<<endl;
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
        ImagePointerType m_shiftedAtlasImage;
        ImageNeighborhoodIteratorType m_atlasNeighborhoodIterator;
        LabelType m_currentDisp;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialNCC, Object);

        virtual void shiftAtlasImage(LabelType displacement){
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
            //translate atlas image
            resampleFilter->SetOutputParametersFromImage(this->m_scaledAtlasImage);
            resampleFilter->SetInput(this->m_scaledAtlasImage);
            resampleFilter->Update();
#if 0
            if (this->m_scale!=1.0){
                m_shiftedAtlasImage=FilterUtils<ImageType>::LinearResample((ConstImagePointerType)(resampleFilter->GetOutput()),this->m_scale);
            }else{
                m_shiftedAtlasImage=resampleFilter->GetOutput();
            }
#endif
            
            m_shiftedAtlasImage=resampleFilter->GetOutput();
            m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_radius,this->m_shiftedAtlasImage, this-> m_shiftedAtlasImage->GetLargestPossibleRegion());
            m_currentDisp=scaledLabel;
          
        }
        virtual double getLocalPotential(IndexType targetIndex){

            double result;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
                if (targetIndex[d]>=(int)m_shiftedAtlasImage->GetLargestPossibleRegion().GetSize()[d]) targetIndex[d]--;

            }
            //  if(!( m_shiftedAtlasImage->GetLargestPossibleRegion().IsInside(targetIndex) && this->m_targetImage->GetLargestPossibleRegion().IsInside(targetIndex))){
            //LOG<<targetIndex<<" "<<m_shiftedAtlasImage->GetLargestPossibleRegion().GetSize()<<" "<<this->m_scaledAtlasImage->GetLargestPossibleRegion().GetSize()<<endl;
            //  }
            this->nIt.SetLocation(targetIndex);
            m_atlasNeighborhoodIterator.SetLocation(targetIndex);

            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
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
