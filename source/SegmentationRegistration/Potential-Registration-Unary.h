
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
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkConstNeighborhoodIterator.h"
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>
#include "Potential-SegmentationRegistration-Pairwise.h"
#include "itkTranslationTransform.h"
#include "TransformationUtils.h"
#include "Log.h"
#include <limits>
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkHistogram.h"
#include "itkScalarImageToHistogramGenerator.h"
#include <limits>
#include "itkIdentityTransform.h"
using namespace std;
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
        SpacingType m_coarseImageSpacing;
        ImageNeighborhoodIteratorType nIt;
        double m_scale;
        SizeType m_scaleITK,m_invertedScaleITK;
        double  m_threshold;
        bool LOGPOTENTIAL;
        bool m_noOutSidePolicy;
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
            m_threshold=std::numeric_limits<double>::max();
            LOGPOTENTIAL=false;
            m_noOutSidePolicy = false;
        }
        ~UnaryPotentialRegistrationNCC(){
            //delete nIt;
        }
        virtual void Compute(){}
        virtual void setDisplacements(std::vector<LabelType> displacements){}
        virtual void setCoarseImage(ImagePointerType img){}
        virtual void setThreshold(double t){m_threshold=t;}
        virtual void setLogPotential(bool b){LOGPOTENTIAL=b;}
        virtual void setNoOutsidePolicy(bool b){ m_noOutSidePolicy = b;}
        virtual void Init(){
            assert(m_targetImage);
            assert(m_atlasImage);
            if ( m_scale!=1.0){
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_targetImage,m_scale,false),m_scale,false);
                m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_atlasImage,m_scale,false),m_scale,false);
                //m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(m_targetImage,m_scale);
                //m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample(m_atlasImage,m_scale);
                //m_scaledTargetImage=FilterUtils<ImageType>::gaussian(FilterUtils<ImageType>::LinearResample(m_targetImage,m_scale),1);
                //m_scaledAtlasImage=FilterUtils<ImageType>::gaussian(FilterUtils<ImageType>::LinearResample(m_atlasImage,m_scale),1);
            }else{
                m_scaledTargetImage=m_targetImage;
                m_scaledAtlasImage=m_atlasImage;
            }
            if (!radiusSet){
                LOG<<"Radius must be set before calling registrationUnaryPotential.Init()"<<endl;
                exit(0);
            }
                
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_scaledRadius[d]=m_scale*m_radius[d];
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
            m_coarseImageSpacing=sp;
            double radiusScaling=1;
            LOGV(2)<<VAR(radiusScaling)<<endl;
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_radius[d]=radiusScaling*sp[d]/m_targetImage->GetSpacing()[d];
            }
            radiusSet=true;
        }

        void SetBaseLabelMap(LabelImagePointerType blm, double scale=1.0){
            m_baseLabelMap=blm;m_haveLabelMap=true;
            if (scale!=1.0){
                m_baseLabelMap=TransfUtils<ImageType>::linearInterpolateDeformationField(blm,m_scaledTargetImage);
            }
        }
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
        virtual void SetAtlasImage(ImagePointerType atlasImage){
            SetAtlasImage(ConstImagePointerType(atlasImage));
        }

    	virtual void SetAtlasImage(ConstImagePointerType atlasImage){
            m_atlasImage=atlasImage;
            m_atlasSize=m_atlasImage->GetLargestPossibleRegion().GetSize();
#if 0
            m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian(m_atlasImage,1),m_scale);
            m_atlasInterpolator=InterpolatorType::New();
            m_atlasInterpolator->SetInputImage(m_scaledAtlasImage);
#endif
        }
        void SetTargetImage(ConstImagePointerType targetImage){
            m_targetImage=targetImage;
            m_targetSize=m_targetImage->GetLargestPossibleRegion().GetSize();

        }
        ConstImagePointerType GetTargetImage(){
            return m_scaledTargetImage;
        }
        ConstImagePointerType GetAtlasImage(){
            return m_scaledAtlasImage;
        }
        virtual double getPotential(IndexType targetIndex, LabelType disp){
            double result=0;
            IndexType idx1=targetIndex;
            PointType pos;
            m_targetImage->TransformIndexToPhysicalPoint(idx1,pos);
            m_scaledTargetImage->TransformPhysicalPointToIndex(pos,idx1);
            //for (int d=0;d<ImageType::ImageDimension;++d){
            //idx1[d]=m_scale*idx1[d];
            //}
                
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
#if 0
                        continue;
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
            if (count<1)
                result=-log(0.5);//100000000;//-log(0.0000000000000000001);{
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
                    result=-log(0.5);
                    //if (sfm>0) result=0;
                    //else result=1;
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
        double sigma1, sigma2, mean1, mean2;
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
        typedef UnaryPotentialRegistrationNCC<TLabelMapper,TImage> Superclass;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;

        typedef typename LabelMapperType::LabelImageType LabelImageType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType,itk::ConstantBoundaryCondition<TImage,TImage> > ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
        typedef typename itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
    protected:
        ImageNeighborhoodIteratorType m_atlasNeighborhoodIterator,m_maskNeighborhoodIterator;
        std::vector<LabelType> m_displacements;
        std::vector<FloatImagePointerType> m_potentials;
        LabelType currentActiveDisplacement;
        FloatImagePointerType currentCachedPotentials;
        ImagePointerType m_coarseImage;
        double m_averageFixedPotential,m_oldAveragePotential;
        double m_normalizationFactor;
        bool m_normalize;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialNCC, Object);
        
        FastUnaryPotentialRegistrationNCC():Superclass(){
            m_normalizationFactor=1.0;
            m_normalize=false;
            
        }
        virtual void compute(){
            //LOG<<"DEPRECATED, too memory intensive!!"<<endl;
            m_potentials=std::vector<FloatImagePointerType>(m_displacements.size(),NULL);
            m_averageFixedPotential=0;
            for (unsigned int n=0;n<m_displacements.size();++n){
                LOGV(9)<<"cachhing unary registrationpotentials for label " <<n<<endl;
                FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(m_coarseImage);
                LabelImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseLabelMap);
                translation->FillBuffer( m_displacements[n]);
                LabelImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseLabelMap);
                ImagePointerType deformedAtlas,deformedMask;
                pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation);
                deformedAtlas=result.first;
                deformedMask=result.second;
                m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
                m_maskNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedMask,deformedMask->GetLargestPossibleRegion());
                FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
                for (coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator)
                    {
                        IndexType coarseIndex=coarseIterator.GetIndex();
                        PointType point;
                        m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex,point);
                        IndexType targetIndex;
                        this->m_scaledTargetImage->TransformPhysicalPointToIndex(point,targetIndex);
                        double localPot=getLocalPotential(targetIndex);
                        coarseIterator.Set(localPot);
                        if (n==m_displacements.size()/2){
                            m_averageFixedPotential+=localPot;
                        }

                    }
                m_potentials[n]=pot;
            }
           
        }
        void setNormalize(bool b){m_normalize=b;}
        void resetNormalize(){
            m_normalize=false;
            m_normalizationFactor=1.0;
        }
#if 1
        void cachePotentials(LabelType displacement){
            LabelType zeroDisp;
            zeroDisp.Fill(0.0);
            //compute average potential for zero displacement.
            bool computeAverage=(displacement == zeroDisp);
            m_averageFixedPotential=computeAverage?0.0:m_averageFixedPotential;

            FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(m_coarseImage);
            LabelImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseLabelMap);
            translation->FillBuffer( displacement);
            LabelImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseLabelMap);
            ImagePointerType deformedAtlas,deformedMask;
            pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation);
            deformedAtlas=result.first;
            deformedMask=result.second;
            m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
            m_maskNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedMask,deformedMask->GetLargestPossibleRegion());

            LOGV(70)<<VAR(m_atlasNeighborhoodIterator.GetRadius())<<" "<<VAR(deformedAtlas->GetLargestPossibleRegion().GetSize())<<endl;
            LOGV(70)<<VAR(this->nIt.GetRadius())<<" "<<VAR(deformedAtlas->GetLargestPossibleRegion().GetSize())<<endl;

            FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
            for (coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator){
                IndexType coarseIndex=coarseIterator.GetIndex();
                PointType point;
                m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex,point);
                IndexType targetIndex;
                this->m_scaledTargetImage->TransformPhysicalPointToIndex(point,targetIndex);
                double localPot=getLocalPotential(targetIndex);
                coarseIterator.Set(localPot);
                if (computeAverage)
                    m_averageFixedPotential+=localPot;
               
            }
            currentCachedPotentials=pot;
            currentActiveDisplacement=displacement;

            if (computeAverage){
                m_averageFixedPotential/= m_coarseImage->GetBufferedRegion().GetNumberOfPixels();
                if (m_normalize){
                    m_normalizationFactor= m_normalizationFactor*m_oldAveragePotential/m_averageFixedPotential;
                }
                LOGV(3)<<VAR(m_normalizationFactor)<<endl;
                m_oldAveragePotential=m_averageFixedPotential;
            }
        }
#else
        //not actually faster :(
        void cachePotentials(LabelType displacement){
            LabelType zeroDisp;
            zeroDisp.Fill(0.0);
            //compute average potential for zero displacement.
            bool computeAverage=(displacement == zeroDisp);
            m_averageFixedPotential=computeAverage?0.0:m_averageFixedPotential;

            LabelImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseLabelMap);
            translation->FillBuffer( displacement);
            LabelImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseLabelMap);
            ImagePointerType deformedAtlas,deformedMask;
            pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation);
            deformedAtlas=result.first;
            deformedMask=result.second;
            
            FloatImagePointerType pot=localPotentials(this->m_scaledAtlasImage,this->m_scaledTargetImage);
            pot = FilterUtils<FloatImageType>::LinearResample(pot, FilterUtils<ImageType,FloatImageType>::cast(m_coarseImage));

            deformedMask = FilterUtils<ImageType>::NNResample(deformedMask, m_coarseImage);
            ImageIteratorType maskIterator=ImageIteratorType(deformedMask,deformedMask->GetLargestPossibleRegion());
            FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
           
            for (maskIterator.GoToBegin(),coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator, ++ maskIterator){
                double localPot = coarseIterator.Get();
                if (this->m_noOutSidePolicy  && !maskIterator.Get()){
                    coarseIterator.Set(1e10);
                }
                if (computeAverage)
                    m_averageFixedPotential+=localPot;
               
            }
            currentCachedPotentials=pot;
            currentActiveDisplacement=displacement;

            if (computeAverage){
                m_averageFixedPotential/= m_coarseImage->GetBufferedRegion().GetNumberOfPixels();
                if (m_normalize){
                    m_normalizationFactor= m_normalizationFactor*m_oldAveragePotential/m_averageFixedPotential;
                }
                LOGV(3)<<VAR(m_normalizationFactor)<<endl;
                m_oldAveragePotential=m_averageFixedPotential;
            }
        }
#endif
        void setDisplacements(std::vector<LabelType> displacements){
            m_displacements=displacements;
        }
        void setCoarseImage(ImagePointerType img){m_coarseImage=img;}

        virtual double getPotential(IndexType coarseIndex, unsigned int displacementLabel){
            //LOG<<"DEPRECATED BEHAVIOUR!"<<endl;
            return m_potentials[displacementLabel]->GetPixel(coarseIndex);
        }
        virtual double getPotential(IndexType coarseIndex){
            //LOG<<"NEW BEHAVIOUR!"<<endl;
            return  m_normalizationFactor*currentCachedPotentials->GetPixel(coarseIndex);
        }
        virtual double getPotential(IndexType coarseIndex, LabelType l){
            LOG<<"ERROR NEVER CALL THIS"<<endl;
            exit(0);
        }

        virtual FloatImagePointerType localPotentials(ConstImagePointerType i1, ConstImagePointerType i2){
            return FilterUtils<ImageType,FloatImageType>::LNCC(i1,i2,i1->GetSpacing()[0]);
        }

        virtual double getLocalPotential(IndexType targetIndex){

            double result;
            this->nIt.SetLocation(targetIndex);
            m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            m_maskNeighborhoodIterator.SetLocation(targetIndex);
            double insideCount=0.0;
            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                   
                insideCount+=inBounds;
                bool inside=m_maskNeighborhoodIterator.GetPixel(i);
                if (!inside)
                    m=0.0;
                if ( inBounds && (inside|| this->m_noOutSidePolicy)  ){
                    double f=this->nIt.GetPixel(i);
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;

                }
            }
            double NCC=0;
            if (count){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    NCC=1.0*sfm/sqrt(smm*sff);
                
                }
            }
            //result=result>0.5?0.5:result; 
            if (this->LOGPOTENTIAL){
                result=(1.0+((NCC)))/2;
                result=result>0?result:0.00000001;
                result=-log(result);
            }else{
                result=(1-(NCC))/2;
            }
            result=min(this->m_threshold,result);
#if 0            
            if (this->m_noOutSidePolicy &&( count != insideCount )){
                return 1e10*count/insideCount;
            } 
#endif     
            return result*insideCount/this->nIt.Size();
        }
    };//FastUnaryPotentialRegistrationNCC
    template<class TLabelMapper,class TImage>
    class FastUnaryPotentialRegistrationSAD: public FastUnaryPotentialRegistrationNCC<TLabelMapper,TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationSAD            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

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
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialSAD, Object);
        
        virtual FloatImagePointerType localPotentials(ImagePointerType i1, ImagePointerType i2){
            return FilterUtils<ImageType,FloatImageType>::LSAD(i1,i2,i1->GetSpacing()[0]);

        }

      
    
        virtual double getLocalPotential(IndexType targetIndex){

            double result;
            this->nIt.SetLocation(targetIndex);
            this->m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            this->m_maskNeighborhoodIterator.SetLocation(targetIndex);
            double insideCount=0.0;
            double count=0;
            double sum=0.0;
            PointType centerPoint,neighborPoint;
            this->m_scaledTargetImage->TransformIndexToPhysicalPoint(targetIndex,centerPoint);
            double maxNorm=this->m_coarseImageSpacing.GetNorm();
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=this->m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                insideCount+=inBounds;
                bool inside=this->m_maskNeighborhoodIterator.GetPixel(i);
               
                if (inside && (inBounds || this->m_noOutSidePolicy)){
                    double f=this->nIt.GetPixel(i);
                    this->m_scaledTargetImage->TransformIndexToPhysicalPoint(this->nIt.GetIndex(i),neighborPoint);
                    double weight=1.0-(centerPoint-neighborPoint).GetNorm()/maxNorm;
                    sum+=weight*fabs(f-m);
                    count+=weight;
                }else if (this->m_noOutSidePolicy &&( inside && ! inBounds )){
                    return 1e10;
                } 
            }
            if (count>0){
                sum/=count;
            }//else          sum=this->nIt.Size();
            //result=result>0.5?0.5:result; 
            if (this->LOGPOTENTIAL){
            }else{
                result=sum;
            }
            result=min(this->m_threshold,result);
          
            return result*insideCount/this->nIt.Size();
        }
    };//FastUnaryPotentialRegistrationSAD
    template<class TLabelMapper,class TImage>
    class FastUnaryPotentialRegistrationSSD: public FastUnaryPotentialRegistrationNCC<TLabelMapper,TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationSSD            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

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
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialSSD, Object);
        virtual FloatImagePointerType localPotentials(ImagePointerType i1, ImagePointerType i2){
            return FilterUtils<ImageType,FloatImageType>::LSSD(i1,i2,i1->GetSpacing()[0]);
        }
     
    
        virtual double getLocalPotential(IndexType targetIndex){

            double result;
            this->nIt.SetLocation(targetIndex);
            this->m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            this->m_maskNeighborhoodIterator.SetLocation(targetIndex);
            double insideCount=0.0;
            double count=0;
            double sum=0.0;
            PointType centerPoint,neighborPoint;
            this->m_scaledTargetImage->TransformIndexToPhysicalPoint(targetIndex,centerPoint);
            double maxNorm=this->m_coarseImageSpacing.GetNorm();
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=this->m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                insideCount+=inBounds;
                bool inside=this->m_maskNeighborhoodIterator.GetPixel(i);
                if (inside && inBounds){
                    double f=this->nIt.GetPixel(i);
                    this->m_scaledTargetImage->TransformIndexToPhysicalPoint(this->nIt.GetIndex(i),neighborPoint);
                    double weight=1.0-(centerPoint-neighborPoint).GetNorm()/maxNorm;
                    sum+=weight*fabs(f-m)*fabs(f-m);
                    count+=weight;
                }else  if (this->m_noOutSidePolicy &&( inside && ! inBounds )){
                    return 1e10;
                } 
            }
            if (count>0){
                sum/=count;
            }//else          sum=this->nIt.Size();
            //result=result>0.5?0.5:result; 
            if (this->LOGPOTENTIAL){
            }else{
                result=sum;
            }
            result=min(this->m_threshold,result);
         
            return result*insideCount/this->nIt.Size();
        }
    };//FastUnaryPotentialRegistrationSSD


#define NMI
    template<class TLabelMapper,class TImage>
    class FastUnaryPotentialRegistrationNMI: public UnaryPotentialRegistrationNCC<TLabelMapper,TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationNMI            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef UnaryPotentialRegistrationNCC<TLabelMapper,TImage> Superclass;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::PixelType PixelType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;

        
        typedef typename LabelMapperType::LabelImageType LabelImageType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
#ifdef NMI        
        typedef typename itk::NormalizedMutualInformationHistogramImageToImageMetric<ImageType,ImageType> NMIMetricType;
        typedef typename NMIMetricType::MeasureType             MeasureType;
        typedef typename NMIMetricType::HistogramType            HistogramType;
        typedef typename HistogramType::SizeType              HistogramSizeType;
        typedef typename HistogramType::MeasurementVectorType MeasurementVectorType;
        typedef typename HistogramType::AbsoluteFrequencyType HistogramFrequencyType;
        typedef typename HistogramType::Iterator              HistogramIteratorType;
        
        typedef Statistics::ScalarImageToHistogramGenerator<ImageType> HistGenType ;
#else
#ifdef MMI
        typedef typename itk::MattesMutualInformationImageToImageMetric<ImageType,ImageType> NMIMetricType;
#else
#ifdef MI
#endif
#endif
#endif
    protected:
        ImageNeighborhoodIteratorType m_atlasNeighborhoodIterator,m_maskNeighborhoodIterator;
        std::vector<LabelType> m_displacements;
        std::vector<FloatImagePointerType> m_potentials;
        ImagePointerType m_coarseImage;
        typename NMIMetricType::Pointer m_metric;
        int m_numberOfBins;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialNMI, Object);
       

        virtual void compute(){
            m_numberOfBins=32;
            m_metric=NMIMetricType::New();
            typename NNInterpolatorType::Pointer nnInt=NNInterpolatorType::New();
            m_metric->SetInterpolator(nnInt);
            m_potentials=std::vector<FloatImagePointerType>(m_displacements.size(),NULL);
#ifdef NMI
            double fixedEntropy=computeEntropy(this->m_scaledTargetImage);
#else
            //m_metric->SetNumberOfSpatialSamples();
            m_metric->SetNumberOfHistogramBins(64);
#endif          
            for (unsigned int n=0;n<m_displacements.size();++n){
                LOGV(9)<<"cachhing unary registrationpotentials for label " <<n<<endl;
                FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(m_coarseImage);
                LabelImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseLabelMap);
                translation->FillBuffer( m_displacements[n]);
                LabelImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseLabelMap);
                ImagePointerType deformedAtlas,deformedMask;
                pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation);
                deformedAtlas=result.first;
                deformedMask=result.second;
                m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
                m_maskNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedMask,deformedMask->GetLargestPossibleRegion());
                m_metric->SetFixedImage(this->m_scaledTargetImage);
                m_metric->SetMovingImage(deformedAtlas);
                nnInt->SetInputImage(deformedAtlas);
#ifdef NMI
                double movingEntropy=computeEntropy((ConstImagePointerType)deformedAtlas);
                typename HistogramType::SizeType histSize(2);
                histSize.Fill(100); 
                m_metric->SetHistogramSize(histSize);
#endif
                FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
                for (coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator){
                    IndexType coarseIndex=coarseIterator.GetIndex();
                    PointType point;
                    m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex,point);
                    IndexType targetIndex;
                    this->m_scaledTargetImage->TransformPhysicalPointToIndex(point,targetIndex);
                    //coarseIterator.Set(getLocalPotential(targetIndex));
                    //double potentialNMI=getLocalPotential(targetIndex,fixedEntropy,movingEntropy);
                    double potentialNMI=getLocalPotential(targetIndex);
                    //double potentialNCC=getLocalPotentialNCC(targetIndex);
                    //LOGGV(5)<<VAR(potentialNMI)<<" "<<VAR(potentialNCC)<<endl;
                    coarseIterator.Set(potentialNMI);

                }
                m_potentials[n]=pot;
                            
            }
        }
#ifdef NMI        
        double computeEntropy(ConstImagePointerType img){
            typename HistGenType::Pointer histGen=HistGenType::New();
            histGen->SetInput(img);
            histGen->SetNumberOfBins(50);
            histGen->Compute();
            typename HistogramType::ConstPointer  hist=histGen->GetOutput();
            LOGV(40)<<hist<<endl;
            MeasureType entropyX = NumericTraits< MeasureType >::Zero;
            typedef typename NumericTraits< HistogramFrequencyType >::RealType HistogramFrequencyRealType;
            HistogramFrequencyRealType totalFreq =
                static_cast< HistogramFrequencyRealType >( hist->GetTotalFrequency() );

            for ( unsigned int i = 0; i < hist->GetSize(0); i++ )
                {
                    HistogramFrequencyRealType freq =
                        static_cast< HistogramFrequencyRealType >( hist->GetFrequency(i, 0) );
                    LOGV(70)<<VAR(i)<<" "<<VAR(freq)<<endl;
                    if ( freq > 0 )
                        {
                            entropyX += freq * vcl_log(freq);
                        }
                }

            entropyX = -entropyX / static_cast< MeasureType >( totalFreq ) + vcl_log(totalFreq);
            LOGV(40)<<VAR(entropyX)<<endl;
            return entropyX;
        }
#endif
        void setDisplacements(std::vector<LabelType> displacements){
            m_displacements=displacements;
        }
        void setCoarseImage(ImagePointerType img){m_coarseImage=img;}

        virtual double getPotential(IndexType coarseIndex, unsigned int displacementLabel){
            return m_potentials[displacementLabel]->GetPixel(coarseIndex);
        }
        virtual double getPotential(IndexType coarseIndex, LabelType l){
            LOG<<"ERROR NEVER CALL THIS"<<endl;
            exit(0);
        }
        virtual double getLocalPotential(IndexType targetIndex){
#if 1
            //use ITK (SLOW!!!)
            typedef itk::IdentityTransform<double,ImageType::ImageDimension> TransType;
            typename TransType::Pointer t=TransType::New();
            IndexType cornerIndex=targetIndex-this->m_scaledRadius;
            IndexType secondCornerIndex=targetIndex+this->m_scaledRadius;
            SizeType regionSize;
            for (unsigned int d=0;d< ImageType::ImageDimension;++d){
                if (cornerIndex[d]<0)
                    cornerIndex[d]=0;
                if ( secondCornerIndex[d] >= this->m_scaledTargetImage->GetLargestPossibleRegion().GetSize()[d])
                    secondCornerIndex[d]=this->m_scaledTargetImage->GetLargestPossibleRegion().GetSize()[d]-1;
                regionSize[d]=secondCornerIndex[d]-cornerIndex[d];
            }
            typename ImageType::RegionType region(cornerIndex,regionSize);
            m_metric->SetTransform(t);
            m_metric->SetFixedImageRegion(region);
            m_metric->Initialize();
            double result=m_metric->GetValue(t->GetParameters());
            return result;
                
#else            
            //use iterators
            double result;
            this->nIt.SetLocation(targetIndex);
            m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            m_maskNeighborhoodIterator.SetLocation(targetIndex);
            
            double count=0;
            //compute joint histogram
            typename HistogramType::Pointer jointHist=HistogramType::New();
            jointHist->SetMeasurementVectorSize(2);
            typename HistogramType::SizeType histSize(2);
            histSize.Fill(m_numberOfBins);

            //find bounds
            MeasurementVectorType m_LowerBound;
            MeasurementVectorType m_UpperBound;
            m_LowerBound.SetSize(2);
            m_UpperBound.SetSize(2);
            m_LowerBound[0]=numeric_limits<PixelType>::max();
            m_LowerBound[1]=numeric_limits<PixelType>::max();
            m_UpperBound[0]=numeric_limits<PixelType>::min();
            m_UpperBound[1]=numeric_limits<PixelType>::min();
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                bool inside=m_maskNeighborhoodIterator.GetPixel(i);
                if (inside && inBounds){
                    double f=this->nIt.GetPixel(i);
                    m_LowerBound[0]=f<m_LowerBound[0]?f:m_LowerBound[0];
                    m_LowerBound[1]=m<m_LowerBound[1]?m:m_LowerBound[1];
                    m_UpperBound[0]=f>m_UpperBound[0]?f:m_UpperBound[0];
                    m_UpperBound[1]=m>m_UpperBound[1]?f:m_UpperBound[1];

                }
            }                                                           
            LOGV(40)<<VAR(histSize)<<" "<<VAR(m_LowerBound)<<" "<<VAR(m_UpperBound)<<endl;
            jointHist->Initialize(histSize,m_LowerBound,m_UpperBound);                    
            typename HistogramType::MeasurementVectorType sample;
            sample.SetSize(2);

            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                bool inside=m_maskNeighborhoodIterator.GetPixel(i);
                if (inside && inBounds){
                    double f=this->nIt.GetPixel(i);
                    sample[0] = f;
                    sample[1] = m;
                    LOGV(70)<<VAR(sample)<<endl;
                    jointHist->IncreaseFrequencyOfMeasurement(sample, 1);
                    count++;
                }
            }
          
            if (count){
                double NMI=-2;
                MeasureType entropyX = NumericTraits< MeasureType >::Zero;
                MeasureType entropyY = NumericTraits< MeasureType >::Zero;
                MeasureType jointEntropy = NumericTraits< MeasureType >::Zero;

                typedef typename NumericTraits< HistogramFrequencyType >::RealType HistogramFrequencyRealType;

                HistogramFrequencyRealType totalFreq =
                    static_cast< HistogramFrequencyRealType >( jointHist->GetTotalFrequency() );

                for ( unsigned int i = 0; i < jointHist->GetSize(0); i++ )
                    {
                        HistogramFrequencyRealType freq =
                            static_cast< HistogramFrequencyRealType >( jointHist->GetFrequency(i, 0) );
                        LOGV(50)<<VAR(i)<<" "<<VAR(freq)<<endl;
                        if ( freq > 0 )
                            {
                                entropyX += freq * vcl_log(freq);
                            }
                    }

                entropyX = -entropyX / static_cast< MeasureType >( totalFreq ) + vcl_log(totalFreq);

                for ( unsigned int i = 0; i < jointHist->GetSize(1); i++ )
                    {
                        HistogramFrequencyRealType freq =
                            static_cast< HistogramFrequencyRealType >( jointHist->GetFrequency(i, 1) );
                        LOGV(50)<<VAR(i)<<" "<<VAR(freq)<<endl;
                        if ( freq > 0 )
                            {
                                entropyY += freq * vcl_log(freq);
                            }
                    }

                entropyY = -entropyY / static_cast< MeasureType >( totalFreq ) + vcl_log(totalFreq);

                HistogramIteratorType it = jointHist->Begin();
                HistogramIteratorType end = jointHist->End();
                while ( it != end )
                    {
                        HistogramFrequencyRealType freq =
                            static_cast< HistogramFrequencyRealType >( it.GetFrequency() );
                        LOGV(50)<<VAR(freq)<<endl;
                        if ( freq > 0 )
                            {
                                jointEntropy += freq * vcl_log(freq);
                            }
                        ++it;
                    }

                jointEntropy = -jointEntropy / static_cast< MeasureType >( totalFreq )
                    + vcl_log(totalFreq);
                LOGV(40)<<VAR(jointEntropy)<<" "<<VAR(entropyX)<<" "<<entropyY<<endl;
#if 1
                double ECC;
                if (( entropyX + entropyY)==0.0)
                    ECC=0;
                else 
                    ECC=2.0 - 2.0*jointEntropy/( entropyX + entropyY );
                result=-ECC;
#else
                if (jointEntropy==0)
                    NMI=0.00000001;
                else
                    NMI= ( entropyX + entropyY ) / jointEntropy;
                result=-(2.0-2.0/NMI);
#endif
            }else return 0;



            return result;
#endif
        }
#ifdef NMI
        virtual double getLocalPotential(IndexType targetIndex, double entropyX, double entropyY){
            if (!this->nIt.Size()) {
                cout<<VAR(this->nIt.Size())<<endl;
            }
            //use iterators
            double result;
            this->nIt.SetLocation(targetIndex);
            m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            m_maskNeighborhoodIterator.SetLocation(targetIndex);
            
            double count=0;
            //compute joint histogram
            typename HistogramType::Pointer jointHist=HistogramType::New();
            jointHist->SetMeasurementVectorSize(2);
            typename HistogramType::SizeType histSize(2);
            histSize.Fill(m_numberOfBins);

            //find bounds
            MeasurementVectorType m_LowerBound;
            MeasurementVectorType m_UpperBound;
            m_LowerBound.SetSize(2);
            m_UpperBound.SetSize(2);
            m_LowerBound[0]=numeric_limits<PixelType>::max();
            m_LowerBound[1]=numeric_limits<PixelType>::max();
            m_UpperBound[0]=numeric_limits<PixelType>::min();
            m_UpperBound[1]=numeric_limits<PixelType>::min();
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                bool inside=m_maskNeighborhoodIterator.GetPixel(i);
                if (inside && inBounds){
                    double f=this->nIt.GetPixel(i);
                    m_LowerBound[0]=f<m_LowerBound[0]?f:m_LowerBound[0];
                    m_LowerBound[1]=m<m_LowerBound[1]?m:m_LowerBound[1];
                    m_UpperBound[0]=f>m_UpperBound[0]?f:m_UpperBound[0];
                    m_UpperBound[1]=m>m_UpperBound[1]?f:m_UpperBound[1];

                }
            }                                                           
            LOGV(40)<<VAR(histSize)<<" "<<VAR(m_LowerBound)<<" "<<VAR(m_UpperBound)<<endl;
            jointHist->Initialize(histSize,m_LowerBound,m_UpperBound);                    
            typename HistogramType::MeasurementVectorType sample;
            sample.SetSize(2);
            double insideCount=0.0;
            //compute joint histogram
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                insideCount+=inBounds;
                bool inside=m_maskNeighborhoodIterator.GetPixel(i);
                if (inside && inBounds){
                    double f=this->nIt.GetPixel(i);
                    sample[0] = f;
                    sample[1] = m;
                    LOGV(80)<<VAR(sample)<<endl;
                    jointHist->IncreaseFrequencyOfMeasurement(sample, 1);
                    count++;
                }
            }
            
            if (count){
                MeasureType jointEntropy = NumericTraits< MeasureType >::Zero;
                typedef typename NumericTraits< HistogramFrequencyType >::RealType HistogramFrequencyRealType;
                
                HistogramFrequencyRealType totalFreq =
                    static_cast< HistogramFrequencyRealType >( jointHist->GetTotalFrequency() );
                
                
                HistogramIteratorType it = jointHist->Begin();
                HistogramIteratorType end = jointHist->End();
                while ( it != end )
                    {
                        HistogramFrequencyRealType freq =
                            static_cast< HistogramFrequencyRealType >( it.GetFrequency() );
                        LOGV(70)<<VAR(freq)<<endl;
                        if ( freq > 0 )
                            {
                                jointEntropy += freq * vcl_log(freq);
                            }
                        ++it;
                    }
                
                jointEntropy = -jointEntropy / static_cast< MeasureType >( totalFreq )
                    + vcl_log(totalFreq);
                LOGV(40)<<VAR(jointEntropy)<<" "<<VAR(entropyX)<<" "<<entropyY<<endl;
                
                if ((entropyX + entropyY)==0.0){
                    cout<<"strange"<<endl;
                    result=0;
                }
                else{
                    result =2  - 2.0* jointEntropy/ ( entropyX + entropyY ) ;
                }
            }else{ result=0;}
            //                    NMI= ( entropyX + entropyY )/jointEntropy ;
            
            result=-result;
            
            //            result=min(this->m_threshold,result);
            LOGV(15)<<VAR(result)<<endl;
            return result*insideCount/this->nIt.Size();
        }
        virtual double getLocalPotentialNCC(IndexType targetIndex){

            double result;
            this->nIt.SetLocation(targetIndex);
            m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            m_maskNeighborhoodIterator.SetLocation(targetIndex);
            
            double count=0;
            double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                double m=m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                bool inside=m_maskNeighborhoodIterator.GetPixel(i);
                if (inside && inBounds){
                    double f=this->nIt.GetPixel(i);
                    sff+=f*f;
                    smm+=m*m;
                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;

                }
            }
            double NCC=0;
            if (count){
                sff -= ( sf * sf / count );
                smm -= ( sm * sm / count );
                sfm -= ( sf * sm / count );
                if (smm*sff>0){
                    NCC=1.0*sfm/sqrt(smm*sff);
                
                }
            }
            //result=result>0.5?0.5:result; 
            if (this->LOGPOTENTIAL){
                result=((1+NCC)/2);
                result=result>0?result:0.00000001;
                result=-log(result);
            }else{
                result=(1-(NCC))/2;
            }
            result=min(this->m_threshold,result);
            
            return result;
        }
#endif
    };//FastUnaryPotentialRegistrationNMI
}//namespace
#endif /* POTENTIALS_H_ */
