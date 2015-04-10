#include "Log.h"
/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _COHERENCEPAIRWISEPOTENTIAL_H_
#define _COHERENCEPAIRWISEPOTENTIAL_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "itkVector.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkStatisticsImageFilter.h"
#include "itkThresholdImageFilter.h"

namespace SRS{



    template<class TImage>
    class PairwisePotentialCoherence : public itk::Object{
    public:
        //itk declarations
        typedef PairwisePotentialCoherence            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename FloatImageType::ConstPointer FloatImageConstPointerType;
        
        typedef typename  itk::Vector<float,ImageType::ImageDimension>  LabelType;
        typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
        typedef typename LabelImageType::Pointer LabelImagePointerType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef typename ImageType::PointType PointType;
        typedef itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
        typedef itk::LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
        typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
        
        SizeType m_targetSize,m_atlasSize;
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;


    protected:
        ConstImagePointerType m_targetImage, m_atlasImage, m_atlasSegmentationImage;
        SegmentationInterpolatorPointerType m_atlasSegmentationInterpolator;
        std::vector<FloatImageInterpolatorPointerType> m_atlasDistanceTransformInterpolators;
        std::vector<FloatImagePointerType> m_distanceTransforms;
        std::vector<double> m_minDists;
        ImageInterpolatorPointerType  m_atlasInterpolator;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        double m_asymm;
      
        double sigma1, sigma2, mean1, mean2, m_tolerance,maxDist,minDist, mDistTarget,mDistSecondary;
        int m_nSegmentationLabels,m_auxiliaryLabel;

    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialCoherence, Object);

        PairwisePotentialCoherence(){
            m_haveLabelMap=false;
            m_asymm=1;
            m_tolerance=9999999999.0;
            m_auxiliaryLabel=1;
        }
        virtual void freeMemory(){
        }
        void SetAuxLabel(int l){m_auxiliaryLabel=l;}
        void SetNumberOfSegmentationLabels(int n){m_nSegmentationLabels=n;}
        void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;m_haveLabelMap=true;}
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
        
        void SetAtlasInterpolator(ImageInterpolatorPointerType atlasImage){
            m_atlasInterpolator=atlasImage;
        }
        void SetAtlasSegmentationInterpolator(SegmentationInterpolatorPointerType atlasSegmentation){
            m_atlasSegmentationInterpolator=atlasSegmentation;
        }
    	void SetAtlasImage(ConstImagePointerType atlasImage){
            m_atlasImage=atlasImage;
            m_atlasSize=m_atlasImage->GetLargestPossibleRegion().GetSize();
            m_atlasInterpolator=ImageInterpolatorType::New();
            m_atlasInterpolator->SetInputImage(m_atlasImage);//,m_atlasImage->GetLargestPossibleRegion());
            LOGV(3)<<"Dimensions of atlas image for coherence potentials "<<m_atlasSize<<endl;
        }
        void SetTargetImage(ConstImagePointerType targetImage){
            m_targetImage=targetImage;
            m_targetSize=m_targetImage->GetLargestPossibleRegion().GetSize();
        }
        void SetAsymmetryWeight(double as){
            m_asymm=1-as;
        }
#if 0
        void SetDistanceTransform(FloatImagePointerType dt){
            m_distanceTransform=dt;
            m_atlasDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            m_atlasDistanceTransformInterpolator->SetInputImage(dt);
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(dt);
            filter->Update();
            sigma1=filter->GetSigma();
            mean1=filter->GetMean();
            LOG<<"distance transform main segmentation sigma :"<<sigma1<<endl;

        }
        void SetBackgroundDistanceTransform(FloatImagePointerType dt){
            m_atlasBackgroundDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            m_atlasBackgroundDistanceTransformInterpolator->SetInputImage(dt);
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(dt);
            filter->Update();
            sigma2=filter->GetSigma();
            mean2=fabs(filter->GetMean());
            LOG<<"distance transform background segmentation sigma :"<<sigma2<<endl;
        }

        FloatImagePointerType GetDistanceTransform(){return  m_distanceTransform;}
#endif        
        FloatImagePointerType GetDistanceTransform(int label){return  m_distanceTransforms[label];}

        virtual void SetAtlasSegmentation(ConstImagePointerType segImage, double scale=1.0){
            logSetStage("Coherence setup");
            m_atlasSegmentationInterpolator= SegmentationInterpolatorType::New();
            m_atlasSegmentationInterpolator->SetInputImage(segImage);
            m_atlasSegmentationImage=segImage;
            typename itk::StatisticsImageFilter< ImageType >::Pointer maxFilter= itk::StatisticsImageFilter< ImageType >::New() ;

            maxFilter->SetInput(segImage);
            maxFilter->Update();
            m_nSegmentationLabels=max( m_nSegmentationLabels,(int)maxFilter->GetMaximumOutput()->Get()+1);
            if (m_nSegmentationLabels<3){m_auxiliaryLabel=-1;}
            if (maxFilter->GetMaximumOutput()->Get()+1>m_nSegmentationLabels){
                LOG<<"WARNING: large number of segmentation labels in atlas segmentation :"<<VAR(m_nSegmentationLabels)<<endl;
                LOG<<VAR(maxFilter->GetMaximumOutput()->Get()+1)<<endl;
                LOGI(6,ImageUtils<ImageType>::writeImage("multilabelAtlas.nii",segImage));
            }
            m_distanceTransforms= std::vector<FloatImagePointerType>( m_nSegmentationLabels ,NULL);
            m_atlasDistanceTransformInterpolators = std::vector<FloatImageInterpolatorPointerType>( m_nSegmentationLabels ,NULL);
            m_minDists=std::vector<double> ( m_nSegmentationLabels ,-1);;

            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();

            for (int l=0;l< m_nSegmentationLabels;++l){
                //get distance transform to foreground label
                FloatImagePointerType dt1=getDistanceTransform(segImage,l);
                //save image for debugging
             
                   
		ostringstream dtFilename;
		dtFilename<<"dt"<<l<<".nii";
		LOGI(6,                        ImageUtils<FloatImageType>::writeImage(dtFilename.str().c_str(),FloatImageConstPointerType(dt1)));
                    
                
                m_distanceTransforms[l]=dt1;
                //feed DT into interpolator
                FloatImageInterpolatorPointerType dtI=FloatImageInterpolatorType::New();
                dtI->SetInputImage(dt1);
                m_atlasDistanceTransformInterpolators[l]=dtI;
                filter->SetInput(dt1);
                filter->Update();
               
                m_minDists[l]=fabs(filter->GetMinimumOutput()->Get());
                LOGV(3)<<"Maximal radius of target object: "<< m_minDists[l]<<endl;
            }
            logResetStage;
        }

        ConstImagePointerType getAtlasSegmentation(){return  m_atlasSegmentationImage;}
        FloatImagePointerType getDistanceTransform(ConstImagePointerType segmentationImage, int value){
            assert(segmentationImage.IsNotNull());
            typedef typename itk::SignedMaurerDistanceMapImageFilter< ImageType, FloatImageType > DistanceTransformType;
            typename DistanceTransformType::Pointer distanceTransform=DistanceTransformType::New();
            typedef typename  itk::ImageRegionConstIterator<ImageType> ImageConstIterator;
            typedef typename  itk::ImageRegionIterator<ImageType> ImageIterator;
            ImagePointerType newImage=ImageUtils<ImageType>::createEmpty(segmentationImage);
            ImageConstIterator imageIt(segmentationImage,segmentationImage->GetLargestPossibleRegion());        
            ImageIterator imageIt2(newImage,newImage->GetLargestPossibleRegion());        
            int count=0,countAll=0;
            
            for (imageIt.GoToBegin(),imageIt2.GoToBegin();!imageIt.IsAtEnd();++imageIt, ++imageIt2,++countAll){
                float val=imageIt.Get();
                imageIt2.Set(val==value);
                count+=(val==value);
            }
            LOGV(8)<<VAR(countAll)<<" "<<VAR(count)<<" "<<VAR(value)<<endl;
            FloatImagePointerType positiveDM;
            if (count!=0){
                //distanceTransform->InsideIsPositiveOn();
                distanceTransform->SetInput(newImage);
                distanceTransform->SquaredDistanceOff ();
                distanceTransform->UseImageSpacingOn();
                distanceTransform->Update();
            typedef typename  itk::ImageRegionIterator<FloatImageType> FloatImageIterator;

            positiveDM=FilterUtils<FloatImageType>::lowerThresholding(distanceTransform->GetOutput(),0.0);

            }else{
                positiveDM=FilterUtils<ImageType,FloatImageType>::createEmptyFrom(newImage);
                positiveDM->FillBuffer(0.0);
            }

            ImageUtils<FloatImageType>::multiplyImage(positiveDM,1.0/this->m_tolerance);
            return  positiveDM;
        }

        void SetTolerance(double t){m_tolerance=t;}


        //edge from registration to segmentation
        inline virtual  double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2;//(targetIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;

            typename ImageType::PointType p;
            this->m_targetImage->TransformIndexToPhysicalPoint(targetIndex1,p);

            for (int i2= 0; i2 < p.Dimension; i2++) {
              p[i2] +=disp[i2];
            }
            this->m_atlasSegmentationImage->TransformPhysicalPointToContinuousIndex(p,idx2);
            int deformedAtlasSegmentation=-1;
            if (!m_atlasSegmentationInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            }
            deformedAtlasSegmentation=int(m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2));
            if (segmentationLabel!=deformedAtlasSegmentation){ 
                double dist=m_atlasDistanceTransformInterpolators[segmentationLabel]->EvaluateAtContinuousIndex(idx2);       
                //double dist2=m_atlasDistanceTransformInterpolators[deformedAtlasSegmentation]->EvaluateAtContinuousIndex(idx2);
                result=std::max(0.0,dist);
            }

            //bool targetSegmentation=(segmentationLabel==this->m_nSegmentationLabels-1 ||  deformedAtlasSegmentation == this->m_nSegmentationLabels-1 );
            //bool auxiliarySegmentation=!targetSegmentation && (segmentationLabel || deformedAtlasSegmentation);
	    bool auxiliarySegmentation=this->m_nSegmentationLabels>2 && ((segmentationLabel == this->m_auxiliaryLabel ) || (deformedAtlasSegmentation == this->m_auxiliaryLabel));
	    
            if (auxiliarySegmentation){
	      result=min(result,1.0);
                LOGV(16)<<VAR(result)<<endl;
            }
            result=0.5*result*result;//exp(result)-1;
           
            result=min(999999.0,result);
            
            return result;
        }

        //Return minimum potential for segmentation node index given a zero displacement
        inline virtual double getMinZeroPotential(PointType pt){
            double minPot=std::numeric_limits<double>::max();
            LabelType disp; disp.Fill(0.0);
            IndexType idx;
            GetDistanceTransform(0)->TransformPhysicalPointToIndex(pt,idx);
            for (int i=1;i<this->m_nSegmentationLabels;++i){
                if (i!=this->m_auxiliaryLabel || this->m_nSegmentationLabels<=2){
                    //double pot=sqrt(this->getPotential(idx,bufferIdx,disp,i));
                    double pot=fabs(GetDistanceTransform(i)->GetPixel(idx));
                    if (pot<minPot){
                        minPot=pot;
                    }
                }
 
            }
	    LOGV(17)<<VAR(minPot)<<" "<<VAR(this->m_auxiliaryLabel)<<" "<<VAR(this->m_nSegmentationLabels)<<std::endl;
            return minPot;
        }
    };//class

     template<class TImage>
    class PairwisePotentialMultilabelCoherence :public PairwisePotentialCoherence<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialMultilabelCoherence            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        
        typedef typename  itk::Vector<float,ImageType::ImageDimension>  LabelType;
        typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
        typedef typename LabelImageType::Pointer LabelImagePointerType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
        typedef itk::LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
        typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
        
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
 
    public:
        itkNewMacro(Self);

          inline virtual  double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2(targetIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;

            typename ImageType::PointType p;
            this->m_targetImage->TransformIndexToPhysicalPoint(targetIndex1,p);
            
            itk::Point<float,ImageType::ImageDimension> newDisplacement;
            for (int i2 = 0; i2 < ImageType::ImageDimension; i2++) {
              newDisplacement[i2] = p[i2] + disp[i2];
            }

            this->m_atlasSegmentationImage->TransformPhysicalPointToContinuousIndex(newDisplacement,idx2);
            int deformedAtlasSegmentation=-1;
            if (!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            }
            deformedAtlasSegmentation=int(this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2));
            if (segmentationLabel!=deformedAtlasSegmentation){
#if 1
                double dist=this->m_atlasDistanceTransformInterpolators[segmentationLabel]->EvaluateAtContinuousIndex(idx2);       
                result=std::max(0.0,dist);
#else
		double dist=this->m_atlasDistanceTransformInterpolators[segmentationLabel]->EvaluateAtContinuousIndex(idx2);       
		double dist2=this->m_atlasDistanceTransformInterpolators[deformedAtlasSegmentation]->Evaluate(p);
                result=std::max(dist2,dist);
#endif
            }
	    ///do not penalize confusion of background and auxiliary label that strongly?
	    bool auxiliarySegmentation=(this->m_nSegmentationLabels>2) && ((segmentationLabel == this->m_auxiliaryLabel && deformedAtlasSegmentation == 0 ) || (deformedAtlasSegmentation == this->m_auxiliaryLabel && segmentationLabel == 0));
	    if (auxiliarySegmentation){
	      result=std::min(result,1.0);
	      LOGV(16)<<VAR(result)<<endl;

            }
            result=0.5*result*result;//exp(result)-1;
           
            result=min(999999.0,result);
            LOGV(8)<<VAR(segmentationLabel)<<" "<<VAR(deformedAtlasSegmentation)<<" "<<VAR(result)<<std::endl;
            return result;
        }

     };//class PairwisePotentialMultilabelCoherence
    template<class TImage>
    class PairwisePotentialSigmoidCoherence :public PairwisePotentialCoherence<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSigmoidCoherence            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        
        typedef typename  itk::Vector<float,ImageType::ImageDimension>  LabelType;
        typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
        typedef typename LabelImageType::Pointer LabelImagePointerType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
        typedef itk::LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
        typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
        
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
 
    public:
        itkNewMacro(Self);

        //edge from registration to segmentation
        inline virtual  double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2(targetIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;

            typename ImageType::PointType p;
            this->m_targetImage->TransformIndexToPhysicalPoint(targetIndex1,p);
            p +=disp;//+this->m_baseLabelMap->GetPixel(targetIndex1);
            this->m_atlasSegmentationImage->TransformPhysicalPointToContinuousIndex(p,idx2);
            int deformedAtlasSegmentation=-1;
            if (!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            }
            deformedAtlasSegmentation=int(this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2));
            if (segmentationLabel!=deformedAtlasSegmentation){ 
                double dist=this->m_atlasDistanceTransformInterpolators[segmentationLabel]->EvaluateAtContinuousIndex(idx2);
                result=dist;//-this->m_minDists[segmentationLabel];
            }
            
            //return 1.0/(1.0+exp(-(result-this->m_tolerance)));
            return 0.5*result*result;
        }
    };//class


    template<class TImage>
    class PairwisePotentialBoneCoherence :public PairwisePotentialCoherence<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialBoneCoherence            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        
        typedef typename  itk::Vector<float,ImageType::ImageDimension>  LabelType;
        typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
        typedef typename LabelImageType::Pointer LabelImagePointerType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
        typedef itk::LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
        typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
        
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
 
    public:
        itkNewMacro(Self);

        void SetAtlasSegmentation(ConstImagePointerType segImage, double scale=1.0){
            if (scale !=1.0 ){
                segImage=FilterUtils<ImageType>::NNResample(segImage,scale,false);
            }
            FloatImagePointerType dt1=getDistanceTransform(segImage, 1);
            this->m_atlasDistanceTransformInterpolator=FloatImageInterpolatorType::New();
            this->m_atlasDistanceTransformInterpolator->SetInputImage(dt1);
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(dt1);
            filter->Update();
            this->sigma1=filter->GetSigma();
            this->mean1=filter->GetMean();
            this->m_distanceTransform=dt1;

        
            typedef itk::ThresholdImageFilter <FloatImageType>
                ThresholdImageFilterType;
            typename ThresholdImageFilterType::Pointer thresholdFilter
                = ThresholdImageFilterType::New();
            thresholdFilter->SetInput(dt1);
            thresholdFilter->ThresholdOutside(0, 1000);
            thresholdFilter->SetOutsideValue(1000);
            typedef itk::RescaleIntensityImageFilter<FloatImageType,ImageType> CasterType;
            typename CasterType::Pointer caster=CasterType::New();
            caster->SetOutputMinimum( numeric_limits<typename ImageType::PixelType>::min() );
            caster->SetOutputMaximum( numeric_limits<typename ImageType::PixelType>::max() );
            caster->SetInput(thresholdFilter->GetOutput());
            caster->Update();
            ImagePointerType output=caster->GetOutput();
           
        }

        inline virtual  double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2(targetIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;
#ifdef PIXELTRANSFORM
           
            idx2+= disp;
            if (m_baseLabelMap){
                itk::Vector<float,ImageType::ImageDimension> baseDisp=m_baseLabelMap->GetPixel(targetIndex2);
                idx2+=baseDisp;
            }
#else
            typename ImageType::PointType p;
            this->m_baseLabelMap->TransformIndexToPhysicalPoint(targetIndex1,p);
            p +=disp+this->m_baseLabelMap->GetPixel(targetIndex1);
            this->m_baseLabelMap->TransformPhysicalPointToContinuousIndex(p,idx2);
#endif
            int deformedAtlasSegmentation=-1;
            double distanceToDeformedSegmentation;
            if (!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            } 
            deformedAtlasSegmentation=int(this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2));
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;
            double atlasIntens=this->m_atlasInterpolator->EvaluateAtContinuousIndex(idx2);
            bool p_bone_SA=atlasIntens>bone;
            bool p_tiss_SA=atlasIntens<tissue;
            distanceToDeformedSegmentation=this->m_atlasDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2);

            if (deformedAtlasSegmentation>0){
                if (segmentationLabel== 0){
                    //atlas 
                    result=fabs(distanceToDeformedSegmentation)/((this->sigma1));//1;
                    
                }else if (segmentationLabel ==this->m_nSegmentationLabels - 1 ){
                    //agreement
                    result=0;
                }
                else if (segmentationLabel){
                    //never label different structure
                    if(p_bone_SA){
                        result=99999999999;
                    }
                }
                
            }else{
                //atlas is not segmented...
                if (segmentationLabel== this->m_nSegmentationLabels - 1){
                    //if we want to label foreground, cost is proportional to distance to atlas foreground
                    distanceToDeformedSegmentation=fabs(this->m_atlasDistanceTransformInterpolator->EvaluateAtContinuousIndex(idx2));
#if 1       
                    if (distanceToDeformedSegmentation>this->m_tolerance){
                        result=1000;
                    }
                    else
#endif
                        {
                            result=(distanceToDeformedSegmentation)/1.0;//((this->sigma1));//1;
                            
                        }
                    if(p_bone_SA){
                        //never ever label as foreground bone in case the atlas is not labeled as bone, but has bone-like intensities!!
                        result=99999999999;
                    }
                }else if (segmentationLabel ){
                    //now we've got a non-target segmentation label and what we do with it will be decided based on the intensities of atlas and image
                    //double targetIntens=this->m_targetImage->GetPixel(targetIndex1);
                    if(false && p_tiss_SA){
                        //if atlas is clearly not bone, don;t allow any bone label
                        result=99999999999;
                    }else{
                        //otherwise, alternative bone label gets more probably the further the point is away from the deformed GT segmentation?
                        if (!distanceToDeformedSegmentation){
                            //distance is even zero, dont allow label 1 at all
                            result=99999999999;
                        }else{
                            result=1.0/(distanceToDeformedSegmentation);
                            //LOG<<result<<endl;
                        }
                    }
                }
                //LOG<<segmentationLabel<<" "<<result<<endl;
            }
            //LOG<<result<<endl;
            return result;
        }
    };//class
    template<class TImage>
    class PairwisePotentialCoherenceBinary :public PairwisePotentialCoherence<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialCoherenceBinary            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        
        typedef typename  itk::Vector<float,ImageType::ImageDimension>  LabelType;
        typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
        typedef typename LabelImageType::Pointer LabelImagePointerType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
        typedef typename ImageInterpolatorType::Pointer ImageInterpolatorPointerType;
        typedef itk::LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
        typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
        typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
        typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
        
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
 
    public:
        itkNewMacro(Self);

        virtual void SetAtlasSegmentation(ConstImagePointerType segImage, double scale=1.0){
            logSetStage("Coherence setup");
            this->m_atlasSegmentationInterpolator= SegmentationInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(segImage);
            this->m_atlasSegmentationImage=segImage;
            typename itk::StatisticsImageFilter< ImageType >::Pointer maxFilter= itk::StatisticsImageFilter< ImageType >::New() ;

            maxFilter->SetInput(segImage);
            maxFilter->Update();
            this->m_nSegmentationLabels=maxFilter->GetMaximumOutput()->Get()+1;
            LOGV(3)<<VAR( this->m_nSegmentationLabels)<<endl;
            if (this->m_nSegmentationLabels>3){
                LOG<<"WARNING: large number of segmentation labels in atlas segmentation :"<<VAR(this->m_nSegmentationLabels)<<endl;
            }
          
            logResetStage;
        }
        inline virtual  double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement, int segmentationLabel){
            double result=0;
            ContinuousIndexType idx2(targetIndex2);
            itk::Vector<float,ImageType::ImageDimension> disp=displacement;

            typename ImageType::PointType p;
            this->m_targetImage->TransformIndexToPhysicalPoint(targetIndex1,p);
            p +=disp;//+this->m_baseLabelMap->GetPixel(targetIndex1);
            this->m_atlasSegmentationImage->TransformPhysicalPointToContinuousIndex(p,idx2);
            int deformedAtlasSegmentation=-1;
            if (!this->m_atlasSegmentationInterpolator->IsInsideBuffer(idx2)){
                for (int d=0;d<ImageType::ImageDimension;++d){
                    if (idx2[d]>=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetEndContinuousIndex()[d]-0.5;
                    }
                    else if (idx2[d]<this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]){
                        idx2[d]=this->m_atlasSegmentationInterpolator->GetStartContinuousIndex()[d]+0.5;
                    }
                }
            }
            deformedAtlasSegmentation=int(this->m_atlasSegmentationInterpolator->EvaluateAtContinuousIndex(idx2));
            if (segmentationLabel!=deformedAtlasSegmentation){ 
                result=1;
            }

            bool auxiliarySegmentation=(this->m_nSegmentationLabels>2) && ((segmentationLabel == this->m_auxiliaryLabel ) || (deformedAtlasSegmentation == this->m_auxiliaryLabel));
	    bool targetSegmentation = ! auxiliarySegmentation;
            if (targetSegmentation){
                result*=2;//1.0+exp(max(1.0,40.0/this->m_tolerance)-1.0);
            }

            return result;
        }
    };//class
   
}//namespace
#endif /* POTENTIALS_H_ */
