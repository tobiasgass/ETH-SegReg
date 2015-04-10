
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
#include "itkTranslationTransform.h"
#include "TransformationUtils.h"
#include "Log.h"
#include <limits>
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkHistogram.h"
#include "itkScalarImageToHistogramGenerator.h"
#include "itkIdentityTransform.h"
#include "Metrics.h"
#include "itkPointSet.h"
#include <iostream>
#include <fstream>
#include "itkPointsLocator.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "SegmentationMapper.hxx"

namespace SRS{

  /** \brief
   * Local NCC registration potential, also serves as base class for the remaining registration potential classes
   */
    template<class TImage>
    class UnaryPotentialRegistrationNCC : public itk::Object{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCC            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        typedef typename ImageType::PointType PointType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;


        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;

        SizeType m_targetSize,m_atlasSize;
    protected:
        ConstImagePointerType m_targetImage, m_atlasImage;
        ConstImagePointerType m_scaledTargetImage, m_scaledAtlasImage,m_atlasMaskImage,m_scaledAtlasMaskImage;
        InterpolatorPointerType m_atlasInterpolator;
        DisplacementImagePointerType m_baseDisplacementMap;
        bool m_haveDisplacementMap;
        bool radiusSet;
        RadiusType m_radius, m_scaledRadius;
        SpacingType m_coarseImageSpacing;
        ImageNeighborhoodIteratorType nIt;
        double m_scale;
        SizeType m_scaleITK,m_invertedScaleITK;
        double  m_threshold;
        bool LOGPOTENTIAL;
        bool m_noOutSidePolicy;
        bool m_useGradient;
        double m_alpha;
        bool m_normalizeImages;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialNCC, Object);

        UnaryPotentialRegistrationNCC(){
            m_haveDisplacementMap=false;
            radiusSet=false;
            m_targetImage=NULL;
            m_atlasImage=NULL;
            m_atlasMaskImage=NULL;
            m_scaledAtlasMaskImage=NULL;
            m_scale=1.0;
            m_scaleITK.Fill(1.0);
            m_threshold=std::numeric_limits<double>::max();
            LOGPOTENTIAL=false;
            m_noOutSidePolicy = false;
            m_useGradient=false;
            m_alpha=0.0;
            m_normalizeImages=0.0;
        }
        ~UnaryPotentialRegistrationNCC(){
            //delete nIt;
        }
        void SetAlpha(double alpha){m_alpha=alpha;}

        virtual void Compute(){}
        virtual void setDisplacements(std::vector<DisplacementType> displacements){}
        virtual void setCoarseImage(ImagePointerType img){}
        virtual void setThreshold(double t){m_threshold=t;}
        virtual void setLogPotential(bool b){LOGPOTENTIAL=b;}
        virtual void setNoOutsidePolicy(bool b){ m_noOutSidePolicy = b;}
        virtual void setNormalizeImages(bool b){m_normalizeImages=b;}
        virtual void Init(){

            assert(m_targetImage);
            assert(m_atlasImage);
            if ( m_scale!=1.0){
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(m_targetImage,m_scale,true);
                m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample(m_atlasImage,m_scale,true);
                if (m_atlasMaskImage.IsNotNull()){
                    m_scaledAtlasMaskImage=FilterUtils<ImageType>::NNResample(m_atlasMaskImage,m_scale,false);                }
            }else{
                m_scaledTargetImage=m_targetImage;
                m_scaledAtlasImage=m_atlasImage;
                m_scaledAtlasMaskImage=m_atlasMaskImage;
            }
            if (!radiusSet){
                LOG<<"Radius must be set before calling registrationUnaryPotential.Init()"<<endl;
                exit(0);
            }
                
            for (int d=0;d<ImageType::ImageDimension;++d){
                m_scaledRadius[d]=max(m_scale*m_radius[d]-1,1.0);
            }
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
        
        void SetBaseDisplacementMap(DisplacementImagePointerType blm, double scale=1.0){
            m_baseDisplacementMap=blm;m_haveDisplacementMap=true;
            if (blm->GetLargestPossibleRegion().GetSize()!=m_scaledTargetImage->GetLargestPossibleRegion().GetSize()){
                m_baseDisplacementMap=TransfUtils<ImageType>::bSplineInterpolateDeformationField(blm,m_scaledTargetImage);
            }
        }
        DisplacementImagePointerType GetBaseDisplacementMap(DisplacementImagePointerType blm){return m_baseDisplacementMap;}
        virtual void SetAtlasImage(ImagePointerType atlasImage){
            SetAtlasImage(ConstImagePointerType(atlasImage));
        }

    	virtual void SetAtlasImage(ConstImagePointerType atlasImage){
            if (! m_useGradient){ 
                if (m_normalizeImages){
                    LOGV(1)<<"Normalizing atlas image to zero mean unit variance"<<endl;
                    m_atlasImage=FilterUtils<ImageType>::normalizeImage(atlasImage);
                }
                else
                    m_atlasImage=atlasImage;
            }else{
                if (m_normalizeImages){
                    LOGV(1)<<"Normalizing atlas gradient image to zero mean unit variance"<<endl;
                    m_atlasImage=FilterUtils<ImageType>::gradient(FilterUtils<ImageType>::normalizeImage(atlasImage));
                }
                else
                    m_atlasImage=FilterUtils<ImageType>::gradient(atlasImage);
            }
            m_atlasSize=m_atlasImage->GetLargestPossibleRegion().GetSize();
        }

        virtual void SetAtlasMaskImage(ConstImagePointerType atlasMaskImage){
            m_atlasMaskImage=atlasMaskImage;

        }
        virtual void SetTargetImage(ConstImagePointerType targetImage){
            if (! m_useGradient){ 
                if (m_normalizeImages){
                    LOGV(1)<<"Normalizing target image to zero mean unit variance"<<endl;
                    m_targetImage=FilterUtils<ImageType>::normalizeImage(targetImage);
                }
                else
                    m_targetImage=targetImage;
            }else{
                if (m_normalizeImages){
                    LOGV(1)<<"Normalizing target gradient image to zero mean unit variance"<<endl;
                    m_targetImage=FilterUtils<ImageType>::gradient(FilterUtils<ImageType>::normalizeImage(targetImage));
                }
                else
                    m_targetImage=FilterUtils<ImageType>::gradient(targetImage);
            }
            m_targetSize=m_targetImage->GetLargestPossibleRegion().GetSize();

        }
        ConstImagePointerType GetTargetImage(){
            return m_scaledTargetImage;
        }
        ConstImagePointerType GetAtlasImage(){
            return m_scaledAtlasImage;
        }
        virtual double getPotential(IndexType targetIndex, DisplacementType disp){
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
                    idx2+=disp+this->m_baseDisplacementMap->GetPixel(neighborIndex)*m_scale;
#else
                          
                    PointType p;
                    m_scaledTargetImage->TransformIndexToPhysicalPoint(neighborIndex,p);

                    for (int i2 = 0; i2 < p.Dimension; i2++) {
                      p[i2] = p[i2] + disp[i2] + this->m_baseDisplacementMap->GetPixel(neighborIndex)[i2];
                    }

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

                    for (int i2= 0; i2 < idx2.Dimension; i2++) {
                      idx2[i2] +=this->m_baseDisplacementMap->GetPixel(neighborIndex)[i2]*m_scale;
                    }

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

 
 /** \brief
   * This registration potential computes all local potentials for a specific displacement at once and caches them, allowing for greater efficiency
   */
    template<class TImage>
    class FastUnaryPotentialRegistrationNCC: public UnaryPotentialRegistrationNCC<TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationNCC            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef UnaryPotentialRegistrationNCC<TImage> Superclass;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const int D=ImageType::ImageDimension;

        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::PixelType PixelType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;


        typedef typename TransfUtils<ImageType>::DeformationFieldType DisplacementImageType;
        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
        //typedef typename itk::ConstNeighborhoodIterator<ImageType,itk::ConstantBoundaryCondition<TImage,TImage> > ImageNeighborhoodIteratorType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
        typedef typename itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
        
        typedef  typename itk::PointSet< PixelType, D >   PointSetType;
        typedef typename  PointSetType::PointsContainer PointsContainerType;
        typedef typename  PointSetType::PointsContainerPointer PointsContainerPointer;
        typedef typename  PointSetType::PointsContainerIterator PointsContainerIterator;
        typedef itk::PointsLocator<PointsContainerType> PointsLocatorType;
        typedef typename PointsLocatorType::Pointer PointsLocatorPointerType;


    protected:
        ImageNeighborhoodIteratorType m_atlasNeighborhoodIterator,m_maskNeighborhoodIterator;
        std::vector<DisplacementType> m_displacements;
        std::vector<FloatImagePointerType> m_potentials;
        DisplacementType m_currentActiveDisplacement;
        FloatImagePointerType m_currentCachedPotentials;
        ImagePointerType m_coarseImage,m_deformedAtlasImage,m_deformedMask;
        double m_averageFixedPotential,m_oldAveragePotential;
        double m_normalizationFactor;
        bool m_normalize;
        PointsContainerPointer m_atlasLandmarks,m_targetLandmarks;
        FloatImagePointerType m_unaryPotentialWeights;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialNCC, Object);
        
        FastUnaryPotentialRegistrationNCC():Superclass(){
            m_normalizationFactor=1.0;
            m_normalize=false;
            m_unaryPotentialWeights=NULL;
            
        }
        void SetPotentialWeights(FloatImagePointerType img){m_unaryPotentialWeights=img;}
        void SetAtlasLandmarks(PointsContainerPointer p){m_atlasLandmarks=p;}
        void SetTargetLandmarks(PointsContainerPointer p){m_targetLandmarks=p;}
        void SetAtlasLandmarksFile(string f){
            SetAtlasLandmarks(readLandmarks(f));
        }
        void SetTargetLandmarksFile(string f){
            SetTargetLandmarks(readLandmarks(f));
        }
        PointsContainerPointer readLandmarks(string f){
            if (f==""){
                return NULL;
            }
            
            typename PointSetType::Pointer  pointSet = PointSetType::New();

            PointsContainerPointer points=pointSet->GetPoints();
            ifstream ifs(f.c_str());
            int i=0;
            while ( ! ifs.eof() ) {
                PointType point;
                bool fullPoint=true;
                for (int d=0;d<D;++d){
                    ifs>>point[d];
                    if (ifs.eof()){
                        fullPoint=false;
                        break;
                    }
                   
                }
                if (fullPoint){
                    points->InsertElement(i, point);
                    ++i;
                }
            } 
            return points;
        }
        virtual void compute(){
            //LOG<<"DEPRECATED, too memory intensive!!"<<endl;
            m_potentials=std::vector<FloatImagePointerType>(m_displacements.size(),NULL);
            m_averageFixedPotential=0;
            for (unsigned int n=0;n<m_displacements.size();++n){
                LOGV(9)<<"cachhing unary registrationpotentials for label " <<n<<endl;
                FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(m_coarseImage);
                DisplacementImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseDisplacementMap);
                translation->FillBuffer( m_displacements[n]);
                DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseDisplacementMap);
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

        //#define PREDEF
        //#define LOCALSIMS
        virtual void initCaching(){
#ifdef PREDEF
            m_deformedAtlasImage=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasImage,this->m_baseDisplacementMap);
            if (this->m_scaledAtlasMaskImage.IsNotNull()){
                m_deformedMask=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasMaskImage,this->m_baseDisplacementMap);
            }else{
                ImagePointerType mask=ImageUtils<ImageType>::createEmpty(this->m_scaledAtlasImage);
                mask->FillBuffer(1);
                m_deformedMask=TransfUtils<ImageType>::warpImage(mask,this->m_baseDisplacementMap);
            }
#endif
        }

        void cachePotentials(DisplacementType displacement){
            LOGV(15)<<"Caching registration unary potential for displacement "<<displacement<<endl;
            PointsLocatorPointerType pointsLocator = PointsLocatorType::New();
            if (m_targetLandmarks.IsNotNull()){
                pointsLocator->SetPoints( m_targetLandmarks );
                pointsLocator->Initialize();
            }
            DisplacementType zeroDisp;
            zeroDisp.Fill(0.0);
            //compute average potential for zero displacement.
            bool computeAverage=(displacement == zeroDisp);
            m_averageFixedPotential=computeAverage?0.0:m_averageFixedPotential;

            FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(m_coarseImage);
            pot->FillBuffer(0.0);
            ImagePointerType deformedAtlas,deformedMask;

#ifndef PREDEF
            DisplacementImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseDisplacementMap);
            translation->FillBuffer( displacement);
            TIME(DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseDisplacementMap));


            typedef typename itk::VectorLinearInterpolateImageFunction<DisplacementImageType, double> DisplacementInterpolatorType;
            typedef typename DisplacementInterpolatorType::Pointer DisplacementInterpolatorPointerType;
            DisplacementInterpolatorPointerType labelInterpolator=DisplacementInterpolatorType::New();
            labelInterpolator->SetInputImage(composedDeformation);

            if (this->m_scaledAtlasMaskImage.IsNotNull()){
                deformedAtlas=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasImage,composedDeformation);
                deformedMask=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasMaskImage,this->m_baseDisplacementMap,true);
                //deformedMask=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasMaskImage,composedDeformation,true);
            }else{
                pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation);
                deformedAtlas=result.first;
                deformedMask=result.second;
            }
            ImageUtils<ImageType>::writeImage("mask.nii",deformedMask);
            ImageUtils<ImageType>::writeImage("deformed.nii",deformedAtlas);
#else
            typedef typename itk::VectorLinearInterpolateImageFunction<DisplacementImageType, double> DisplacementInterpolatorType;
            typedef typename DisplacementInterpolatorType::Pointer DisplacementInterpolatorPointerType;
            DisplacementInterpolatorPointerType labelInterpolator=DisplacementInterpolatorType::New();
            labelInterpolator->SetInputImage(this->m_baseDisplacementMap);
            TIME(deformedAtlas=TransfUtils<ImageType>::translateImage(this->m_deformedAtlasImage,displacement));
            TIME(deformedMask=TransfUtils<ImageType>::translateImage(this->m_deformedMask,displacement,true));
            ImageUtils<ImageType>::writeImage("mask.nii",deformedMask);
            ImageUtils<ImageType>::writeImage("deformed.nii",deformedAtlas);

#endif
            m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
            m_maskNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedMask,deformedMask->GetLargestPossibleRegion());

            LOGV(70)<<VAR(m_atlasNeighborhoodIterator.GetRadius())<<" "<<VAR(deformedAtlas->GetLargestPossibleRegion().GetSize())<<endl;
            LOGV(70)<<VAR(this->nIt.GetRadius())<<" "<<VAR(deformedAtlas->GetLargestPossibleRegion().GetSize())<<endl;

            FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
            ImageIteratorType coarseMaskIterator(FilterUtils<ImageType>::NNResample(deformedMask,m_coarseImage,false),pot->GetLargestPossibleRegion());
            coarseMaskIterator.GoToBegin();
            int c=0;
            double radius=2*m_coarseImage->GetSpacing()[0];
#ifndef LOCALSIMS
            for (coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator,++coarseMaskIterator){
                IndexType coarseIndex=coarseIterator.GetIndex();
                //if the coarse mask is zero, then all mask pixels in the neighborhood are zero and computing the potential does not make sense :)
                if (true || coarseMaskIterator.Get()){
                    bool validPotential=true;
                    
                    if (this->m_noOutSidePolicy){
#if 0
                        //THIS SEEMS SUPER BROKEN!

                        //check if border policy is violated
                        for (int d=0;d<D;++d){
                            int idx=coarseIndex[d];
                            int s=pot->GetLargestPossibleRegion().GetSize()[d] -1;
                            if (idx == 0){
                                double dx=1.0*idx+displacement.GetElement(d);
                                if (dx<0){
                                    validPotential=false;
                                    break;
                                }
                            }else if (idx ==s ){
                                double dx=1.0*idx+displacement.GetElement(d);
                                if (dx>s){
                                    validPotential=false;
                                    break;
                                }
                            }
                        }
#endif
                    }
                    if (validPotential){
                        LOGV(36)<<VAR(coarseIndex)<<" "<<VAR(c)<<endl;
                        PointType point;
                        m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex,point);
                        IndexType targetIndex;
                        this->m_scaledTargetImage->TransformPhysicalPointToIndex(point,targetIndex);
                        double localPot=0;
                        double weight=1.0;
                        if (m_unaryPotentialWeights.IsNotNull()){
                            IndexType weightIndex;
                            m_unaryPotentialWeights->TransformPhysicalPointToIndex(point,weightIndex);
                            weight=m_unaryPotentialWeights->GetPixel(weightIndex);

                        }
                        if (this->m_alpha<1.0) localPot=(1.0-this->m_alpha)*weight*getLocalPotential(targetIndex);
                        if (this->m_alpha>0.0 && m_atlasLandmarks.IsNotNull() && m_targetLandmarks.IsNotNull()){
                            //localPot+=getLandmarkPotential
                            //find landmarks close to point
                            //add distance to target landmark to potential, with weights?
                            
                            typename PointsLocatorType::NeighborsIdentifierType neighborhood;
                            pointsLocator->Search( point , radius, neighborhood );
                            LOGV(1)<<VAR(point)<<" "<<neighborhood.size()<<endl;

                            for (int n=0;n<neighborhood.size();++n){
                                int ptI=neighborhood[n];
                                PointType targetPoint=m_targetLandmarks->GetElement(ptI);
                                PointType atlasPoint=m_atlasLandmarks->GetElement(ptI);
                                LOGV(10)<<VAR(point)<<" "<<VAR(targetPoint)<<endl;
                                //compute linear weight based on distance between grid point and target point
                                double w=1.0;
#ifdef LINEARWEIGHT
                                for (int d=0;d<D;++d){
                                    double axisWeight=max(0.0,1.0-fabs(targetPoint[d]-point[d])/(2*m_coarseImage->GetSpacing()[d]));
                                    w*=axisWeight;
                                }
#else
                                w=exp(- (targetPoint-point).GetNorm()/radius);
#endif
                                //get displacement at targetPoint
                                DisplacementType displacement=labelInterpolator->Evaluate(targetPoint);
                                //get error
                                DisplacementType newVector;
                                for (int i2= 0; i2 < D; i2++) {
                                  newVector[i2] = targetPoint[i2] + displacement[i2] - atlasPoint[i2];
                                }
                                double error= newVector.GetNorm();
                                localPot+=(this->m_alpha)*w*5.0*(error);

                            }
                        }
                        coarseIterator.Set(localPot);
                        if (computeAverage)
                            m_averageFixedPotential+=localPot;
                        ++c;
                    }else{
                        coarseIterator.Set(1e10);
                    }
                }else{
                    //???
                    //this should happen only when the deformed atlas mask is zero at this point, indicating that no displacement should do this (transforms out of the moving image)
                    // why is this only 1 ?
                    coarseIterator.Set(1);
                }
               
            }
#else
            FloatImagePointerType highResPots=localPotentials((ConstImagePointerType)this->m_scaledTargetImage,(ConstImagePointerType)deformedAtlas);
            pot=FilterUtils<FloatImageType>::NNResample(highResPots,pot,false);
#endif
            //LOG<<VAR(c)<<endl;
            if (computeAverage &&c!=0 ){
                
                m_averageFixedPotential/= c;
                m_normalizationFactor=1.0;
                if (m_normalize && (m_averageFixedPotential<std::numeric_limits<float>::epsilon())){
                    m_normalizationFactor= m_normalizationFactor*m_oldAveragePotential/m_averageFixedPotential;
                }
                LOGV(3)<<VAR(m_normalizationFactor)<<endl;
                m_oldAveragePotential=m_averageFixedPotential;
            }
            m_currentCachedPotentials=pot;
            m_currentActiveDisplacement=displacement;

           
        }

        void setDisplacements(std::vector<DisplacementType> displacements){
            m_displacements=displacements;
        }
        void setCoarseImage(ImagePointerType img){m_coarseImage=img;}

        virtual double getPotential(IndexType coarseIndex, unsigned int displacementDisplacement){
            //LOG<<"DEPRECATED BEHAVIOUR!"<<endl;
            return m_potentials[displacementDisplacement]->GetPixel(coarseIndex);
        }
        virtual double getPotential(IndexType coarseIndex){
            //LOG<<"NEW BEHAVIOUR!"<<endl;
            LOGV(90)<<" "<<VAR(m_currentCachedPotentials->GetLargestPossibleRegion().GetSize())<<endl;
            return  m_normalizationFactor*m_currentCachedPotentials->GetPixel(coarseIndex);
        }
        virtual double getPotential(IndexType coarseIndex, DisplacementType l){
            LOG<<"ERROR NEVER CALL THIS"<<endl;
            exit(0);
        }

        virtual FloatImagePointerType localPotentials(ImagePointerType i1, ImagePointerType i2){
            return Metrics<ImageType,FloatImageType>::LNCC((ConstImagePointerType)i1,(ConstImagePointerType)i2,i1->GetSpacing()[0]);
        }
        virtual FloatImagePointerType localPotentials(ConstImagePointerType i1, ConstImagePointerType i2){
            return Metrics<ImageType,FloatImageType>::LNCC(i1,i2,i1->GetSpacing()[0]);
        }

        //virtual double getLocalPotential(IndexType targetIndex){
        inline double getLocalPotential(IndexType targetIndex){

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
            LOGV(15)<<VAR(result*insideCount/this->nIt.Size())<<" "<< VAR(this->nIt.Size()) << std::endl;
            return result*insideCount/this->nIt.Size();
        }
    };//FastUnaryPotentialRegistrationNCC
  
    template<class TImage>
    class FastUnaryPotentialRegistrationSAD: public FastUnaryPotentialRegistrationNCC<TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationSAD            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;

        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
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
            return Metrics<ImageType,FloatImageType,float>::LSAD(i1,i2,i1->GetSpacing()[0]);
            //return Metrics<ImageType,FloatImageType,float>::integralSAD(i1,i2);
            

        }

      
    
        inline double getLocalPotential(IndexType targetIndex){

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
    template<class TImage>
    class FastUnaryPotentialRegistrationSSD: public FastUnaryPotentialRegistrationNCC<TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationSSD            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;


        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
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
            //return Metrics<ImageType,FloatImageType,float>::LSSD(i1,i2,i1->GetSpacing()[0]);
            return Metrics<ImageType,FloatImageType,float>::integralSSD(i1,i2);
        }
     
    
        inline double getLocalPotential(IndexType targetIndex){

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
                }else  if (this->m_noOutSidePolicy &&( !inside && inBounds )){
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
    template<class TImage>
    class FastUnaryPotentialRegistrationNMI: public UnaryPotentialRegistrationNCC<TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationNMI            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef UnaryPotentialRegistrationNCC<TImage> Superclass;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::PixelType PixelType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;

        

        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
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
        
        typedef itk::Statistics::ScalarImageToHistogramGenerator<ImageType> HistGenType ;
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
        std::vector<DisplacementType> m_displacements;
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
#else
            //m_metric->SetNumberOfSpatialSamples();
            m_metric->SetNumberOfHistogramBins(64);
#endif          
            for (unsigned int n=0;n<m_displacements.size();++n){
                LOGV(9)<<"cachhing unary registrationpotentials for label " <<n<<endl;
                FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(m_coarseImage);
                DisplacementImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseDisplacementMap);
                translation->FillBuffer( m_displacements[n]);
                DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseDisplacementMap);
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
            MeasureType entropyX = itk::NumericTraits< MeasureType >::Zero;
            typedef typename itk::NumericTraits< HistogramFrequencyType >::RealType HistogramFrequencyRealType;
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
        void setDisplacements(std::vector<DisplacementType> displacements){
            m_displacements=displacements;
        }
        void setCoarseImage(ImagePointerType img){m_coarseImage=img;}

        virtual double getPotential(IndexType coarseIndex, unsigned int displacementDisplacement){
            return m_potentials[displacementDisplacement]->GetPixel(coarseIndex);
        }
        virtual double getPotential(IndexType coarseIndex, DisplacementType l){
            LOG<<"ERROR NEVER CALL THIS"<<endl;
            exit(0);
        }
        inline double getLocalPotential(IndexType targetIndex){
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
        inline double getLocalPotential(IndexType targetIndex, double entropyX, double entropyY){
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
                MeasureType jointEntropy = itk::NumericTraits< MeasureType >::Zero;
                typedef typename itk::NumericTraits< HistogramFrequencyType >::RealType HistogramFrequencyRealType;
                
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
        inline double getLocalPotentialNCC(IndexType targetIndex){

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


    
    template<class TImage>
    class FastUnaryPotentialRegistrationCategorical: public FastUnaryPotentialRegistrationNCC<TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationCategorical            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef FastUnaryPotentialRegistrationNCC<TImage> Superclass;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const int D=ImageType::ImageDimension;

        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;


        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
        //typedef typename itk::ConstNeighborhoodIterator<ImageType,itk::ConstantBoundaryCondition<TImage,TImage> > ImageNeighborhoodIteratorType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType > ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
        typedef typename itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
    protected:
        DisplacementType m_currentDisplacement;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialCategorical, Object);
        
     
        virtual void compute(){
            //LOG<<"DEPRECATED, too memory intensive!!"<<endl;
            this->m_potentials=std::vector<FloatImagePointerType>(this->m_displacements.size(),NULL);
            this->m_averageFixedPotential=0;
            for (unsigned int n=0;n<this->m_displacements.size();++n){
                LOGV(9)<<"cachhing unary registrationpotentials for label " <<n<<endl;
                FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(this->m_coarseImage);
                DisplacementImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseDisplacementMap);
                translation->FillBuffer( this->m_displacements[n]);
                DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseDisplacementMap);
                ImagePointerType deformedAtlas,deformedMask;
                //todo: NNinterpolation
                pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation,true);
                deformedAtlas=result.first;
                deformedMask=result.second;
                this->m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
                this->m_maskNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedMask,deformedMask->GetLargestPossibleRegion());
                FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
                for (coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator)
                    {
                        IndexType coarseIndex=coarseIterator.GetIndex();
                        PointType point;
                        this->m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex,point);
                        IndexType targetIndex;
                        this->m_scaledTargetImage->TransformPhysicalPointToIndex(point,targetIndex);
                        double localPot=getLocalPotential(targetIndex);
                        coarseIterator.Set(localPot);
                        if (n==this->m_displacements.size()/2){
                            this->m_averageFixedPotential+=localPot;
                        }

                    }
                this->m_potentials[n]=pot;
            }
           
        }
        void setNormalize(bool b){this->m_normalize=b;}
        void resetNormalize(){
            this->m_normalize=false;
            this->m_normalizationFactor=1.0;
        }

        void cachePotentials(DisplacementType displacement){
            LOGV(15)<<"Caching registration unary potential for displacement "<<displacement<<endl;
            m_currentDisplacement=displacement;
            DisplacementType zeroDisp;
            zeroDisp.Fill(0.0);
            //compute average potential for zero displacement.
            bool computeAverage=(displacement == zeroDisp);
            this->m_averageFixedPotential=computeAverage?0.0:this->m_averageFixedPotential;

            FloatImagePointerType pot=FilterUtils<ImageType,FloatImageType>::createEmpty(this->m_coarseImage);
            DisplacementImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseDisplacementMap);
            translation->FillBuffer( displacement);
            DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseDisplacementMap);
            //DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(this->m_baseDisplacementMap,translation);
            ImagePointerType deformedAtlas,deformedMask;

            if (this->m_scaledAtlasMaskImage.IsNotNull()){
                deformedAtlas=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasImage,composedDeformation,true);
                deformedMask=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasMaskImage,this->m_baseDisplacementMap,true);
                //deformedMask=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasMaskImage,composedDeformation,true);
            }else{
                //todo NN interpolation
                pair<ImagePointerType,ImagePointerType> result=TransfUtils<ImageType>::warpImageWithMask(this->m_scaledAtlasImage,composedDeformation,true);
                deformedAtlas=result.first;
                deformedMask=result.second;
            }
            this->m_atlasNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
            this->m_maskNeighborhoodIterator=ImageNeighborhoodIteratorType(this->m_scaledRadius,deformedMask,deformedMask->GetLargestPossibleRegion());

            LOGV(70)<<VAR(this->m_atlasNeighborhoodIterator.GetRadius())<<" "<<VAR(deformedAtlas->GetLargestPossibleRegion().GetSize())<<endl;
            LOGV(70)<<VAR(this->nIt.GetRadius())<<" "<<VAR(deformedAtlas->GetLargestPossibleRegion().GetSize())<<endl;

            FloatImageIteratorType coarseIterator(pot,pot->GetLargestPossibleRegion());
            ImageIteratorType coarseMaskIterator(FilterUtils<ImageType>::NNResample(deformedMask,this->m_coarseImage,false),pot->GetLargestPossibleRegion());
            coarseMaskIterator.GoToBegin();
            int c=0;
            for (coarseIterator.GoToBegin();!coarseIterator.IsAtEnd();++coarseIterator,++coarseMaskIterator){
                IndexType coarseIndex=coarseIterator.GetIndex();
                //if the coarse mask is zero, then all mask pixels in the neighborhood are zero and computing the potential does not make sense :)
                if (coarseMaskIterator.Get()){
                    bool validPotential=true;
                    
                    if (this->m_noOutSidePolicy){
                        //check if border policy is violated
                        for (int d=0;d<D;++d){
                            int idx=coarseIndex[d];
                            int s=pot->GetLargestPossibleRegion().GetSize()[d] -1;
                            if (idx == 0){
                                double dx=1.0*idx+displacement.GetElement(d);
                                if (dx<0){
                                    validPotential=false;
                                    break;
                                }
                            }else if (idx ==s ){
                                double dx=1.0*idx+displacement.GetElement(d);
                                if (dx>s){
                                    validPotential=false;
                                    break;
                                }
                            }
                        }
                    
                    }
                    if (validPotential){
                        LOGV(36)<<VAR(coarseIndex)<<" "<<VAR(c)<<endl;
                        PointType point;
                        this->m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex,point);
                        IndexType targetIndex;
                        this->m_scaledTargetImage->TransformPhysicalPointToIndex(point,targetIndex);
                        double localPot=getLocalPotential(targetIndex);
                        coarseIterator.Set(localPot);
                        if (computeAverage)
                            this->m_averageFixedPotential+=localPot;
                        ++c;
                    }else{
                        coarseIterator.Set(1e10);
                    }
                }else{
                    coarseIterator.Set(1);
                }
               
            }
            this->m_currentCachedPotentials=pot;
            this->m_currentActiveDisplacement=displacement;

            if (computeAverage &&c!=0 ){
                
                this->m_averageFixedPotential/= c;
                if (this->m_normalize){
                    this->m_normalizationFactor= this->m_normalizationFactor*this->m_oldAveragePotential/this->m_averageFixedPotential;
                }
                LOGV(3)<<VAR(this->m_normalizationFactor)<<endl;
                this->m_oldAveragePotential=this->m_averageFixedPotential;
            }
        }


        virtual FloatImagePointerType localPotentials(ConstImagePointerType i1, ConstImagePointerType i2){
            LOG<<"THIS WILL FAIL!"<<endl;
            return NULL;
        }

        inline double getLocalPotential(IndexType targetIndex){

            double result;
            this->nIt.SetLocation(targetIndex);
            this->m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            this->m_maskNeighborhoodIterator.SetLocation(targetIndex);
            double insideCount=0.0;
            double count=0;
            int penalty=0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                int m=this->m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                   
                insideCount+=inBounds;
                bool inside=this->m_maskNeighborhoodIterator.GetPixel(i);
                if (!inside)
                    m=0.0;
                if ( inBounds && (inside|| this->m_noOutSidePolicy)  ){
                    int f=this->nIt.GetPixel(i);
                    penalty+=f!=m;
                    count+=1;

                }
            }
          
            result=penalty;
            
            result=min(this->m_threshold,result);
            LOGV(15)<<VAR(result*insideCount/this->nIt.Size())<<" "<< VAR(this->nIt.Size()) << std::endl;
            return result*insideCount/this->nIt.Size();
        }

        virtual void Init(){


            assert(this->m_targetImage);
            assert(this->m_atlasImage);
            if ( this->m_scale!=1.0){
                this->m_scaledTargetImage=FilterUtils<ImageType>::NNResample(this->m_targetImage,this->m_scale,false);
                this->m_scaledAtlasImage=FilterUtils<ImageType>::NNResample(this->m_atlasImage,this->m_scale,false);
              
                if (this->m_atlasMaskImage.IsNotNull()){
                    this->m_scaledAtlasMaskImage=FilterUtils<ImageType>::NNResample(this->m_atlasMaskImage,this->m_scale,false);                }
            }else{
                this->m_scaledTargetImage=this->m_targetImage;
                this->m_scaledAtlasImage=this->m_atlasImage;
                this->m_scaledAtlasMaskImage=this->m_atlasMaskImage;
            }
            if (!this->radiusSet){
                LOG<<"Radius must be set before calling registrationUnaryPotential.Init()"<<endl;
                exit(0);
            }
                
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=max(this->m_scale*this->m_radius[d],1.0);
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            LOGV(2)<<"Registration unary patch radius " << this->m_radius << " scale "<< this->m_scale << " scaledRadius "<< this->m_scaledRadius << endl;
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->m_atlasInterpolator=InterpolatorType::New();
            this->m_atlasInterpolator->SetInputImage(this->m_scaledAtlasImage);
        }
    };//FastUnaryPotentialRegistrationCategorical
  
    template<class TImage>
    class FastUnaryPotentialRegistrationCategoricalDistanceBased: public FastUnaryPotentialRegistrationCategorical<TImage> {
    public:
        //itk declarations
        typedef FastUnaryPotentialRegistrationCategoricalDistanceBased<TImage>            Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef FastUnaryPotentialRegistrationCategorical<TImage> Superclass;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const int D=ImageType::ImageDimension;

        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;

        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;


        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
        //typedef typename itk::ConstNeighborhoodIterator<ImageType,itk::ConstantBoundaryCondition<TImage,TImage> > ImageNeighborhoodIteratorType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType > ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
        typedef typename itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
        typedef SegmentationMapper<ImageType> SegmentationMapperType;
    private:
        std::vector<FloatImagePointerType> *m_distanceTransforms,* m_scaledDistanceTransforms;
        SegmentationMapperType * m_segmentationMapper;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(FastRegistrationUnaryPotentialCategoricalDistanceBased, Object);
      //    Self(){
      //        SuperClass();
      //        m_distanceTransforms=NULL;
      //        m_scaledDistanceTransforms=NULL;
      //        m_segmentationMapper=NULL;
      //    }
      //    ~Self(){
      //        if (m_distanceTransforms)delete m_distanceTransforms;
      //     
      //        if (m_scaledDistanceTransforms)delete m_scaledDistanceTransforms;
      //        if (m_segmentationMapper)delete  m_segmentationMapper;
      //        ~SuperClass();
      //    }

        inline double getLocalPotential(IndexType targetIndex){

            double result;
            this->nIt.SetLocation(targetIndex);
            this->m_atlasNeighborhoodIterator.SetLocation(targetIndex);
            this->m_maskNeighborhoodIterator.SetLocation(targetIndex);
            double insideCount=0.0;
            double count=0;
            int penalty=0;
            for (unsigned int i=0;i<this->nIt.Size();++i){
                bool inBounds;
                int m=this->m_atlasNeighborhoodIterator.GetPixel(i,inBounds);
                   
                insideCount+=inBounds;
                bool inside=this->m_maskNeighborhoodIterator.GetPixel(i);
                if ( inBounds && (inside|| this->m_noOutSidePolicy)  ){
                    int f=this->nIt.GetPixel(i);
                    if (inside){ //(f!=m) || !inside){
                        IndexType idx=this->nIt.GetIndex(i);
                        LOGV(9)<<VAR(f)<<" "<<VAR(m)<<" "<<VAR((*m_scaledDistanceTransforms)[f]->GetPixel(idx))<<endl;
                        double dist=(*m_scaledDistanceTransforms)[m]->GetPixel(idx);
                        penalty+=1.0*(dist);
//                          IndexType idx2=idx;
//                          for (int d=0;d<D;++d){
//                              idx2[d]=idx[d]-this->m_currentDisplacement[d]/this->m_scaledTargetImage->GetSpacing()[d];
//                          }
//                          dist=(*m_scaledDistanceTransforms)[f]->GetPixel(idx);
//                          penalty+=5*dist*dist;
//  

                    }
                    count+=1;

                }
            }
          
            result=penalty;
            
            result=min(this->m_threshold,result);
            LOGV(15)<<VAR(result*insideCount/this->nIt.Size())<<" "<< VAR(this->nIt.Size()) << std::endl;
            return result*insideCount/this->nIt.Size();
        }

        virtual void Init(){

            assert(this->m_targetImage);
            assert(this->m_atlasImage);
            if ( this->m_scale!=1.0){
                this->m_scaledTargetImage=FilterUtils<ImageType>::NNResample(this->m_targetImage,this->m_scale,false);
                
                this->m_scaledAtlasImage=FilterUtils<ImageType>::NNResample(this->m_atlasImage,this->m_scale,false);
                if (this->m_atlasMaskImage.IsNotNull()){
                    this->m_scaledAtlasMaskImage=FilterUtils<ImageType>::NNResample(this->m_atlasMaskImage,this->m_scale,false);                
                }
               
                          
            }else{
                this->m_scaledTargetImage=this->m_targetImage;
                this->m_scaledAtlasImage=this->m_atlasImage;
                this->m_scaledAtlasMaskImage=this->m_atlasMaskImage;
            }
            int nSegs=FilterUtils<ImageType>::getMax(this->m_targetImage)+1;
            if (m_scaledDistanceTransforms==NULL){
                m_scaledDistanceTransforms=new std::vector<FloatImagePointerType>(nSegs);
            }
            LOGV(3)<<"downsampling distance transforms for "<<nSegs<<" labels; scale:"<<this->m_scale<<endl;
            for (unsigned int s=0;s<nSegs;++s){
                LOGV(3)<<"downsampling distance transforms for label:"<<s<<endl;
                (*m_scaledDistanceTransforms)[s]=FilterUtils<FloatImageType>::NNResample((*m_distanceTransforms)[s],this->m_scale,false);
                //(*m_scaledDistanceTransforms)[s]=FilterUtils<FloatImageType>::LinearResample((*m_distanceTransforms)[s],this->m_scale,true);
            }
            if (!this->radiusSet){
                LOG<<"Radius must be set before calling registrationUnaryPotential.Init()"<<endl;
                exit(0);
            }
           
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=max(this->m_scale*this->m_radius[d],1.0);
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            LOGV(2)<<"Registration unary patch radius " << this->m_radius << " scale "<< this->m_scale << " scaledRadius "<< this->m_scaledRadius << endl;
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->m_atlasInterpolator=InterpolatorType::New();
            this->m_atlasInterpolator->SetInputImage(this->m_scaledAtlasImage);
        }

     	virtual void SetAtlasImage(ConstImagePointerType atlasImage){
            if (m_segmentationMapper!=NULL){
                this->m_atlasImage=m_segmentationMapper->ApplyMap(atlasImage);
            }else{
                this->m_atlasImage=(atlasImage);
            }
            this->m_atlasSize=this->m_atlasImage->GetLargestPossibleRegion().GetSize();
        }

      
        void SetTargetImage(ConstImagePointerType targetImage){
            m_segmentationMapper=new SegmentationMapperType;
            this->m_targetImage=m_segmentationMapper->FindMapAndApplyMap(targetImage);
            int nSegs=FilterUtils<ImageType>::getMax(this->m_targetImage)+1;
            m_distanceTransforms=new std::vector<FloatImagePointerType>(nSegs);
            LOGV(3)<<"Computing distance transforms for "<<nSegs<<" labels"<<endl;
            for (unsigned int s=0;s<nSegs;++s){
                LOGV(3)<<"Computing distance transforms for label:"<<s<<endl;
                //(*m_distanceTransforms)[s]=FilterUtils<ImageType,FloatImageType>::distanceMapByFastMarcher(this->m_targetImage,s);
                (*m_distanceTransforms)[s]=FilterUtils<ImageType,FloatImageType>::distanceMapBySignedMaurer(this->m_targetImage,s);
                if (s==0){
                    ImageUtils<FloatImageType>::writeImage("DT-zero.nii", (*m_distanceTransforms)[s]);
                }
            }
            this->m_targetSize=this->m_targetImage->GetLargestPossibleRegion().GetSize();
            if (this->m_atlasImage.IsNotNull()){
                this->m_atlasImage=m_segmentationMapper->ApplyMap(this->m_atlasImage);
            }

        }
     
    };//FastUnaryPotentialRegistrationCategoricalDistanceBased

    template<class TImage>
    class UnaryPotentialRegistrationSAD : public UnaryPotentialRegistrationNCC< TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationSAD           Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        typedef typename ImageType::PointType PointType;

    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationUnaryPotentialSAD, Object);

        UnaryPotentialRegistrationSAD(){}
        
        virtual double getPotential(IndexType targetIndex, DisplacementType disp){
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
                    p +=disp+this->m_baseDisplacementMap->GetPixel(neighborIndex);
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

#if 0    
    template<class TImage>
    class UnaryPotentialRegistrationNCCWithSegmentationPrior : public UnaryPotentialRegistrationNCC< TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCCWithSegmentationPrior           Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;

        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
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
        
        virtual double getPotential(IndexType targetIndex, DisplacementType disp){
            double result=0;
          
            DisplacementType trueDisplacement=disp;
            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
            }
            DisplacementType baseDisp=this->m_baseDisplacementMap->GetPixel(targetIndex);
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
                    DisplacementType baseDisplacement=this->m_baseDisplacementMap->GetPixel(neighborIndex);
                    DisplacementType finalDisplacement=disp+baseDisplacement*this->m_scale;
                    
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
                        int segmentationPriorDisplacement=(this->m_segmentationPrior->GetPixel(neighborIndex));
                        //double penalty=weight*this->m_srsPotential->getPotential(neighborIndex,neighborIndex,disp,segmentationPriorDisplacement);
                        double penalty=weight*this->m_srsPotential->getPotential(trueIndex,trueIndex,trueDisplacement+baseDisplacement,segmentationPriorDisplacement);
                        segmentationPenalty+=penalty;
                        //LOG<<targetIndex<<" "<<neighborIndex<<" "<<weight<<" "<<segmentationPriorDisplacement<<" "<<penalty<<endl;
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
#endif
    template<class TImage>
    class UnaryPotentialRegistrationNCCWithBonePrior : public UnaryPotentialRegistrationNCC< TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCCWithBonePrior           Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;

        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
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
        
        double getSegmentationCost(int deformedSegmentationDisplacement, double imageIntensity, int s){
            
            int segmentationProb;
            int tissue=(-500+1000)*255.0/2000;
            if (deformedSegmentationDisplacement>0) {
                segmentationProb = (imageIntensity < tissue) ? 1:0;
            }else{
                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000  &&s>0 ) ? 1 : 0;
            }
            return segmentationProb;

        }
          
        void SetTargetSheetness(ConstImagePointerType img){
            m_targetSheetness=img;
            if (this->m_scale!=1.0){
                m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((img),this->m_scale,true);

            }else{
                m_scaledTargetSheetness=img;
            }
        }
        virtual void Init(){
            logSetStage("InitRegUnary");
            assert(this->m_targetImage);
            assert(this->m_atlasImage);
            assert(this->m_targetSheetness);
            assert(this->m_atlasSegmentation);
            if (this->m_scale!=1.0){
                this->m_scaledTargetImage=FilterUtils<ImageType>::LinearResample((this->m_targetImage),this->m_scale,true);
                this->m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample((this->m_atlasImage),this->m_scale,true);
                this->m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((m_atlasSegmentation),this->m_scale,false);
                this->m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((m_targetSheetness),this->m_scale,true);
            }
            assert(this->radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=max(this->m_radius[d]*this->m_scale,1.0);
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->m_atlasInterpolator=InterpolatorType::New();
            this->m_atlasInterpolator->SetInputImage(this->m_scaledAtlasImage);
            this->m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(this->m_scaledAtlasSegmentation);
            logResetStage;
        }
        virtual double getPotential(IndexType targetIndex, DisplacementType disp){
            double result=0;
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
            }
            //DisplacementType baseDisp=this->m_baseDisplacementMap->GetPixel(targetIndex);
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

                    idx2+=disp+this->m_baseDisplacementMap->GetPixel(neighborIndex)*this->m_scale;

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
    template<class TImage>
    class UnaryPotentialRegistrationNCCWithDistanceBonePrior : public UnaryPotentialRegistrationNCC< TImage>{
    public:
        //itk declarations
        typedef UnaryPotentialRegistrationNCCWithDistanceBonePrior           Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;


        typedef typename TransfUtils<ImageType>::DisplacementType DisplacementType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef itk::NearestNeighborInterpolateImageFunction<ImageType> NNInterpolatorType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;

        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename TransfUtils<ImageType>::DeformationFieldPointerType DisplacementImagePointerType;
        typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        typedef typename itk::Image<float,ImageType::ImageDimension> FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
        typedef itk::LinearInterpolateImageFunction<FloatImageType> FloatImageInterpolatorType;
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
        
        double getSegmentationCost(int deformedSegmentationDisplacement, double imageIntensity, int s){
            
            int segmentationProb;
            int tissue=(-500+1000)*255.0/2000;
            if (deformedSegmentationDisplacement>0) {
                segmentationProb = (imageIntensity < tissue) ? 1:0;
            }else{
                segmentationProb = ( imageIntensity > (300+1000)*255.0/2000  &&s>0 ) ? 1 : 0;
            }
            return segmentationProb;

        }
          
        void SetTargetSheetness(ConstImagePointerType img){
            m_targetSheetness=img;
            if (this->m_scale!=1.0){
                m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((img),this->m_scale,true);

            }else{
                m_scaledTargetSheetness=img;
            }
        }
        virtual void Init(){
            logSetStage("InitRegUnary");
            assert(this->m_targetImage);
            assert(this->m_atlasImage);
            assert(this->m_targetSheetness);
            assert(this->m_atlasSegmentation);
            if (this->m_scale!=1.0){
                this->m_scaledTargetImage=FilterUtils<ImageType>::LinearResample((this->m_targetImage),this->m_scale,true);
                this->m_scaledAtlasImage=FilterUtils<ImageType>::LinearResample((this->m_atlasImage),this->m_scale,true);
                this->m_scaledAtlasSegmentation=FilterUtils<ImageType>::NNResample((m_atlasSegmentation),this->m_scale);
                this->m_scaledTargetSheetness=FilterUtils<ImageType>::LinearResample((m_targetSheetness),this->m_scale,true);
            }
            assert(this->radiusSet);
            for (int d=0;d<ImageType::ImageDimension;++d){
                this->m_scaledRadius[d]=max(this->m_radius[d]*this->m_scale,1.0);
            }
            //nIt=new ImageNeighborhoodIteratorType(this->m_radius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->nIt=ImageNeighborhoodIteratorType(this->m_scaledRadius,this->m_scaledTargetImage, this->m_scaledTargetImage->GetLargestPossibleRegion());
            this->m_atlasInterpolator=InterpolatorType::New();
            this->m_atlasInterpolator->SetInputImage(this->m_scaledAtlasImage);
            this->m_atlasSegmentationInterpolator=NNInterpolatorType::New();
            this->m_atlasSegmentationInterpolator->SetInputImage(this->m_scaledAtlasSegmentation);
            logResetStage;
        }
        virtual double getPotential(IndexType targetIndex, DisplacementType disp){
            double result=0;
            int bone=(300+1000)*255.0/2000;
            int tissue=(-500+1000)*255.0/2000;

            for (short unsigned int d=0; d<ImageType::ImageDimension;++d){
                targetIndex[d]*=this->m_scale;
            }
            //DisplacementType baseDisp=this->m_baseDisplacementMap->GetPixel(targetIndex);
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

                    idx2+=disp+this->m_baseDisplacementMap->GetPixel(neighborIndex)*this->m_scale;

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

 


}//namespace
#endif /* POTENTIALS_H_ */
