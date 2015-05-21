
/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#pragma once

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
    class UnaryRegistrationPotentialBase : public itk::Object{
    public:
        //itk declarations
        typedef UnaryRegistrationPotentialBase<TImage>            Self;
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
		itkTypeMacro(Self, Object);

        UnaryRegistrationPotentialBase(){
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
        ~UnaryRegistrationPotentialBase(){
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
			return 0;
        }
       
    };//class

 
 /** \brief
   * This registration potential computes all local potentials for a specific displacement at once and caches them, allowing for greater efficiency
   */
    template<class TImage, class TLocalSimilarity>
    class UnaryRegistrationPotentialWithCaching: public UnaryRegistrationPotentialBase<TImage> {
    public:
        //itk declarations
		typedef UnaryRegistrationPotentialWithCaching<TImage, TLocalSimilarity>           Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef UnaryRegistrationPotentialBase<TImage> Superclass;

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

		typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
        typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
        typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
        typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
		typedef typename itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
		typedef typename itk::ImageRegionConstIterator<ImageType> ImageConstRegionIteratorType;

        typedef  typename itk::PointSet< PixelType, D >   PointSetType;
        typedef typename  PointSetType::PointsContainer PointsContainerType;
        typedef typename  PointSetType::PointsContainerPointer PointsContainerPointer;
        typedef typename  PointSetType::PointsContainerIterator PointsContainerIterator;
        typedef itk::PointsLocator<PointsContainerType> PointsLocatorType;
        typedef typename PointsLocatorType::Pointer PointsLocatorPointerType;

		typedef TLocalSimilarity LocalSimilarityFunctionType;
		typedef typename LocalSimilarityFunctionType::Pointer LocalSimilarityFunctionPointer;

    protected:

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
		itkTypeMacro(Self, Object);
        
        UnaryRegistrationPotentialWithCaching():Superclass(){
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
       
        void setNormalize(bool b){m_normalize=b;}
        void resetNormalize(){
            m_normalize=false;
            m_normalizationFactor=1.0;
        }

#define PREDEF
        virtual void initCaching(){
#ifdef PREDEF
            m_deformedAtlasImage=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasImage,this->m_baseDisplacementMap);
            if (this->m_scaledAtlasMaskImage.IsNotNull()){
                m_deformedMask=TransfUtils<ImageType>::warpImage(this->m_scaledAtlasMaskImage,this->m_baseDisplacementMap);
            }
#ifdef USE_ROI_MASK
			else{
                ImagePointerType mask=ImageUtils<ImageType>::createEmpty(this->m_scaledAtlasImage);
                mask->FillBuffer(1);
                m_deformedMask=TransfUtils<ImageType>::warpImage(mask,this->m_baseDisplacementMap);
            }
#endif

#endif
        }

		void cachePotentials(DisplacementType displacement){
			LOGV(15) << "Caching registration unary potential for displacement " << displacement << endl;
			PointsLocatorPointerType pointsLocator = PointsLocatorType::New();
			if (m_targetLandmarks.IsNotNull()){
				pointsLocator->SetPoints(m_targetLandmarks);
				pointsLocator->Initialize();
			}
			DisplacementType zeroDisp;
			zeroDisp.Fill(0.0);
			//compute average potential for zero displacement.
			bool computeAverage = (displacement == zeroDisp);
			m_averageFixedPotential = computeAverage ? 0.0 : m_averageFixedPotential;

			FloatImagePointerType pot = FilterUtils<ImageType, FloatImageType>::createEmpty(m_coarseImage);
			pot->FillBuffer(0.0);
			ImagePointerType deformedAtlas, deformedMask;

#ifndef PREDEF
			DisplacementImagePointerType translation=TransfUtils<ImageType>::createEmpty(this->m_baseDisplacementMap);
			translation->FillBuffer( displacement);
			TIME(DisplacementImagePointerType composedDeformation=TransfUtils<ImageType>::composeDeformations(translation,this->m_baseDisplacementMap));


			typedef typename itk::VectorLinearInterpolateImageFunction<DisplacementImageType, double> DisplacementInterpolatorType;
			typedef typename DisplacementInterpolatorType::Pointer DisplacementInterpolatorPointerType;
			DisplacementInterpolatorPointerType displacementFieldInterpolator=DisplacementInterpolatorType::New();
			displacementFieldInterpolator->SetInputImage(composedDeformation);

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

			TIME(deformedAtlas = TransfUtils<ImageType>::translateImage(this->m_deformedAtlasImage, displacement));
			if (deformedMask.IsNotNull()){
				TIME(deformedMask = TransfUtils<ImageType>::translateImage(this->m_deformedMask, displacement, true));
			}
			// ImageUtils<ImageType>::writeImage("mask.nii",deformedMask);
			//  ImageUtils<ImageType>::writeImage("deformed.nii",deformedAtlas);

#endif



			///compute local similarity
			//LocalSimilarityFunctionPointer filter = LocalSimilarityFunctionType::New();
			typename MultiThreadedLocalSimilarityNCC<FloatImageType,ImageType>::Pointer filter = MultiThreadedLocalSimilarityNCC<FloatImageType,ImageType>::New();
			filter->SetCoarseImage(pot);
			filter->SetFirstImage(this->m_scaledTargetImage);
//			filter->SetNthInput(1, const_cast<ImageType*>(this->m_scaledTargetImage.GetPointer()));

			filter->SetSecondImage(deformedAtlas);
			if (deformedMask.IsNotNull()){
				filter->SetMask(deformedMask);
			}
			filter->Update();
			pot = filter->GetOutput();
			int c = 0;
			//compute landmark similarity if landmarks are present and weight is not zero
			if (this->m_alpha > 0.0 && m_atlasLandmarks.IsNotNull() && m_targetLandmarks.IsNotNull()){
				typedef typename itk::VectorLinearInterpolateImageFunction<DisplacementImageType, double> DisplacementInterpolatorType;
				typedef typename DisplacementInterpolatorType::Pointer DisplacementInterpolatorPointerType;
				DisplacementInterpolatorPointerType displacementFieldInterpolator = DisplacementInterpolatorType::New();
				displacementFieldInterpolator->SetInputImage(this->m_baseDisplacementMap);
				FloatImageIteratorType potentialIterator(pot, pot->GetLargestPossibleRegion());

			
				double radius = 2 * m_coarseImage->GetSpacing()[0];

				for (potentialIterator.GoToBegin(); !potentialIterator.IsAtEnd(); ++potentialIterator){

					//get original potential
					double localPotential = (1.0 - this->m_alpha)*potentialIterator.Get();

					IndexType coarseIndex = potentialIterator.GetIndex();
					PointType point;

					//search landmarks in neighborhood of current point
					m_coarseImage->TransformIndexToPhysicalPoint(coarseIndex, point);
					typename PointsLocatorType::NeighborsIdentifierType neighborhood;
					pointsLocator->Search(point, radius, neighborhood);
					LOGV(1) << VAR(point) << " " << neighborhood.size() << endl;

					//compute similarity
					for (int n = 0; n < neighborhood.size(); ++n){
						int ptI = neighborhood[n];
						PointType targetPoint = m_targetLandmarks->GetElement(ptI);
						PointType atlasPoint = m_atlasLandmarks->GetElement(ptI);
						LOGV(10) << VAR(point) << " " << VAR(targetPoint) << endl;
						//compute linear weight based on distance between grid point and target point
						double w = 1.0;
#ifdef LINEARWEIGHT
						for (int d = 0; d < D; ++d){
							double axisWeight = max(0.0, 1.0 - fabs(targetPoint[d] - point[d]) / (2 * m_coarseImage->GetSpacing()[d]));
							w *= axisWeight;
						}
#else
						w = exp(-(targetPoint - point).GetNorm() / radius);
#endif
						//get displacement at targetPoint
						DisplacementType displacement = displacementFieldInterpolator->Evaluate(targetPoint);
						//get error
						DisplacementType newVector;
						for (int i2 = 0; i2 < D; i2++) {
							newVector[i2] = targetPoint[i2] + displacement[i2] - atlasPoint[i2];
						}
						double error = newVector.GetNorm();
						localPotential += (this->m_alpha)*w*5.0*(error);

					}


					potentialIterator.Set(localPotential);
					if (computeAverage)
						m_averageFixedPotential += localPotential;
					++c;
				}
			}
			//LOG<<VAR(c)<<endl;
			if (computeAverage &&c != 0){

				m_averageFixedPotential /= c;
				m_normalizationFactor = 1.0;
				if (m_normalize && (m_averageFixedPotential < std::numeric_limits<float>::epsilon())){
					m_normalizationFactor = m_normalizationFactor*m_oldAveragePotential / m_averageFixedPotential;
				}
				LOGV(3) << VAR(m_normalizationFactor) << endl;
				m_oldAveragePotential = m_averageFixedPotential;
			}
			m_currentCachedPotentials = pot;
			m_currentActiveDisplacement = displacement;



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
			return -1;
        }
		
    };//UnaryRegistrationPotentialWithCaching
  
   

}//namespace
