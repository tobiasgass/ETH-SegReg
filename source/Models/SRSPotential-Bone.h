/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SRSPOTENTIALBONE_H_
#define _SRSPOTENTIALBONE_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include "SRSPotential.h"
#include <utility>
#include "Classifier.h"
#include <boost/numeric/ublas/matrix.hpp>
#include "itkImageConstIteratorWithIndex.h"
#include "itkLinearInterpolateImageFunction.h"
#include <itkNearestNeighborInterpolateImageFunction.h>

#include <iostream>
#include "ImageUtils.h"
namespace itk{


template<class TLabelMapper,class TImage>
class BoneSegmentationRegistrationUnaryPotential : public SegmentationRegistrationUnaryPotential<TLabelMapper,TImage>{
public:
	//itk declarations
	typedef BoneSegmentationRegistrationUnaryPotential            Self;
	typedef SegmentationRegistrationUnaryPotential<TLabelMapper,TImage>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;
	typedef typename ImageType::SpacingType SpacingType;
	typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
	typedef typename ImageInterpolatorType::Pointer InterpolatorPointerType;
	typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
	typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
	typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
	typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
	//	typedef pairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef truePairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef segmentationClassifier<ImageType> segmentationClassifierType;

	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::LinearInterpolateImageFunction<ProbImageType> FloatImageInterpolatorType;
	typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;
	typedef typename itk::ConstNeighborhoodIterator<ImageType>::RadiusType RadiusType;

public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(BoneSegmentationRegistrationUnaryPotential, Object);


	virtual double getLocalPotential(IndexType fixedIndex, LabelType label, double &segCosts, double & regCosts ){
		double result=0;
		//get index in moving image/segmentation
		ContinuousIndexType idx2=getMovingIndex(fixedIndex);
		//current discrete discplacement
		itk::Vector<float,ImageType::ImageDimension> disp=
				LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		idx2+=disp;
		//if in a multiresolution scheme, also add displacement from former iterations
		itk::Vector<float,ImageType::ImageDimension> baseDisp=
				LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
		idx2+=baseDisp;

		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
		bool ooB=false;
		int oobFactor=1;
		//check outofbounds and clip deformation
		if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
			for (int d=0;d<ImageType::ImageDimension;++d){
				if (idx2[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
					idx2[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
				}
				if (idx2[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
					idx2[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
				}
			}
			ooB=true;
			oobFactor=1.5;
		}
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
		double log_p_XA_T;
		//		if (imageIntensity<10000 ){
		//			log_p_XA_T=0;
		//		}
		//		else{
		log_p_XA_T=fabs(imageIntensity-movingIntensity)/65535;
        if (ooB) log_p_XA_T=fabs(imageIntensity)/65535;
		//		}
		//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		//		if (m_fixedSegmentation){
		//			segmentationLabel=LabelMapperType::getSegmentation(this->m_baseLabelMap->GetPixel(fixedIndex));
		//		}
		int deformedSegmentation=this->m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;
		//		segmentationLabel=deformedSegmentation;
		//		std::cout<<bla<<std::endl;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
		//		double newIdea=1000*-log(0.00001+fabs(m_segmentationProbabilities->GetPixel(fixedIndex)-deformedSegmentation));
		//		std::cout<<fixedIndex<<" "<<label<<" "<<m_segmentationProbabilities->GetPixel(fixedIndex)<<" "<<deformedSegmentation<<" "<<newIdea<<std::endl;

		//-log( p(X,A|T))
		log_p_XA_T=this->m_intensWeight*(log_p_XA_T>10000000?10000:log_p_XA_T);
		//		intensSum+=weightlog_p_XA_T;
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =this->m_segmentationWeight* (segmentationLabel!=deformedSegmentation);
		result+=log_p_XA_T+log_p_SA_TSX;
		regCosts=log_p_XA_T;
		segCosts=log_p_SA_TSX;
		if (this->m_posteriorWeight>0){
			int s=this->m_groundTruthImage->GetPixel(fixedIndex);
			double segmentationProb=1;
			switch (segmentationLabel) {
			case 1  :
				segmentationProb = (imageIntensity < -500 ) ? 1 : 0;
				if (!deformedSegmentation) segmentationProb=segmentationProb==1?1:1;
				break;

			case 0:
				segmentationProb = ( imageIntensity > 400) && ( s > 0 ) ? 1 : 0;
				break;

			default:
				assert(false);
			}

			double segmentationPosterior=this->m_posteriorWeight*(segmentationProb);
			result+=segmentationPosterior;
			segCosts+=segmentationPosterior;
			//			segSum+=weight*(segmentationPosterior+log_p_SA_TSX);
			//			segCount+=weight*deformedSegmentation;
		}
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		return oobFactor*result;//*m_segmentationProbabilities->GetPixel(fixedIndex)[1];
	}
};
//#include "SRSPotential.cxx"
}//namespace
#endif /* POTENTIALS_H_ */
