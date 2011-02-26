/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _BASEPOTENTIALS_H_
#define _BASEPOTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>

namespace itk{



template<class TLabelMapper,class TImage>
class BaseUnaryPotential : public itk::Object{
public:
	//itk declarations
	typedef BaseUnaryPotential            Self;
	typedef Object                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;

protected:
	ImagePointerType m_fixedImage, m_movingImage;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(BaseUnaryPotential, Object);

	BaseUnaryPotential(){
	}
	virtual void freeMemory(){
	}
	void SetMovingImage(ImagePointerType movingImage){
		m_movingImage=movingImage;
	}
	void SetFixedImage(ImagePointerType fixedImage){
		m_fixedImage=fixedImage;
	}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		return 1.0;
	}
};//class


template<class TLabelMapper,class TImage,class TInterpolator>
class RegistrationUnaryPotential : public BaseUnaryPotential<TLabelMapper,TImage>{
public:
	//itk declarations
	typedef RegistrationUnaryPotential            Self;
	typedef BaseUnaryPotential<TLabelMapper,TImage>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;
	typedef typename ImageType::SpacingType SpacingType;
	typedef TInterpolator InterpolatorType;
	typedef typename InterpolatorType::Pointer InterpolatorPointerType;
	typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
	typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
protected:
	InterpolatorPointerType m_movingInterpolator;
	SpacingType m_displacementFactor;
	LabelImagePointerType m_baseLabelMap;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(RegistrationUnaryPotential, Object);

	RegistrationUnaryPotential(){
		m_displacementFactor=1.0;
	}
	virtual void freeMemory(){
	}
	void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;}
	LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
	void SetMovingInterpolator(InterpolatorPointerType movingImage){
		m_movingInterpolator=movingImage;
	}
	void SetDisplacementFactor(const SpacingType & f){m_displacementFactor=f;}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		double result=0;
		ContinuousIndexType idx2(fixedIndex);
	//	itk::Vector<float,2> disp=LabelMapperType::getDisplacement(label).elementMult(m_displacementFactor);
		itk::Vector<float,ImageType::ImageDimension> disp=LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		idx2+= disp;
		if (m_baseLabelMap){
			itk::Vector<float,2> baseDisp=LabelMapperType::getDisplacement(m_baseLabelMap->GetPixel(fixedIndex));
			idx2+=baseDisp;
		}
		if (m_movingInterpolator->IsInsideBuffer(idx2)){
			result=fabs(this->m_fixedImage->GetPixel(fixedIndex)-m_movingInterpolator->EvaluateAtContinuousIndex(idx2));
		}else{
			result=999999;
		}
		return result;
	}
};//class
template<class TLabelMapper,class TImage,class TSegmentationInterpolator, class TImageInterpolator>
class SegmentationRegistrationUnaryPotential : public RegistrationUnaryPotential<TLabelMapper,TImage, TImageInterpolator>{
public:
	//itk declarations
	typedef SegmentationRegistrationUnaryPotential            Self;
	typedef RegistrationUnaryPotential<TLabelMapper,TImage,TImageInterpolator>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;
	typedef typename ImageType::SpacingType SpacingType;
	typedef TImageInterpolator ImageInterpolatorType;
	typedef typename ImageInterpolatorType::Pointer InterpolatorPointerType;
	typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
	typedef TSegmentationInterpolator SegmentationInterpolatorType;
	typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
	typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
protected:
	SegmentationInterpolatorPointerType m_segmentationInterpolator;
	double m_intensWeight,m_posteriorWeight,m_segmentationWeight;

public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(SegmentationRegistrationUnaryPotential, Object);

	SegmentationRegistrationUnaryPotential(){
	}
	virtual void freeMemory(){
	}
	void SetSegmentationInterpolator(SegmentationInterpolatorPointerType segmentedImage){
		m_segmentationInterpolator=segmentedImage;
	}
	void SetWeights(double intensWeight, double posteriorWeight, double segmentationWeight)
	{
		m_intensWeight=(intensWeight);
		m_posteriorWeight=(posteriorWeight);
		m_segmentationWeight=(segmentationWeight);
	}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		double result=0;
		//get index in moving image/segmentation
		ContinuousIndexType idx2(fixedIndex);
		//current discrete discplacement label
		itk::Vector<float,ImageType::ImageDimension> disp=LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		//multiply by current factor
		idx2+= disp;//.elementMult(this->m_displacementFactor);
		//if in a multiresolution scheme, also add displacement from former iterations
		if (this->m_baseLabelMap){
			itk::Vector<float,2> baseDisp=LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
			idx2+=baseDisp;
		}
		double outOfBoundsPenalty=999999;
		if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
			return outOfBoundsPenalty;
		}

		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		int deformedSegmentation=m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;
		//-log( p(X,A|T))
		double log_p_XA_T=m_intensWeight*fabs(imageIntensity-movingIntensity);//*m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),1);
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =m_segmentationWeight*1000* (segmentationLabel!=deformedSegmentation);
		//-log(  p(S_x|X,A,S_a,T) )
		//for each index there are nlables/nsegmentation probabilities
		//		long int probposition=fixedIntIndex*m_labelConverter->nLabels()/2;
		//we are then interested in the probability of the displacementlabel only, disregarding the segmentation
		//		probposition+=+m_labelConverter->getIntegerLabel(label)%m_labelConverter->nLabels();
		double log_p_SX_XASAT = 0;//m_posteriorWeight*1000*(-log(m_pairwiseSegmentationProbs(probposition,segmentationLabel)));
		//-log( p(S_a|A) )
		double log_p_SX_X = 0;//m_posteriorWeight*1000*-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));
		//		std::cout<<"UNARIES: "<<log_p_XA_T<<" "<<log_p_SA_TSX<<" "<<log_p_SX_XASAT<<" "<<log_p_SA_A<<std::endl;
		result+=log_p_XA_T+log_p_SA_TSX+log_p_SX_XASAT+log_p_SX_X;
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		return result;
	}
};


}//namespace
#endif /* POTENTIALS_H_ */
