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
	typedef TInterpolator InterpolatorType;
	typedef typename InterpolatorType::Pointer InterpolatorPointerType;
	typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
protected:
	InterpolatorPointerType m_movingInterpolator;
	float m_displacementFactor;
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
	void SetMovingInterpolator(InterpolatorPointerType movingImage){
		m_movingInterpolator=movingImage;
	}
	void SetDisplacementFactor(const float & f){m_displacementFactor=f;}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		double result=0;
		ContinuousIndexType idx2(fixedIndex);
		itk::Vector<float,2> disp=LabelMapperType::getDisplacement(label);
		idx2+= disp*m_displacementFactor;
		if (m_movingInterpolator->IsInsideBuffer(idx2)){
			result=fabs(this->m_fixedImage->GetPixel(fixedIndex)-m_movingInterpolator->EvaluateAtContinuousIndex(idx2));
		}else{
			result=999999;
		}
		return result;
	}
};//class



}//namespace
#endif /* POTENTIALS_H_ */
