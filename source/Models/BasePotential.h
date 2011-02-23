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



template<class TLabel,class TImage>
class BaseUnaryPotential : public itk::Object{
public:
	//itk declarations
	typedef BaseUnaryPotential            Self;
	typedef Object                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabel LabelType;
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

}//namespace
#endif /* POTENTIALS_H_ */
