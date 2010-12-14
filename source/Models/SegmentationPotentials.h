/*
 * SegmentationPotentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SegmentationPotentialS_H_
#define _SegmentationPotentialS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>

namespace itk{


template<class TLabelConverter>
class PairwiseSegmentationPotential : public itk::Object{
public:
	typedef PairwiseSegmentationPotential         Self;
	typedef Object                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;


	typedef TLabelConverter LabelConverterType;
	typedef	typename LabelConverterType::ImageType ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename LabelConverterType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(PairwiseSegmentationPotential, Object);

	virtual double getWeight(int idx1, int idx2){
		return 1.0;
	}
	virtual double getPotential(LabelType l1, LabelType l2){
		return (l1!=l2);
	}
};


template<class TLabelConverter>
class UnarySegmentationPotential : public itk::Object{
public:
	//itk declarations
	typedef UnarySegmentationPotential            Self;
	typedef Object                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	//additional types
	typedef TLabelConverter LabelConverterType;
	typedef	typename LabelConverterType::ImageType ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
//	typedef Grid<ImageType> GridType;
	typedef typename LabelConverterType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;

private:
	ImagePointerType m_fixedImage, m_movingImage;
	LabelConverterType *m_labelConverter;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(UnarySegmentationPotential, Object);

	UnarySegmentationPotential(){
	}

	void SetMovingImage(ImagePointerType movingImage){
		m_movingImage=movingImage;
	}
	void SetFixedImage(ImagePointerType fixedImage){
		m_fixedImage=fixedImage;
	}
	void setLabelConverter(LabelConverterType *LC){
		m_labelConverter=LC;
	}
	LabelConverterType * getLabelConverter(){
		return m_labelConverter;
	}
	double getPotential(IndexType fixedIndex, LabelType label){

		double tmp=0;
		bool outOfBounds=false;
		int D=ImageType::ImageDimension;

		double value=m_fixedImage->GetPixel(fixedIndex);
		double threshold=30000;
		if (label==0){
			if (value>threshold)
				tmp=1;
		}
		else{
			if (value<threshold)
				tmp=1;

		}
		return tmp;
	}
};


}
#endif /* SegmentationPotentialS_H_ */
