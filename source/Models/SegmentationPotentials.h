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
#include "Grid.h"

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
	typedef Grid<ImageType> GridType;
protected:
	ImagePointerType m_fixedImage;
	GridType * m_Grid;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(PairwiseSegmentationPotential, Object);

	void SetFixedImage(ImagePointerType fixedImage){
		m_fixedImage=fixedImage;
	}
	void SetGrid(GridType *grid){
		m_Grid=grid;
	}
	virtual double getPotential2(LabelType l1, LabelType l2){
			return 0;
		}
	virtual double getWeight(int idx1, int idx2){
		IndexType i1=m_Grid->getImagePositionAtIndex(idx1);
		IndexType i2=m_Grid->getImagePositionAtIndex(idx2);
//		double result=abs(m_fixedImage->GetPixel(i1)-m_fixedImage->GetPixel(i2))*1.0;
//
////		result=exp(-result);
//		result=int(65535-result)/4200;
//		result*=result;
//		std::cout<<m_fixedImage->GetPixel(i1)<<" "<<m_fixedImage->GetPixel(i2)<<" "<<result<<std::endl;
		double result=1.0*abs(this->m_fixedImage->GetPixel(i1)-this->m_fixedImage->GetPixel(i2))/32500;

		result=exp(-result);
		return result;
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
	virtual void freeMemory(){}
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
		double threshold=25000;
		if (label==0){
			if (value>threshold)
				tmp=(value-threshold)/30000;

		}
		else{
			if (value<threshold)
				tmp=(threshold-value)/30000;
		}
//		std::cout<<"tmp :"<<tmp<<std::endl;
		return tmp;
	}
};


}
#endif /* SegmentationPotentialS_H_ */
