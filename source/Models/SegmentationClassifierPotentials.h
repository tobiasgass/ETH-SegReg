/*
 * SegmentationPotentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SegmentationClassifierPotentialS_H_
#define _SegmentationClassifierPotentialS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "Grid.h"
#include "Classifier.h"
#include "SegmentationPotentials.h"

namespace itk{



template<class TLabelConverter>
class UnarySegmentationClassifierPotential : public UnarySegmentationPotential<TLabelConverter>{
public:
	//itk declarations
	typedef UnarySegmentationClassifierPotential            Self;
	typedef  UnarySegmentationPotential<TLabelConverter>                   Superclass;
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
	ImagePointerType m_movingSegmentationImage;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(UnarySegmentationClassifierPotential, Object);

	UnarySegmentationClassifierPotential(){
	}
	virtual void freeMemory(){}
	void SetMovingSegmentationImage(ImagePointerType movingSegmentationImage){
		m_movingSegmentationImage=movingSegmentationImage;
	}



	double getPotential(IndexType fixedIndex, LabelType label){

		double tmp=0;
		bool outOfBounds=false;
		int D=ImageType::ImageDimension;

		double value=this->m_fixedImage->GetPixel(fixedIndex);
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
