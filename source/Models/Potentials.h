/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _POTENTIALS_H_
#define _POTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>

namespace itk{


template<class TLabelConverter>
class PairwisePotential : public itk::Object{
public:
	typedef PairwisePotential         Self;
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
	//itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(PairwisePotential, Object);

	virtual double getWeight(int idx1, int idx2){
		return 1.0;
	}
	virtual double getPotential(LabelType l1, LabelType l2){
		return 0.0;
	}
};

template<class TLabelConverter>
class EuclideanPairwisePotential : public PairwisePotential<TLabelConverter>{
public:
	typedef EuclideanPairwisePotential         Self;
	typedef PairwisePotential<TLabelConverter>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;
	typedef	typename Superclass::ImageType ImageType;
	typedef typename Superclass::ImagePointerType ImagePointerType;
	typedef typename Superclass::LabelType LabelType;
	typedef typename Superclass::IndexType IndexType;
	static const int m_dim=ImageType::ImageDimension;

public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(EuclideanPairwisePotential, Object);
	virtual double getPotential(LabelType l1, LabelType l2){
		float tmp1=0.0,tmp;
		for (int d=0;d<m_dim;++d){
			tmp=l1[d]-l2[d];
			tmp1+=tmp*tmp;
		}
		int thresh=8;
		int replacement=999;
		if (tmp1>thresh) tmp1=replacement;
		//		std::cout<<l1<<" "<<l2<<" "<<tmp
		return tmp1;
	}
};

template<class TLabelConverter>
class UnaryPotential : public itk::Object{
public:
	//itk declarations
	typedef UnaryPotential            Self;
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
	itkTypeMacro(UnaryPotential, Object);

	UnaryPotential(){
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
		//		std::cout<<label<<" "<<fixedIndex<< std::endl;
		IndexType movingIndex=m_labelConverter->getMovingIndex(fixedIndex,label);

		double tmp=0;
		bool outOfBounds=false;
		int D=ImageType::ImageDimension;
		SizeType movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
		for (int d=0;d<D;++d){
			if (movingIndex[d]<0 or movingIndex[d]>=movingSize[d])
				return 999999;
		}
		//		std::cout<<movingIndex<<" "<<fixedIndex<< std::endl;
		tmp=fabs(m_fixedImage->GetPixel(fixedIndex)-m_movingImage->GetPixel(movingIndex));
		//	int threshold=100;
		//tmp=(threshold<tmp?threshold:tmp);
#if 0
		double ctThresh=60000;
		if (m_fixedImage->GetPixel(fixedIndex) >ctThresh && m_movingImage->GetPixel(movingIndex)>ctThresh)
			tmp=0;
		else
			tmp=1;
#endif
		return tmp;
		//		return int(256*(1-exp(-tmp)));
	}
};//class

}//namespace
#endif /* POTENTIALS_H_ */
