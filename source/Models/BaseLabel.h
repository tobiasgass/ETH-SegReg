/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_

#include "itkVector.h"
#include "itkVariableLengthVector.h"
#include "itkNumericTraits.h"

template<class TImage>
class BaseLabel : public itk::VariableLengthVector<short int>{//TImage::OffsetType{
public:
	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename TImage::OffsetType Superclass;
	typedef typename itk::VariableLengthVector<short int> Superclass;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	//	typedef typename Superclass::RealType RealType;
private:
	int m_index,m_segmentation;

public:
	BaseLabel(){}
	BaseLabel(int NSegmentations, int NDisplacementSamples):Superclass(TImage::ImageDimension+1){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
		k=TImage::ImageDimension+1;
	}
	BaseLabel(int index):Superclass(TImage::ImageDimension+1){
		m_segmentation=index/nDisplacements;
		index=index%nDisplacements;
		int divisor=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension-1);
		for (int d=0;d<TImage::ImageDimension;++d){
			(*this)[d]=index/divisor-nDisplacementSamples;
			index-=((*this)[d]+nDisplacementSamples)*divisor;
			divisor/=2*nDisplacementSamples+1;
		}
		(*this)[k-1]=m_segmentation;
	}

	const int getIndex(){
		int index=0;
		index+=m_segmentation*nDisplacements;
		int factor=1;
		for (int d=TImage::ImageDimension-1;d>=0;--d){
			index+=factor*((*this)[d]+nDisplacementSamples);
			factor*=2*nDisplacementSamples+1;
		}
		return index;
	}
	static const BaseLabel getLabel(int index){
		BaseLabel result(index);
		return result;
	}
	virtual OffsetType getDisplacement(){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=(*this)[d];
		}

		return off;
	}
	int getSegmentation(){
		return m_segmentation;
	}
	void setSegmentation(int s){
		m_segmentation=s;
	}
	BaseLabel operator+(BaseLabel &l){
		BaseLabel tmp=(*this)+(l);
		tmp.setSegmentation(m_segmentation+l.getSegmentation());
		return tmp;
	}
	BaseLabel operator-(BaseLabel &l){
		BaseLabel tmp=(*this)-(l);
		tmp.setSegmentation(m_segmentation-l.getSegmentation());
		return tmp;
	}
	void operator+=(BaseLabel &l){
		(*this)+=(l);
		m_segmentation+=l.getSegmentation();
	}
	void operator-=(BaseLabel &l){
		(*this)-=(l);
		m_segmentation-=l.getSegmentation();
	}

	BaseLabel operator*(double s){
		BaseLabel tmp=s*(*this);
		tmp.setSegmentation(s*m_segmentation);
		return tmp;
	}
	void operator*=(double s){
		(*this)*=s;
		m_segmentation*=s;

	}
	BaseLabel operator/(double s){
		BaseLabel tmp=(*this)/s;
		tmp.m_Segmentation=s/m_segmentation;
		return tmp;
	}
	void operator/=(double s){
		(*this)/=s;
		m_segmentation/=s;

	}
};
namespace itk{
template <class ImageType>
class NumericTraits<BaseLabel<ImageType> > {
public:
#if 1
	typedef short int ValueType;
	typedef short int PrintType;
	typedef short int RealType;
	typedef short int AccumulateType;
	typedef short int AbsType;
	typedef short int FloatType;
#else
	typedef VariableLengthVector<short int> ValueType;
	typedef VariableLengthVector<short int> PrintType;
	typedef VariableLengthVector<short int> RealType;
	typedef VariableLengthVector<short int> AccumulateType;
	typedef VariableLengthVector<short int> AbsType;
	typedef VariableLengthVector<short int> FloatType;
#endif
};
}
#endif /* LABEL_H_ */
