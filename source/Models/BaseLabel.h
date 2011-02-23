/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_



template<class TImage>
class BaseLabel : public TImage::OffsetType{
public:
	typedef typename TImage::OffsetType OffsetType;
	typedef typename TImage::OffsetType Superclass;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples;

private:
	int m_index,m_segmentation;

public:
	BaseLabel(){}
	BaseLabel(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
	}
	BaseLabel(int index){
		m_segmentation=index/nDisplacements;
		index=index%nDisplacements;
		int divisor=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension-1);
		for (int d=0;d<TImage::ImageDimension;++d){
			(*this)[d]=index/divisor-nDisplacementSamples;
			index-=((*this)[d]+nDisplacementSamples)*divisor;
			divisor/=2*nDisplacementSamples+1;
		}
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
		BaseLabel result;
		return result;
	}
	virtual OffsetType getDisplacement(){
		return (OffsetType)(*this);
	}
	int getSegmentation(){
		return m_segmentation;
	}
	void setSegmentation(int s){
		m_segmentation=s;
	}
	BaseLabel operator+(BaseLabel &l){
		BaseLabel tmp=Superclass(*this)+Superclass(l);
		tmp.setSegmentation(m_segmentation+l.getSegmentation());
		return tmp;
	}
	BaseLabel operator-(BaseLabel &l){
		BaseLabel tmp=Superclass(*this)-Superclass(l);
		tmp.setSegmentation(m_segmentation-l.getSegmentation());
		return tmp;
	}
	void operator+=(BaseLabel &l){
		Superclass(*this)+=Superclass(l);
		m_segmentation+=l.getSegmentation();
	}
	void operator-=(BaseLabel &l){
		Superclass(*this)-=Superclass(l);
		m_segmentation-=l.getSegmentation();
	}

	BaseLabel operator*(double s){
		BaseLabel tmp=s*Superclass(*this);
		tmp.setSegmentation(s*m_segmentation);
		return tmp;
	}
	void operator*=(double s){
		Superclass(*this)*=s;
		m_segmentation*=s;

	}
	BaseLabel operator/(double s){
		BaseLabel tmp=s/Superclass(*this);
		tmp.m_Segmentation=s/m_segmentation;
		return tmp;
	}
	void operator/=(double s){
		Superclass(*this)/=s;
		m_segmentation/=s;

	}
};

#endif /* LABEL_H_ */
