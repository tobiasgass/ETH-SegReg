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
private:
	int m_index,m_segmentation;
	static int m_nLabels,m_nDisplacements,m_nSegmentations,m_nDisplacementSamples;

public:
	BaseLabel(){}
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
