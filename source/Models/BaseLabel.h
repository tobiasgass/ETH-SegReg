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
private:
	int m_index,m_Segmentation;
	static int m_nLabels,m_nDisplacements,m_nSegmentations,m_nDisplacementSamples;

public:
	BaseLabel(){}
	BaseLabel();
	virtual OffsetType getDisplacement(){
		return *(*OffsetType)this;
	};
	static int getSegmentation(){
		return m_Segmentation;
	};
};

#endif /* LABEL_H_ */
