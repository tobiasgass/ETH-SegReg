/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef SEGLABEL_H_
#define SEGLABEL_H_

#include "Grid.h"

typedef unsigned short SegmentationLabelType;

template<class TImage, class TLabel>
class SegmentationLabelConverter {
public:
	typedef TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename ImageType::SizeType SizeType;
	typedef TLabel LabelType;
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType OffsetType;
	typedef Grid<ImageType> GridType;
	typedef SegmentationLabelType FieldElementType;
	typedef typename itk::Image<  FieldElementType , ImageType::ImageDimension > LabelFieldType;

private:
	ImagePointerType m_fixedImage;
	SizeType fixedSize, ;
	int m_nLabels;
	int m_Dim;
	//	GridType * m_Grid;
	OffsetType m_resolution;

public:
	SegmentationLabelConverter(ImagePointerType fImg, int nLabels){
		m_fixedImage=fImg;
		m_Dim=ImageType::ImageDimension;
		m_nLabels=nLabels;
		fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
	}
	/*
	 * convert offset to index
	 */
	virtual int getIntegerLabel(const LabelType &L){
		return L;

	}

	/*
	 * Convert index into 'label'
	 */
	virtual LabelType getLabel( int idx) {
		return idx;
	}

	virtual FieldElementType getFieldElement(int idx){
		return idx*65535/m_nLabels;
	}

	int nLabels(){return m_nLabels;}
};


#endif /* SEGLABEL_H_ */
