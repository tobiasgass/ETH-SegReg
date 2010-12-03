/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef LABEL_H_
#define LABEL_H_

#include "Grid.h"

template<class TImage>
class RegistrationLabel;

template<class TImage, class TLabel>
class RegistrationLabelConverter {
public:
	typedef TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename ImageType::SizeType SizeType;
	typedef TLabel LabelType;
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType OffsetType;

	typedef Grid<ImageType> GridType;


private:
	ImagePointerType m_fixedImage, m_movingImage;
	SizeType movingSize, fixedSize, relativeSize;
	int m_nLabels;
	int m_Dim;
	int m_SamplesPerAxis;
	//	GridType * m_Grid;
	OffsetType m_resolution;

public:
	RegistrationLabelConverter(ImagePointerType fImg, ImagePointerType mImg, OffsetType resolution,int nMaxDisplacementsPerAxis){
		m_fixedImage=fImg;
		m_movingImage=mImg;
		m_SamplesPerAxis=nMaxDisplacementsPerAxis;
		//		m_Grid=grid;
		m_resolution=resolution;
		m_Dim=ImageType::ImageDimension;
		m_nLabels=pow(m_SamplesPerAxis,m_Dim);

	}
	/*
	 * convert offset to index
	 */
	virtual int getIntegerLabel(const LabelType &L){
		int idx=0;
		for (int i=0;i<m_Dim;++i){
			//			the offset is shifted by half the sample range in order to make it positive, then multiplied with n^i
			idx+=(L[i]+m_SamplesPerAxis/2)*pow(m_SamplesPerAxis,i);
		}
		return idx;

	}
	/*
	 * Convert index into offset
	 */
	virtual LabelType getLabel( int idx) {
		LabelType L;
		int positiveIndex;
		for (int i=m_Dim-1;i>=0;--i){
			int actDim=int(pow(m_SamplesPerAxis,(i)));
			//get fraction
			positiveIndex=idx/actDim;
			//substract half sample number in order to get back positive and negative offsets
			L[i]=positiveIndex-m_SamplesPerAxis/2;
			//remove dimension from index
			idx-=positiveIndex*actDim;
		}
		return L;
	}

	//	convert an index/label pair to an index in the moving image
	virtual IndexType getMovingIndex(const IndexType & fixedIndex, const LabelType & label) const{
		IndexType idx;
		for (int i=0;i<m_Dim;++i){
			idx[i]=int(1.0*(movingSize[i])/(fixedSize[i])*fixedIndex[i])+label[i]*m_resolution[i];
		}
		return idx;
	}
	//	convert an index/labelindex to an index in the moving image
	IndexType getMovingIndex(const IndexType & fixedIndex,  int labelIndex) {
		LabelType label=getLabel(labelIndex);
		return getMovingIndex(fixedIndex,label);
	}
	int nLabels(){return m_nLabels;}
};

template<class TImage>
class RegistrationLabel : public TImage::OffsetType{
public:
	typedef typename TImage::OffsetType OffsetType;


private:
	int m_index;
	RegistrationLabelConverter<TImage, RegistrationLabel> * m_LabelConverter;
public:
	RegistrationLabel(){}
	RegistrationLabel(OffsetType off, RegistrationLabelConverter<TImage, RegistrationLabel> * LC):OffsetType(off){
		m_LabelConverter=LC;
		m_index=m_LabelConverter->getIntegerLabel(*this);
	}

	int getIndex(){return m_index;}
};

#endif /* LABEL_H_ */
