/*
 * RegistrationSegmentationLabel.h
 *
 *  Created on: Dec 13, 2010
 *      Author: gasst
 */

#ifndef REGISTRATIONSEGMENTATIONLABEL_H_
#define REGISTRATIONSEGMENTATIONLABEL_H_
#include "Label.h"

template<class TImage>
class RegistrationSegmentationLabel;


template<class TImage, class TLabel>
class RegistrationSegmentationLabelConverter: public RegistrationLabelConverter<TImage,TLabel> {
public:
	typedef RegistrationLabelConverter<TImage,TLabel> Superclass;
	typedef TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename ImageType::SizeType SizeType;
	typedef TLabel LabelType;
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType OffsetType;
	typedef Grid<ImageType> GridType;
	typedef typename LabelType::DeformationType DeformationType;
	typedef typename itk::Image<  DeformationType , ImageType::ImageDimension > DeformationFieldType;
private:
	int m_nSegmentations,m_nDisplacements;

public:
	/*
	 * Constructor
	 */
	RegistrationSegmentationLabelConverter(ImagePointerType fImg, ImagePointerType mImg,
			int nMaxDisplacementsPerAxis, int nDisplacementSamplesPerAxis, int nSegmentations=2):
				Superclass(fImg,mImg,nMaxDisplacementsPerAxis, nDisplacementSamplesPerAxis)
	{
		m_nDisplacements=this->m_nLabels;
		m_nSegmentations=nSegmentations;
		this->m_nLabels=m_nDisplacements*m_nSegmentations;

	}
	/*
	 * convert offset to index
	 */
	virtual int getIntegerLabel(const LabelType &L){
		int idx=Superclass::getIntegerLabel(L);

		return idx;

	}
};

template<class TImage>
class RegistrationSegmentationLabel : public RegistrationLabel<TImage>{
public:
	typedef typename TImage::OffsetType OffsetType;
	typedef typename itk::Vector< float, TImage::ImageDimension> DeformationType;
private:
	int m_segmentation;
	RegistrationSegmentationLabelConverter<TImage, RegistrationSegmentationLabel> * m_LabelConverter;

public:
	RegistrationSegmentationLabel(){}
	RegistrationSegmentationLabel(OffsetType off, RegistrationSegmentationLabelConverter<TImage, RegistrationSegmentationLabel> * LC, int segmentationLabel=0):OffsetType(off){
		m_LabelConverter=LC;
		this->m_index=m_LabelConverter->getIntegerLabel(*this);
		m_segmentation=segmentationLabel;
	}
	int getSegmentation(){return m_segmentation;}

};

#endif /* REGISTRATIONSEGMENTATIONLABEL_H_ */
