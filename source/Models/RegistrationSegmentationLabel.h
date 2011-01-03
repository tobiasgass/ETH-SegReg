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
	typedef typename itk::Image<LabelType> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	typedef typename Superclass::DisplacementType DisplacementType;
	typedef typename Superclass::DisplacementFieldType DisplacementFieldType;
	typedef typename Superclass::DisplacementFieldPointerType DisplacementFieldPointerType;

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
	virtual int getIntegerLabel(const LabelType &L) {
		int idx=Superclass::getIntegerLabel(L);
		int segLabel=L.getSegmentation();
		idx+=segLabel*m_nDisplacements;
		return idx;
	}
	/*
	 * Convert index into offset
	 */
	virtual LabelType getLabel( int idx) {
		LabelType L;
		int positiveIndex;
		// set segmentation label
		L.setSegmentation(idx/m_nDisplacements);
		// remove segmentation from index
		idx=idx%m_nDisplacements;
		// get displacement
		for (int i=this->m_Dim-1;i>=0;--i){
			int actDim=int(pow(this->m_SamplesPerAxis,(i)));
			//get fraction
			positiveIndex=idx/actDim;
			//substract half sample number in order to get back positive and negative offsets
			L[i]=positiveIndex-this->m_SamplesPerAxis/2;
			//remove dimension from index
			idx-=positiveIndex*actDim;
		}
		return L;
	}
//	virtual FieldElementType getFieldElement(LabelType label){
//		FieldElementType result;
//		for (int d=0;d<TImage::ImageDimension;++d){
//			result[d]=label[d];
//		}
//		return result;
//	}
	//	convert an index/label pair to an index in the moving image
	virtual IndexType getMovingIndex(const IndexType & fixedIndex, const LabelType & label) const{
		IndexType idx;
		for (int i=0;i<this->m_Dim;++i){
			idx[i]=int((1.0*this->movingSize[i])/(this->fixedSize[i])*fixedIndex[i])+label[i]*this->m_resolution[i];
		}
		return idx;
	}
	//	convert an index/labelindex to an index in the moving image
	IndexType getMovingIndex(const IndexType & fixedIndex,  int labelIndex) {
		LabelType label=getLabel(labelIndex);
		IndexType movingIndex=getMovingIndex(fixedIndex,label);
		return movingIndex;
	}

	ImagePointerType getSegmentationField(LabelImagePointerType labelImage){
			ImagePointerType segmentation=ImageType::New();
			segmentation->SetRegions(labelImage->GetLargestPossibleRegion());
			segmentation->Allocate();
			itk::ImageRegionIteratorWithIndex<LabelImageType> labelImageIterator(labelImage, labelImage->GetLargestPossibleRegion());
			for (labelImageIterator.GoToBegin(); !labelImageIterator.IsAtEnd();  ++labelImageIterator) {
				LabelType label=labelImageIterator.Get();
				segmentation->SetPixel(labelImageIterator.GetIndex(),label.getSegmentation()*65535);
			}
			return segmentation;
		}

};

template<class TImage>
class RegistrationSegmentationLabel : public RegistrationLabel<TImage>{
public:
	typedef typename TImage::OffsetType OffsetType;
	typedef typename itk::Vector< float, TImage::ImageDimension> DeformationType;
private:
	short int m_segmentationLabel;
	RegistrationSegmentationLabelConverter<TImage, RegistrationSegmentationLabel> * m_LabelConverter;

public:
	RegistrationSegmentationLabel(){}
	RegistrationSegmentationLabel(OffsetType off, RegistrationSegmentationLabelConverter<TImage, RegistrationSegmentationLabel> * LC, int segmentationLabel=0):OffsetType(off){
		m_LabelConverter=LC;
		this->m_index=m_LabelConverter->getIntegerLabel(*this);
		m_segmentationLabel=segmentationLabel;
	}
	short int getSegmentation() const {return m_segmentationLabel;}
	short int setSegmentation(short int segLabel) {m_segmentationLabel=segLabel;}

};

#endif /* REGISTRATIONSEGMENTATIONLABEL_H_ */
