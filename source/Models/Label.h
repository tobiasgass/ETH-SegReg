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
	typedef typename itk::Image<LabelType> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	typedef typename itk::Vector< float, TImage::ImageDimension> DisplacementType;
	typedef typename itk::Image<  DisplacementType , ImageType::ImageDimension > DisplacementFieldType;
	typedef typename DisplacementFieldType::Pointer DisplacementFieldPointerType;
	//	typedef typename LabelType::FieldElementType FieldElementType;
	//	typedef typename itk::Image<  FieldElementType , ImageType::ImageDimension > LabelFieldType;


protected:
	ImagePointerType m_fixedImage, m_movingImage;
	SizeType movingSize, fixedSize, relativeSize;
	int m_nLabels;
	int m_Dim;
	int m_SamplesPerAxis;
	//	GridType * m_Grid;
	OffsetType m_resolution;

public:
	RegistrationLabelConverter(ImagePointerType fImg, ImagePointerType mImg, int nMaxDisplacementsPerAxis, int nDisplacementSamplesPerAxis){
		m_fixedImage=fImg;
		m_movingImage=mImg;
		m_SamplesPerAxis=nMaxDisplacementsPerAxis;
		//		m_Grid=grid;
		m_Dim=ImageType::ImageDimension;
		for (int d=0;d<m_Dim;++d){
			m_resolution[d]=nMaxDisplacementsPerAxis/nDisplacementSamplesPerAxis;
			assert(m_resolution[d]>0);
		}

		m_nLabels=pow(m_SamplesPerAxis,m_Dim);
		movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
		fixedSize=m_fixedImage->GetLargestPossibleRegion().GetSize();

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
			idx[i]=int((1.0*movingSize[i])/(fixedSize[i])*fixedIndex[i])+label[i]*m_resolution[i];
//			if (idx[i]<0)
//				idx[i]=0;
//			else if (idx[i]>=fixedSize[i])
//				idx[i]=fixedSize[i]-1;
		}
		return idx;
	}

	//	convert an index/labelindex to an index in the moving image
	IndexType getMovingIndex(const IndexType & fixedIndex,  int labelIndex) {
		LabelType label=getLabel(labelIndex);
		IndexType movingIndex=getMovingIndex(fixedIndex,label);
		return movingIndex;
	}


	int nLabels(){return m_nLabels;}
	int labelSampling(){return m_SamplesPerAxis;}

	// convert labels into displacement vector field
	DisplacementFieldPointerType getDisplacementField(LabelImagePointerType labelImage){
		DisplacementFieldPointerType deformation=DisplacementFieldType::New();
		deformation->SetRegions(labelImage->GetLargestPossibleRegion());
		deformation->SetDirection(labelImage->GetDirection());
		deformation->Allocate();
		itk::ImageRegionIteratorWithIndex<LabelImageType> labelImageIterator(labelImage, labelImage->GetLargestPossibleRegion());
		for (labelImageIterator.GoToBegin(); !labelImageIterator.IsAtEnd();  ++labelImageIterator) {
			DisplacementType displacement;
			LabelType label=labelImageIterator.Get();
			for (int d=0;d<m_Dim;++d){
				displacement[d]=label[d];
			}
			deformation->SetPixel(labelImageIterator.GetIndex(),displacement);
		}
		return deformation;
	}

	ImagePointerType transformImage(ImagePointerType img, LabelImagePointerType labelImage){
		ImagePointerType transformedImage(this->m_fixedImage);
		itk::ImageRegionIteratorWithIndex<LabelImageType> labelImageIterator(labelImage, labelImage->GetLargestPossibleRegion());
		for (labelImageIterator.GoToBegin(); !labelImageIterator.IsAtEnd();  ++labelImageIterator) {
			IndexType currentImageIndex=labelImageIterator.GetIndex();
			LabelType label=labelImage->GetPixel(currentImageIndex);
			IndexType movingIndex=getMovingIndex(currentImageIndex,label);
			transformedImage->SetPixel(currentImageIndex,img->GetPixel(movingIndex));

		}
		return transformedImage;
	}

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
