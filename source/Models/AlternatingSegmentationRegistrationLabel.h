/*
 * AlternatingSegmentationRegistrationLabel.h
 *
 *  Created on: Feb 21, 2011
 *      Author: gasst
 */

#ifndef ALTERNATINGSEGMENTATIONREGISTRATIONLABEL_H_
#define ALTERNATINGSEGMENTATIONREGISTRATIONLABEL_H_

template<class TImage, class TLabel>
class AlternatingRegistrationSegmentationLabelConverter: public RegistrationSegmentationLabelConverter<TImage,TLabel> {
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
	typedef typename Superclass::PixelType PixelType;
	enum Mode {SEGMENTATION,REGISTRATION};
private:
	Mode m_mode;
public:
	AlternatingRegistrationSegmentationLabelConverter(ImagePointerType fImg, ImagePointerType mImg, ImagePointerType mSeg,
			Mode mode, int nMaxDisplacementsPerAxis, int nDisplacementSamplesPerAxis, int nSegmentations=2):
				Superclass(fImg,mImg, mSeg, nMaxDisplacementsPerAxis, nDisplacementSamplesPerAxis, nSegmentations),m_mode(mode)
	{
		if (m_Mode==SEGMENTATION){
			this->m_nLabels=this->m_nSegmentations;
		}
		else if (m_Mode==REGISTRATION){
			this->m_nLabels=this->m_nRegistrations;
		}
	}
	/*
	 * convert offset to index
	 */
	virtual int getIntegerLabel(const LabelType &L) {

	if (m_Mode==SEGMENTATION){
		return int segLabel=L.getSegmentation();
	}

	}
	else if (m_Mode==REGISTRATION){
		return Superclass::Superclass::getIntegerLabel(L);
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



#endif /* ALTERNATINGSEGMENTATIONREGISTRATIONLABEL_H_ */
