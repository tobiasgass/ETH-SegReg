/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SRS_POTENTIALS_H_
#define _SRS_POTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "Potentials.h"

namespace itk{



template<class TLabelConverter>
class UnarySRSPotential : public itk::Object{
public:
	//itk declarations
	typedef UnarySRSPotential            Self;
	typedef Object                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	//additional types
	typedef TLabelConverter LabelConverterType;
	typedef	typename LabelConverterType::ImageType ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	//	typedef Grid<ImageType> GridType;
	typedef typename LabelConverterType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;

private:
	ImagePointerType m_fixedImage, m_movingImage, m_movingSegmentation;
	LabelConverterType *m_labelConverter;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(UnarySRSPotential, Object);

	UnarySRSPotential(){
	}

	void SetMovingImage(ImagePointerType movingImage){
		m_movingImage=movingImage;
	}
	void SetMovingSegmentationImage(ImagePointerType movingSegmentationImage){
		m_movingSegmentation=movingSegmentationImage;
	}
	void SetFixedImage(ImagePointerType fixedImage){
		m_fixedImage=fixedImage;
	}
	void setLabelConverter(LabelConverterType *LC){
		m_labelConverter=LC;
	}
	LabelConverterType * getLabelConverter(){
		return m_labelConverter;
	}
	bool outOfMovingBounds(const IndexType & movingIndex){
			int D=ImageType::ImageDimension;
			SizeType movingSize=m_movingImage->GetLargestPossibleRegion().GetSize();
			for (int d=0;d<D;++d){
				if (movingIndex[d]<0 or movingIndex[d]>=movingSize[d])
					return true;
			}
			return false;
		}

	//computes -log(p(f,m|T,S)*p(S|T))=-log(p(f,m|T)*p(f|S)*p(S|T))
	double getPotential(IndexType fixedIndex, LabelType label){
		IndexType movingIndex=m_labelConverter->getMovingIndex(fixedIndex,label);
		double outOfBoundsPenalty=9999;
		if (this->outOfMovingBounds(movingIndex)){
			return outOfBoundsPenalty;
		}

		// log(p(f,m|T))
		double log_p_fm_R=0;
		double imageIntensity=m_fixedImage->GetPixel(fixedIndex);
		log_p_fm_R=fabs(imageIntensity-m_movingImage->GetPixel(movingIndex));///65535;

		// log(p(f|S))
		double log_p_f_S=0;
		int segmentationLabel=label.getSegmentation();
		double threshold=25000;
		if (segmentationLabel==0){
			if (imageIntensity>threshold)
				log_p_f_S=(imageIntensity-threshold)/30000;

		}
		else{
			if (imageIntensity<threshold)
				log_p_f_S=(threshold-imageIntensity)/30000;
		}

		// log(p(S|T))
		int deformedSegmentation=m_movingSegmentation->GetPixel(movingIndex)>0;
		int log_S_R=(segmentationLabel!=deformedSegmentation);

		double log_p_fm_SR=0.0;
//		if (log_S_R==0)
//			//same label
//			log_p_fm_SR+=(1+log_p_f_S)*log_p_fm_R;
//		else
//			//different label
//			log_p_fm_SR+=(1+log_p_f_S)*4*log_p_fm_R;
		log_p_fm_SR+=(1+log_p_f_S)*log_p_fm_R;
		double result = 30000*log_S_R + log_p_fm_SR;

//		std::cout<<label<<" "<<segmentationLabel<<" "<<imageIntensity<<" "<<log_p_fm_R<<" "<<log_p_f_S<<" "<<log_S_R<<" "<<deformedSegmentation<<" "<<result<<std::endl;
//		double result=log_p_fm_R+1*log_p_f_S+log_S_R;
		if (result<0) std::cout<<"ERROR UNARY POTENTIAL SMALLER THAN ZERO!!!"<<std::endl;
		return result;
	}
};//class

}//namespace
#endif /* POTENTIALS_H_ */
