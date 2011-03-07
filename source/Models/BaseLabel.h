/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_
#include "itkLinearInterpolateImageFunction.h"



template<class TImage, class TLabel>
class BaseLabelMapper{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	static const int Dimension=TImage::ImageDimension+1;

	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	BaseLabelMapper(){}
	BaseLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
		k=TImage::ImageDimension+1;
	}
	void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
	}
	void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations*nDisplacements;
	}
	static const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

	static const LabelType getLabel(int index){
		LabelType result;
		int m_segmentation;
		if (nDisplacements){
			m_segmentation=index/nDisplacements;
			index=index%nDisplacements;
		}
		else{
			m_segmentation=index;
		}
		int divisor=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension-1);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=index/divisor-nDisplacementSamples;
			index-=(result[d]+nDisplacementSamples)*divisor;
			divisor/=2*nDisplacementSamples+1;
		}
		result[k-1]=m_segmentation;
		return result;
	}

	static const int getIndex(const LabelType & label){
		int index=0;
		index+=label[k-1]*(nDisplacements>0?nDisplacements:1);
		int factor=1;
		for (int d=TImage::ImageDimension-1;d>=0;--d){
			index+=factor*(label[d]+nDisplacementSamples);
			factor*=2*nDisplacementSamples+1;
		}
		return index;
	}
	static const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
	static const float getSegmentation(const LabelType & label){
		return label[k-1];
	}
	static const void setSegmentation(LabelType & label, int seg){
		label[k-1]=seg;
	}
	static const void setDisplacement(LabelType & label, const OffsetType &off){
		for (int d=0;d<TImage::ImageDimension;++d){
			label[d]=off[d];
		}
	}
};
template<class TImage, class TLabel>
class SparseLabelMapper : public BaseLabelMapper<TImage,TLabel>{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType, TImage::ImageDimension> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
//	static const int Dimension=TImage::ImageDimension+1;
	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	SparseLabelMapper(){}
	SparseLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
		k=TImage::ImageDimension+1;
	}
	void setSegmentationLabels(int labels){
			nSegmentations=labels;
			nLabels=nSegmentations*nDisplacements;
		}
	void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
	}
	static const LabelType getLabel(int index){
		LabelType result;
		result.Fill(0);
		int m_segmentation=0;
		if (nDisplacements){
			m_segmentation=index/nDisplacements;
			index=index%nDisplacements;
		}
		else if(nSegmentations){
			m_segmentation=index;
		}
		int divisor=(double(2*nDisplacementSamples+1));
		result[index/divisor]=index%divisor-nDisplacementSamples;
		result[k-1]=m_segmentation;
		return result;
	}

	static const int getIndex(const LabelType & label){
		int index=0;
		if (nSegmentations){
			index+=label[k-1]*(nDisplacements>0?nDisplacements:1);
		}
		//		std::cout<<label<<" "<<label[k-1]<<" "<<index<<std::endl;
		//find out direction
		if (nDisplacements){
			itk::Vector<double,TImage::ImageDimension> sums;
			sums.Fill(0);
			for (int d=0;d<TImage::ImageDimension;++d){
				for (int d2=0;d2<TImage::ImageDimension;++d2){
					if (d2!=d){
						//					std::cout<<d<<" "<<d2<<" "<<sums[d]<<std::endl;
						sums[d]+=abs(label[d2]);//+nDisplacementSamples;
					}
				}
				if (sums[d]==0){
					//found it!
					index+=(d)*(2*nDisplacementSamples+1)+label[d]+nDisplacementSamples;
					break;
				}
			}
		}
		return index;
	}
	static const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
	static const float getSegmentation(const LabelType & label){
		return label[k-1];
	}
	static const void setSegmentation(LabelType & label, int seg){
			label[k-1]=seg;
		}
		static const void setDisplacement(LabelType & label, const OffsetType &off){
			for (int d=0;d<TImage::ImageDimension;++d){
				label[d]=off[d];
			}
		}

};
template<class T,class L> int  BaseLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::k=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::k=-1;
#endif /* LABEL_H_ */
