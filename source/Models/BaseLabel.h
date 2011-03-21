/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_
#include "itkLinearInterpolateImageFunction.h"



template<typename T, unsigned int D>
class DenseSegRegLabel: public itk::Vector<T,D+1>{
public:

	typedef typename itk::Vector<T,D> Offsettype;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	static const int Dimension=D+1;

	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	DenseSegRegLabel(){}
	DenseSegRegLabel(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),D);
		nLabels=nSegmentations*nDisplacements;
		k=D+1;
	}
	static const void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),D);
		nLabels=nSegmentations*nDisplacements;
	}
	static const void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations*nDisplacements;
	}
	static const DenseSegRegLabel scaleDisplacement(const DenseSegRegLabel & label,const itk::Vector<float,D> & scaling){
		DenseSegRegLabel result(label);
		for (int d=0;d<D;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

	static const DenseSegRegLabel getLabel(int index){
		DenseSegRegLabel result;
		int m_segmentation;
		if (nDisplacements){
			m_segmentation=index/nDisplacements;
			index=index%nDisplacements;
		}
		else{
			m_segmentation=index;
		}
		int divisor=pow(double(2*nDisplacementSamples+1),D-1);
		for (int d=0;d<D;++d){
			result[d]=index/divisor-nDisplacementSamples;
			index-=(result[d]+nDisplacementSamples)*divisor;
			divisor/=2*nDisplacementSamples+1;
		}
		result[k-1]=m_segmentation;
		return result;
	}

	static const int getIndex(const DenseSegRegLabel & label){
		int index=0;
		index+=label[k-1]*(nDisplacements>0?nDisplacements:1);
		int factor=1;
		for (int d=D-1;d>=0;--d){
			index+=factor*(label[d]+nDisplacementSamples);
			factor*=2*nDisplacementSamples+1;
		}
		return index;
	}
	static const Offsettype getDisplacement(const DenseSegRegLabel & label){
		Offsettype off;
		for (int d=0;d<D;++d){
			off[d]=label[d];
		}
		return off;
	}
	static const float getSegmentation(const DenseSegRegLabel & label){
		return label[k-1];
	}
	static const void setSegmentation(DenseSegRegLabel & label, int seg){
		label[k-1]=seg;
	}
	static const void setDisplacement(DenseSegRegLabel & label, const Offsettype &off){
		for (int d=0;d<D;++d){
			label[d]=off[d];
		}
	}
};
template<typename T, unsigned int D>
class SparseSegRegLabel : public itk::Vector<T,D+1>{
public:
	//	typedef typename TImage::Offsettype Offsettype;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndextype Offsettype;
	typedef typename itk::Vector<T,D> Offsettype;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;

public:
	SparseSegRegLabel(){}
	SparseSegRegLabel(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*D);
		nLabels=nSegmentations*nDisplacements;
		k=D+1;
	}
	static const  void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations*nDisplacements;
	}
	static const void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*D);
		nLabels=nSegmentations*nDisplacements;
	}
	static const SparseSegRegLabel getLabel(int index){
		SparseSegRegLabel result;
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

	static const int getIndex(const SparseSegRegLabel & label){
		int index=0;
		if (nSegmentations){
			index+=label[k-1]*(nDisplacements>0?nDisplacements:1);
		}
		//		std::cout<<label<<" "<<label[k-1]<<" "<<index<<std::endl;
		//find out direction
		if (nDisplacements){
			itk::Vector<double,D> sums;
			sums.Fill(0);
			for (int d=0;d<D;++d){
				for (int d2=0;d2<D;++d2){
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
	static const Offsettype getDisplacement(const SparseSegRegLabel & label){
		Offsettype off;
		for (int d=0;d<D;++d){
			off[d]=label[d];
		}
		return off;
	}
	static const float getSegmentation(const SparseSegRegLabel & label){
		return label[k-1];
	}
	static const void setSegmentation(SparseSegRegLabel & label, int seg){
		label[k-1]=seg;
	}
	static const void setDisplacement(SparseSegRegLabel & label, const Offsettype &off){
		for (int d=0;d<D;++d){
			label[d]=off[d];
		}
	}
	static const SparseSegRegLabel scaleDisplacement(const SparseSegRegLabel & label,const itk::Vector<float,D> & scaling){
		SparseSegRegLabel result(label);
		for (int d=0;d<D;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

};
template<typename T, unsigned int D> int  DenseSegRegLabel<T,D>::nLabels=-1;
template<typename T, unsigned int D> int  DenseSegRegLabel<T,D>::nDisplacements=-1;
template<typename T, unsigned int D> int  DenseSegRegLabel<T,D>::nSegmentations=-1;
template<typename T, unsigned int D> int  DenseSegRegLabel<T,D>::nDisplacementSamples=-1;
template<typename T, unsigned int D> int  DenseSegRegLabel<T,D>::k=-1;
template<typename T, unsigned int D> int  SparseSegRegLabel<T,D>::nLabels=-1;
template<typename T, unsigned int D> int  SparseSegRegLabel<T,D>::nDisplacements=-1;
template<typename T, unsigned int D> int  SparseSegRegLabel<T,D>::nSegmentations=-1;
template<typename T, unsigned int D> int  SparseSegRegLabel<T,D>::nDisplacementSamples=-1;
template<typename T, unsigned int D> int  SparseSegRegLabel<T,D>::k=-1;
#endif /* LABEL_H_ */
