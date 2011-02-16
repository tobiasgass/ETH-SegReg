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
#include "SegmentationPotentials.h"
#include <boost/numeric/ublas/matrix.hpp>
using namespace std;
using namespace boost::numeric::ublas;

#include "Classifier.h"
namespace itk{

template<class TLabelConverter>
class JointEuclideanPairwisePotential : public PairwiseSegmentationPotential<TLabelConverter>{
public:
	typedef JointEuclideanPairwisePotential         Self;
	typedef PairwiseSegmentationPotential<TLabelConverter>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;
	typedef	typename Superclass::ImageType ImageType;
	typedef typename Superclass::ImagePointerType ImagePointerType;
	typedef typename Superclass::LabelType LabelType;
	typedef typename Superclass::IndexType IndexType;
	static const int m_dim=ImageType::ImageDimension;

public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(JointEuclideanPairwisePotential, Object);

	virtual double getPotential(LabelType l1, LabelType l2){
		float tmp1=0.0,tmp;
		for (int d=0;d<m_dim;++d){
			tmp=l1[d]-l2[d];
			tmp1+=tmp*tmp;
		}
		int thresh=8;
		int replacement=9999;
		if (tmp1>thresh) tmp1=replacement;
		//		std::cout<<l1<<" "<<l2<<" "<<tmp1<<std::endl;
		if (tmp1<0) std::cout<<"ERROR PAIRWISE POTENTIAL SMALLER ZERO!"<<std::endl;

		int tmp2=l1.getSegmentation()!=l2.getSegmentation();
		return tmp1+tmp2;
	}
};


template<class TLabelConverter>
class UnarySRSPotentialv2 : public itk::Object{
public:
	//itk declarations
	typedef UnarySRSPotentialv2            Self;
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

	typedef pairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef segmentationClassifier<ImageType> segmentationClassifierType;

private:
	ImagePointerType m_fixedImage, m_movingImage, m_movingSegmentation,m_fixedSegmentation;
	LabelConverterType *m_labelConverter;
	segmentationClassifierType m_segmenter;
	pairwiseSegmentationClassifierType m_pairwiseSegmenter;
	matrix<double> m_segmentationProbs,m_pairwiseSegmentationProbs;

public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(UnarySRSPotential, Object);

	UnarySRSPotentialv2(){
		//construct new classifiers
		m_segmenter=segmentationClassifierType();
		m_pairwiseSegmenter=pairwiseSegmentationClassifierType();
	}
	ImagePointerType trainClassifier(){
		assert(m_movingImage);
		assert(m_movingSegmentation);
		m_segmenter.setData(m_movingImage,m_movingSegmentation);
		m_segmenter.train();
		m_pairwiseSegmenter.setData(m_movingImage,m_movingSegmentation);
		m_pairwiseSegmenter.train();
		m_pairwiseSegmenter.eval(m_fixedImage,m_fixedSegmentation,&m_pairwiseSegmentationProbs);
		return m_segmenter.eval(m_fixedImage,m_fixedSegmentation,&m_segmentationProbs);
	}

	void buildSegmentationPosteriorMatrix(){

	}

	void SetMovingImage(ImagePointerType movingImage){
		m_movingImage=movingImage;
	}
	void SetMovingSegmentationImage(ImagePointerType movingSegmentationImage){
		m_movingSegmentation=movingSegmentationImage;
	}
	void SetFixedSegmentationImage(ImagePointerType movingSegmentationImage){
		m_fixedSegmentation=movingSegmentationImage;
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

	//computes -log( p(X,A|T,Sx,Sa) * p(S_a|T,S_x) )
	//		  =-log( p(S_x|X,A,S_a,T) * p(S_a|A) * p(S_a|T,S_x) )
	double getPotential(IndexType fixedIndex, LabelType label){
		double result=0.0;
		IndexType movingIndex=m_labelConverter->getMovingIndex(fixedIndex,label);
		double imageIntensity=m_fixedImage->GetPixel(fixedIndex);
		double log_p_fm_R=fabs(imageIntensity-m_movingImage->GetPixel(movingIndex));///65535;

		double outOfBoundsPenalty=9999;
		if (this->outOfMovingBounds(movingIndex)){
			return outOfBoundsPenalty;
		}

		int segmentationLabel=label.getSegmentation();
		int deformedSegmentation=m_movingSegmentation->GetPixel(movingIndex)>0;
		int log_S_R=(segmentationLabel!=deformedSegmentation);
//		std::cout<<fixedIndex<<" "<<m_labelConverter->getIntegerImageIndex(fixedIndex)<<" "<<-log(m_confidences(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));
//		std::cout<<" "<<-log(m_segmenter.posterior(imageIntensity,segmentationLabel))<<std::endl;
		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		if (result<0) std::cout<<"ERROR UNARY POTENTIAL SMALLER THAN ZERO!!!"<<std::endl;
		return result;
	}
};//class

}//namespace
#endif /* POTENTIALS_H_ */
