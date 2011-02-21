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
	virtual double getWeight(int idx1, int idx2){
		IndexType i1=this->m_Grid->getImagePositionAtIndex(idx1);
		IndexType i2=this->m_Grid->getImagePositionAtIndex(idx2);
		double result=1.0*abs(this->m_fixedImage->GetPixel(i1)-this->m_fixedImage->GetPixel(i2))/32500;

		result=exp(-result);
		//		result=int(65535-result)/4200;
		//		result*=result;
		//		std::cout<<"WEIGHT: "<<this->m_fixedImage->GetPixel(i1)<<" "<<this->m_fixedImage->GetPixel(i2)<<" "<<result<<std::endl;
		return result;

	}
	virtual double getPotential2(LabelType l1, LabelType l2){
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
		return tmp1;

	}
	virtual double getPotential(LabelType l1, LabelType l2){
		int tmp2=l1.getSegmentation()!=l2.getSegmentation();
		return tmp2;
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
	ImagePointerType trainClassifiers(){
		assert(m_movingImage);
		assert(m_movingSegmentation);
		std::cout<<"Training the segmentation classifiers.."<<std::endl;
		m_segmenter.setData(m_movingImage,m_movingSegmentation);
		std::cout<<"set Data..."<<std::endl;
		m_segmenter.train();
		std::cout<<"trained"<<std::endl;
		ImagePointerType probImage=m_segmenter.eval(m_fixedImage,m_fixedSegmentation,&m_segmentationProbs);
		std::cout<<"stored confidences"<<std::endl;
		m_pairwiseSegmenter.setData(m_movingImage,m_movingSegmentation);

		m_pairwiseSegmenter.train();
		std::cout<<"computing&caching test data segmentation posteriors"<<std::endl;
		int nData=m_labelConverter->nLabels();
		for (int d=0;d<ImageType::ImageDimension;++d){
			nData*=m_fixedImage->GetLargestPossibleRegion().GetSize()[d];
		}
		matrix<float> testData(nData,10);
		std::vector<int> testLabels(nData);
		itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator(m_fixedImage,m_fixedImage->GetLargestPossibleRegion());
		int i=0;
		for (ImageIterator.GoToBegin();!ImageIterator.IsAtEnd();++ImageIterator){
			double intensity=ImageIterator.Get()/65535;
			IndexType idx=ImageIterator.GetIndex();
			for (int l=0;l<m_labelConverter->nLabels();++l){
				LabelType label=m_labelConverter->getLabel(l);
				int segmentation=label.getSegmentation();
				//	we only need the dataset once, the classifier then generates posteriors for every segmentation
				if (segmentation){
					double deformedIntensity=m_labelConverter->getMovingIntensity(idx,label)/65535;
					int deformedSegmentation=m_labelConverter->getMovingSegmentation(idx,label)>0;
					// build features
					testData(i,0)=intensity;
					testData(i,1)=deformedIntensity;
					testData(i,2)=deformedIntensity-intensity;
					testData(i,3)=fabs(deformedIntensity-intensity);
					testData(i,4)=(deformedIntensity-intensity)*(deformedIntensity-intensity);
					testData(i,5)=deformedIntensity*deformedIntensity;
					testData(i,6)=intensity*intensity;
					testData(i,7)=intensity*deformedIntensity;
					testData(i,8)=fabs(deformedIntensity*intensity);
					testData(i,9)=deformedSegmentation;
					++i;
					testLabels[i]=m_fixedSegmentation->GetPixel(idx)>0;
				}
			}
		}
		testData.resize(i,10);
		std::cout<<i<< " testcases"<<std::endl;
		testLabels.resize(i);
		m_pairwiseSegmenter.eval(testData,testLabels,&m_pairwiseSegmentationProbs);

		std::cout<<"done"<<std::endl;
		return probImage;
	}

	void buildSegmentationPosteriorMatrix(){



	}
	virtual void freeMemory(){
		m_segmentationProbs=matrix<double>(1,1);
		m_pairwiseSegmentationProbs=matrix<double>(1,1);
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
	//		  =-log( p(S_x|X,A,S_a,T) * p(X,A|T) * p(S_a|A) * p(S_a|T,S_x) )
	double getPotential(IndexType fixedIndex, LabelType label){
		int fixedIntIndex=m_labelConverter->getIntegerImageIndex(fixedIndex);
		double result=0.0;
		IndexType movingIndex=m_labelConverter->getMovingIndex(fixedIndex,label);
		double outOfBoundsPenalty=9999;
		if (this->outOfMovingBounds(movingIndex)){
			return outOfBoundsPenalty;
		}

		double imageIntensity=m_fixedImage->GetPixel(fixedIndex);
		double movingIntensity=m_labelConverter->getMovingIntensity(fixedIndex,label);
		int segmentationLabel=label.getSegmentation();
		int deformedSegmentation=m_movingSegmentation->GetPixel(movingIndex)>0;
		//-log( p(X,A|T))
		double log_p_XA_T=1.0*fabs(imageIntensity-movingIntensity);
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =4000* (segmentationLabel!=deformedSegmentation);
		//-log(  p(S_x|X,A,S_a,T) )
		//for each index there are nlables/nsegmentation probabilities
		int probposition=fixedIntIndex*m_labelConverter->nLabels()/2;
		//we are then interested in the probability of the displacementlabel only, disregarding the segmentation
		probposition+=+m_labelConverter->getIntegerLabel(label)%m_labelConverter->nLabels();
		double log_p_SX_XASAT = 1000*-log(m_pairwiseSegmentationProbs(probposition,segmentationLabel));
		//-log( p(S_a|A) )
		double log_p_SA_A = -log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));
		//		std::cout<<"UNARIES: "<<log_p_XA_T<<" "<<log_p_SA_TSX<<" "<<log_p_SX_XASAT<<" "<<log_p_SA_A<<std::endl;
		result+=log_p_XA_T+log_p_SA_TSX+log_p_SX_XASAT;
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));

		if (result<0) std::cout<<"ERROR UNARY POTENTIAL SMALLER THAN ZERO!!!"<<std::endl;
		return result;
	}
};//class

}//namespace
#endif /* POTENTIALS_H_ */
