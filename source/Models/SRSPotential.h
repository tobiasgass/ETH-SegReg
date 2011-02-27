/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SRSPOTENTIALS_H_
#define _SRSPOTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include "BasePotential.h"
#include <utility>
#include "Classifier.h"
#include <boost/numeric/ublas/matrix.hpp>
#include "itkImageConstIteratorWithIndex.h"

namespace itk{


template<class TLabelMapper,class TImage,class TSegmentationInterpolator, class TImageInterpolator>
class SegmentationRegistrationUnaryPotential : public RegistrationUnaryPotential<TLabelMapper,TImage, TImageInterpolator>{
public:
	//itk declarations
	typedef SegmentationRegistrationUnaryPotential            Self;
	typedef RegistrationUnaryPotential<TLabelMapper,TImage,TImageInterpolator>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;
	typedef typename ImageType::SpacingType SpacingType;
	typedef TImageInterpolator ImageInterpolatorType;
	typedef typename ImageInterpolatorType::Pointer InterpolatorPointerType;
	typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
	typedef TSegmentationInterpolator SegmentationInterpolatorType;
	typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
	typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
	typedef pairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef segmentationClassifier<ImageType> segmentationClassifierType;

	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::LinearInterpolateImageFunction<ProbImageType> FloatImageInterpolatorType;
	typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

protected:
	SegmentationInterpolatorPointerType m_segmentationInterpolator;
	double m_intensWeight,m_posteriorWeight,m_segmentationWeight;
	segmentationClassifierType m_segmenter;
	pairwiseSegmentationClassifierType m_pairwiseSegmenter;
	matrix<float> m_segmentationProbs,m_pairwiseSegmentationProbs;
	ImagePointerType m_movingSegmentation;
	ProbImagePointerType m_segmentationProbabilities,m_movingSegmentationProbabilities;
	FloatImageInterpolatorPointerType m_movingSegmentationProbabilityInterpolator;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(SegmentationRegistrationUnaryPotential, Object);

	SegmentationRegistrationUnaryPotential(){
		m_segmentationWeight=1.0;
		m_intensWeight=1.0;
		m_posteriorWeight=1.0;
		m_segmenter=segmentationClassifierType();
		m_pairwiseSegmenter=pairwiseSegmentationClassifierType();
	}
	virtual void freeMemory(){
	}
	void SetSegmentationInterpolator(SegmentationInterpolatorPointerType segmentedImage){
		m_segmentationInterpolator=segmentedImage;
	}
	void SetMovingSegmentation(ImagePointerType movSeg){
		m_movingSegmentation=movSeg;
	}
	void SetWeights(double intensWeight, double posteriorWeight, double segmentationWeight)
	{
		m_intensWeight=(intensWeight);
		m_posteriorWeight=(posteriorWeight);
		m_segmentationWeight=(segmentationWeight);
	}
	ImagePointerType trainClassifiers(){
		assert(m_movingImage);
		assert(m_movingSegmentation);
		std::cout<<"Training the segmentation classifiers.."<<std::endl;
		m_segmenter.setData(this->m_movingImage,this->m_movingSegmentation);
		std::cout<<"set Data..."<<std::endl;
		m_segmenter.train();
		std::cout<<"trained"<<std::endl;
		ImagePointerType probImage=m_segmenter.eval(this->m_fixedImage,this->m_movingSegmentation,m_segmentationProbabilities);
		ImagePointerType refProbs=m_segmenter.eval(this->m_movingImage,this->m_movingSegmentation,m_movingSegmentationProbabilities);
		ImageUtils<ImageType>::writeImage("train-classified.png",refProbs);
		m_movingSegmentationProbabilityInterpolator=FloatImageInterpolatorType::New();
		m_movingSegmentationProbabilityInterpolator->SetInputImage(m_movingSegmentationProbabilities);
		std::cout<<"stored confidences"<<m_segmentationProbabilities->GetLargestPossibleRegion().GetSize()<<std::endl;
		return probImage;
	}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		double result=0;
		//get index in moving image/segmentation
		ContinuousIndexType idx2(fixedIndex);
		//current discrete discplacement label
		itk::Vector<float,ImageType::ImageDimension> disp=
				LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		//multiply by current factor
		idx2+= disp;//.elementMult(this->m_displacementFactor);
		//if in a multiresolution scheme, also add displacement from former iterations
		if (this->m_baseLabelMap){
			itk::Vector<float,2> baseDisp=LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
			idx2+=baseDisp;
		}
		double outOfBoundsPenalty=999999;
		if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
			return outOfBoundsPenalty;
		}

		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
		//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		int deformedSegmentation=m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;

//		double bla=5000*(m_segmentationProbabilities->GetPixel(fixedIndex)-m_movingSegmentationProbabilityInterpolator->EvaluateAtContinuousIndex(idx2));
//		std::cout<<bla<<std::endl;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
//		double newIdea=1000*-log(0.00001+fabs(m_segmentationProbabilities->GetPixel(fixedIndex)-deformedSegmentation));
//		std::cout<<fixedIndex<<" "<<label<<" "<<m_segmentationProbabilities->GetPixel(fixedIndex)<<" "<<deformedSegmentation<<" "<<newIdea<<std::endl;

		//-log( p(X,A|T))
		double log_p_XA_T=m_intensWeight*fabs(imageIntensity-movingIntensity);//*m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),1);
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =m_segmentationWeight*1000* (segmentationLabel!=deformedSegmentation);
		//-log(  p(S_x|X,A,S_a,T) )
		//for each index there are nlables/nsegmentation probabilities
		//		long int probposition=fixedIntIndex*m_labelConverter->nLabels()/2;
		//we are then interested in the probability of the displacementlabel only, disregarding the segmentation
		//		probposition+=+m_labelConverter->getIntegerLabel(label)%m_labelConverter->nLabels();
		double log_p_SX_XASAT = 0;//m_posteriorWeight*1000*(-log(m_pairwiseSegmentationProbs(probposition,segmentationLabel)));
		//-log( p(S_a|A) )
		//		int fixedIntIndex=getIntegerImageIndex(fixedIndex);
		double segmentationPenalty=fabs(m_segmentationProbabilities->GetPixel(fixedIndex)[segmentationLabel]) ;
//		segmentationPenalty/=2;
		double log_p_SX_X = m_posteriorWeight*1000*
				(-log(segmentationPenalty+0.000000001));
//		std::cout<<"UNARIES: "<<imageIntensity<<" "<<movingIntensity<<" "<<segmentationLabel<<" "<<deformedSegmentation<<" "<<log_p_XA_T<<" "<<log_p_SA_TSX<<" "<<log_p_SX_XASAT<<" "<<log_p_SX_X<<std::endl;
		result+=log_p_XA_T+log_p_SA_TSX+log_p_SX_XASAT+log_p_SX_X;//+newIdea;
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		return result;//*m_segmentationProbabilities->GetPixel(fixedIndex);
	}
};


}//namespace
#endif /* POTENTIALS_H_ */
