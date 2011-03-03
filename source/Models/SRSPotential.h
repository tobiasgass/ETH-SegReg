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
	float *probs;
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
		this->m_baseLabelMap=NULL;
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
		if (m_posteriorWeight>0){

			assert(this->m_movingImage);
			assert(this->m_movingSegmentation);
#if 1
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
			m_segmenter.freeMem();

#endif
			int nIntensities=255;
#if 0
			probs=new float[2*nIntensities*nIntensities];
			m_pairwiseSegmenter.setData(this->m_movingImage,this->m_movingSegmentation);
			m_pairwiseSegmenter.train();
			m_pairwiseSegmenter.eval(probs, nIntensities);
			int idx=0;
			for (int i1=0;i1<nIntensities;++i1){
				for (int i2=0;i2<nIntensities;++i2){
					for (int s=0;s<2;++s,++idx){
						//std::cout<<i1<<" "<<i2<<" "<<s<<" "<<probs[idx]<<" "<<probs[i1*255*2+i2*2+s]<<" "<<probs[i2*255*2+i1*2+s]<<std::endl;
					}
				}
			}
			ofstream myFile ("treeProbs.bin", ios::out | ios::binary);
			myFile.write ((char*)probs,2*nIntensities*nIntensities*sizeof(float) );
			m_pairwiseSegmenter.freeMem();
#else
			probs=new float[2*nIntensities*nIntensities];
			ifstream myFile ("treeProbs.bin", ios::in | ios::binary);
			if (myFile){
				myFile.read((char*)probs,2*nIntensities*nIntensities *sizeof(float));
				std::cout<<" read posterior probs from disk"<<std::endl;
			}else{
				std::cout<<" error reading probs"<<std::endl;
				exit(0);

			}
#endif
			return NULL;
		}
		else return ImageType::New();
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
		if (this->m_haveLabelMap){
			itk::Vector<float,ImageType::ImageDimension> baseDisp=
					LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
			idx2+=baseDisp;
		}


		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);

		double outOfBoundsPenalty=99999999;
		bool ooB=false;
		int oobFactor=1;

		if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
			for (int d=0;d<ImageType::ImageDimension;++d){
				if (idx2[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
					idx2[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
				}
				if (idx2[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
					idx2[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
				}
			}
			ooB=true;
			oobFactor=1;
//			return outOfBoundsPenalty;
//			return m_intensWeight*imageIntensity;
		}
//		std::cout<<idx2<<" "<<this->m_movingInterpolator->GetEndContinuousIndex()<<std::endl;
		assert(this->m_movingInterpolator->IsInsideBuffer(idx2));
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
		//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;

		int deformedSegmentation=m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;

		//		std::cout<<bla<<std::endl;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
		//		double newIdea=1000*-log(0.00001+fabs(m_segmentationProbabilities->GetPixel(fixedIndex)-deformedSegmentation));
		//		std::cout<<fixedIndex<<" "<<label<<" "<<m_segmentationProbabilities->GetPixel(fixedIndex)<<" "<<deformedSegmentation<<" "<<newIdea<<std::endl;

		//-log( p(X,A|T))
		double log_p_XA_T=fabs(imageIntensity-movingIntensity);//*m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),1);
		log_p_XA_T=m_intensWeight*(log_p_XA_T>10000000?10000:log_p_XA_T);
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =m_segmentationWeight*1000* (segmentationLabel!=deformedSegmentation);
		result+=log_p_XA_T+log_p_SA_TSX;
		if (m_posteriorWeight>0){
			//-log(  p(S_x|X,A,S_a,T) )
			//for each index there are nlables/nsegmentation probabilities
			//		long int probposition=fixedIntIndex*m_labelConverter->nLabels()/2;
			//we are then interested in the probability of the displacementlabel only, disregarding the segmentation
			//		probposition+=+m_labelConverter->getIntegerLabel(label)%m_labelConverter->nLabels();
			double log_p_SX_XASAT = 0;//m_posteriorWeight*1000*(-log(m_pairwiseSegmentationProbs(probposition,segmentationLabel)));
			//-log( p(S_a|A) )
			//		int fixedIntIndex=getIntegerImageIndex(fixedIndex);
			double segmentationPenalty=0;//fabs(m_segmentationProbabilities->GetPixel(fixedIndex)[segmentationLabel]) ;
			//		segmentationPenalty/=2;
			double tissueProb=probs[deformedSegmentation+int(imageIntensity/256)*2+int(movingIntensity/256)*2*255];
			double segmentationPenalty2;
			double threshold=0.0;
			if (segmentationLabel){
				segmentationPenalty2=tissueProb>threshold?1.0:tissueProb;
				if (ooB){
					segmentationPenalty2=0;
				}
			}else{
				segmentationPenalty2=(1-tissueProb)>threshold?1.0:(1-tissueProb);
			}

			double p_SA_AXT=m_movingSegmentationProbabilityInterpolator->EvaluateAtContinuousIndex(idx2)[deformedSegmentation];
//			double log_p_SA_AXT=-1000*log();
//			p_SA_AXT=p_SA_AXT>threshold?1.0:p_SA_AXT;
			//			std::cout<<(int)movingIntensity/255<<" "<<(int)imageIntensity/255<<" "<<deformedSegmentation<<" "<<tissueProb<<" "<<segmentationPenalty<<" "<<segmentationPenalty2<<std::endl;
			double segmentationPosterior=m_posteriorWeight*1000*-log(p_SA_AXT*segmentationPenalty2+0.0000001);
			double log_p_SX_X = 0;//m_posteriorWeight*segmentationLabel*500*(-log(segmentationPenalty+0.000000001));
			double log_p_SA_AT = 0;//m_posteriorWeight*1000*(-log(m_movingSegmentationProbabilityInterpolator->EvaluateAtContinuousIndex(idx2)[deformedSegmentation]+0.000000001));
//			std::cout<<deformedSegmentation<<" "<<movingIntensity<<" "<<p_SA_AXT<<" "<<tissueProb<<" "<<log_p_XA_T<<" "<<log_p_SA_TSX<<std::endl;

			//			if (segmentationLabel){
			//				log_p_SX_X/=10;
			//			}

			//		std::cout<<"UNARIES: "<<imageIntensity<<" "<<movingIntensity<<" "<<segmentationLabel<<" "<<deformedSegmentation<<" "<<log_p_XA_T<<" "<<log_p_SA_TSX<<" "<<log_p_SX_XASAT<<" "<<log_p_SX_X<<std::endl;
			result+=+log_p_SX_XASAT+log_p_SX_X+segmentationPosterior+log_p_SA_AT;//+newIdea;
		}
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		return oobFactor*result;//*m_segmentationProbabilities->GetPixel(fixedIndex)[1];
	}
};

template<class TLabelMapper,class TImage,class TSegmentationInterpolator, class TImageInterpolator>
class Class4SegmentationRegistrationUnaryPotential : public SegmentationRegistrationUnaryPotential<TLabelMapper,TImage, TSegmentationInterpolator,TImageInterpolator>{
public:
	//itk declarations
	typedef Class4SegmentationRegistrationUnaryPotential            Self;
	typedef SegmentationRegistrationUnaryPotential<TLabelMapper,TImage,TSegmentationInterpolator,TImageInterpolator>                    Superclass;
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
	typedef truePairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef segmentationClassifier<ImageType> segmentationClassifierType;

	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::LinearInterpolateImageFunction<ProbImageType> FloatImageInterpolatorType;
	typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;

protected:
	pairwiseSegmentationClassifierType m_pairwiseSegmenter;
	float *probs;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(Class4SegmentationRegistrationUnaryPotential, Object);

	Class4SegmentationRegistrationUnaryPotential(){
		m_pairwiseSegmenter=pairwiseSegmentationClassifierType();
	}

	ImagePointerType trainClassifiers(){
		if (this->m_posteriorWeight>0){

			assert(this->m_movingImage);
			assert(this->m_movingSegmentation);
			int nIntensities=255;
			int nProbs=4*nIntensities*nIntensities;
#if 1
			probs=new float[nProbs];
			m_pairwiseSegmenter.setData(this->m_movingImage,this->m_movingSegmentation);
			m_pairwiseSegmenter.train();
			m_pairwiseSegmenter.eval(probs, nIntensities);

			ofstream myFile ("treeProbs4Class.bin", ios::out | ios::binary);
			myFile.write ((char*)probs,nProbs*sizeof(float) );
			m_pairwiseSegmenter.freeMem();
#else
			probs=new float[nProbs];
			ifstream myFile ("treeProbs4Class.bin", ios::in | ios::binary);
			if (myFile){
				myFile.read((char*)probs,2*nIntensities*nIntensities *sizeof(float));
				std::cout<<" read posterior probs from disk"<<std::endl;
			}else{
				std::cout<<" error reading probs"<<std::endl;
				exit(0);

			}
#endif

			return NULL;
		}
		else return ImageType::New();
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
		if (this->m_haveLabelMap){
			itk::Vector<float,ImageType::ImageDimension> baseDisp=
					LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
			idx2+=baseDisp;
		}


		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);

		double outOfBoundsPenalty=99999999;
		bool ooB=false;
		int oobFactor=1;

		if (!this->m_movingInterpolator->IsInsideBuffer(idx2)){
			for (int d=0;d<ImageType::ImageDimension;++d){
				if (idx2[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
					idx2[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
				}
				if (idx2[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
					idx2[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
				}
			}
			ooB=true;
			oobFactor=1.5;
//			return outOfBoundsPenalty;
//			return m_intensWeight*imageIntensity;
		}
//		std::cout<<idx2<<" "<<this->m_movingInterpolator->GetEndContinuousIndex()<<std::endl;
		assert(this->m_movingInterpolator->IsInsideBuffer(idx2));
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
		//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;

		int deformedSegmentation=this->m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;

		//		double bla=5000*(m_segmentationProbabilities->GetPixel(fixedIndex)-m_movingSegmentationProbabilityInterpolator->EvaluateAtContinuousIndex(idx2));
		//		std::cout<<bla<<std::endl;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
		//		double newIdea=1000*-log(0.00001+fabs(m_segmentationProbabilities->GetPixel(fixedIndex)-deformedSegmentation));
		//		std::cout<<fixedIndex<<" "<<label<<" "<<m_segmentationProbabilities->GetPixel(fixedIndex)<<" "<<deformedSegmentation<<" "<<newIdea<<std::endl;

		//-log( p(X,A|T))
		double log_p_XA_T=fabs(imageIntensity-movingIntensity);//*m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),1);
		log_p_XA_T=this->m_intensWeight*(log_p_XA_T>10000000?10000:log_p_XA_T);
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =this->m_segmentationWeight*1000* (segmentationLabel!=deformedSegmentation);
		result+=log_p_XA_T+log_p_SA_TSX;
		if (this->m_posteriorWeight>0){
			//-log(  p(S_x|X,A,S_a,T) )
			//for each index there are nlables/nsegmentation probabilities
			//		long int probposition=fixedIntIndex*m_labelConverter->nLabels()/2;
			//we are then interested in the probability of the displacementlabel only, disregarding the segmentation

			double segmentationPosterior=probs[segmentationLabel+deformedSegmentation+4*(int(imageIntensity/256)+int(movingIntensity/256)*255)];
			std::cout<<segmentationLabel<<" "<<deformedSegmentation<<" "<<int(imageIntensity/256)<<" "<<int(movingIntensity/256)<<" "<<segmentationPosterior<<std::endl;
//			if (segmentationPosterior>0.4)
//				segmentationPosterior=1.0;
			//			std::cout<<(int)movingIntensity/255<<" "<<(int)imageIntensity/255<<" "<<deformedSegmentation<<" "<<tissueProb<<" "<<segmentationPenalty<<" "<<segmentationPenalty2<<std::endl;
			double segmentationPenalty=this->m_posteriorWeight*1000*-log(segmentationPosterior+0.0000001);
						//		std::cout<<"UNARIES: "<<imageIntensity<<" "<<movingIntensity<<" "<<segmentationLabel<<" "<<deformedSegmentation<<" "<<log_p_XA_T<<" "<<log_p_SA_TSX<<" "<<log_p_SX_XASAT<<" "<<log_p_SX_X<<std::endl;
			result+=segmentationPosterior;//+newIdea;
		}
		return oobFactor*result;//*m_segmentationProbabilities->GetPixel(fixedIndex)[1];
	}
};


}//namespace
#endif /* POTENTIALS_H_ */
