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
#include "itkLinearInterpolateImageFunction.h"
#include <iostream>
#include "ImageUtils.h"
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
	float *m_segmentationPosteriorProbs,*m_segmentationLikelihoodProbs;
	bool m_fixedSegmentation;
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
		m_fixedSegmentation=false;
	}
	void setFixedSegmentation(bool f){m_fixedSegmentation=f;}
	virtual void freeMemory(){
		delete[] m_segmentationPosteriorProbs;
		delete[] m_segmentationLikelihoodProbs;
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
			int nIntensities=256;

#if 0
			m_segmentationLikelihoodProbs=new float[nIntensities];
			std::cout<<"Training the segmentation classifiers.."<<std::endl;
			m_segmenter.setData(this->m_movingImage,this->m_movingSegmentation);
			std::cout<<"set Data..."<<std::endl;
			m_segmenter.train();
			std::cout<<"trained"<<std::endl;
#if 0
			ImagePointerType probImage=m_segmenter.eval(this->m_fixedImage,this->m_movingSegmentation,m_segmentationProbabilities);
			ImagePointerType refProbs=m_segmenter.eval(this->m_movingImage,this->m_movingSegmentation,m_movingSegmentationProbabilities);
			ImageUtils<ImageType>::writeImage("train-classified.png",refProbs);
			m_movingSegmentationProbabilityInterpolator=FloatImageInterpolatorType::New();
			m_movingSegmentationProbabilityInterpolator->SetInputImage(m_movingSegmentationProbabilities);
#endif
			m_segmenter.eval(nIntensities,m_segmentationLikelihoodProbs);

			std::cout<<"stored confidences"<<std::endl;
			m_segmenter.freeMem();
			ofstream myFileL ("treeSegmentationLikelihoodProbsCROP-FEMUR-CT.bin", ios::out | ios::binary);
			myFileL.write ((char*)m_segmentationLikelihoodProbs,nIntensities*sizeof(float) );
#else
			m_segmentationLikelihoodProbs=new float[nIntensities];
			ifstream myFileL("treeSegmentationLikelihoodProbsCROP-FEMUR-CT.bin", ios::in | ios::binary);
			if (myFileL){
				myFileL.read((char*)m_segmentationLikelihoodProbs,nIntensities *sizeof(float));
				std::cout<<" read posterior m_segmentationLikelihoodProbs from disk"<<std::endl;
			}else{
				std::cout<<" error reading m_segmentationLikelihoodProbs"<<std::endl;
				exit(0);

			}
#endif
			typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
			typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
			duplicator->SetInputImage(this->m_movingImage);
			duplicator->Update();
			ImagePointerType returnImage=duplicator->GetOutput();
			itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator(returnImage,returnImage->GetLargestPossibleRegion());
			for (ImageIterator.GoToBegin();!ImageIterator.IsAtEnd();++ImageIterator){
				//			LabelImageIterator.Set(predictions[i]*65535);
				double bone=m_segmentationLikelihoodProbs[(int)ImageIterator.Get()/255];
				int label=this->m_movingSegmentation->GetPixel(ImageIterator.GetIndex())>0;
				//				if (!label)
				//					tissue=1-tissue;
				//				tissue=tissue>0.5?1.0:tissue;
				//	tissue*=tissue;
				ImageIterator.Set(bone*65535);
			}
			ImageUtils<ImageType>::writeImage("moving-segmentaitonProbs.nii",returnImage);
#if 0
			m_segmentationPosteriorProbs=new float[2*nIntensities*nIntensities];
			m_pairwiseSegmenter.setData(this->m_movingImage,this->m_movingSegmentation);
			m_pairwiseSegmenter.train();
			m_pairwiseSegmenter.eval(m_segmentationPosteriorProbs, nIntensities);
			ofstream myFile ("treeProbsCROP-FEMUR-CT.bin", ios::out | ios::binary);
			myFile.write ((char*)m_segmentationPosteriorProbs,2*nIntensities*nIntensities*sizeof(float) );
			m_pairwiseSegmenter.freeMem();
#else
			m_segmentationPosteriorProbs=new float[2*nIntensities*nIntensities];
			ifstream myFile ("treeProbsCROP-FEMUR-CT.bin", ios::in | ios::binary);
			if (myFile){
				myFile.read((char*)m_segmentationPosteriorProbs,2*nIntensities*nIntensities *sizeof(float));
				std::cout<<" read posterior m_segmentationPosteriorProbs from disk"<<std::endl;
			}else{
				std::cout<<" error reading m_segmentationPosteriorProbs"<<std::endl;
				exit(0);

			}
#endif
			return NULL;
		}
		else return ImageType::New();
	}
	ContinuousIndexType getMovingIndex(IndexType fixedIndex){
		ContinuousIndexType result;

		for (int d=0;d<ImageType::ImageDimension;++d){
			result[d]=1.0*this->m_fixedSize[d]*1.0*fixedIndex[d]/this->m_movingSize[d];
		}
		return result;
	}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		double result=0;
		//get index in moving image/segmentation
		ContinuousIndexType idx2=getMovingIndex(fixedIndex);
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
		double log_p_XA_T;
		if (imageIntensity<10000 ){
			log_p_XA_T=0;
		}
		else{
			log_p_XA_T=fabs(imageIntensity-movingIntensity);
		}
		//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		if (m_fixedSegmentation){
			segmentationLabel=LabelMapperType::getSegmentation(this->m_baseLabelMap->GetPixel(fixedIndex));
		}
		int deformedSegmentation=m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;

		//		std::cout<<bla<<std::endl;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
		//		double newIdea=1000*-log(0.00001+fabs(m_segmentationProbabilities->GetPixel(fixedIndex)-deformedSegmentation));
		//		std::cout<<fixedIndex<<" "<<label<<" "<<m_segmentationProbabilities->GetPixel(fixedIndex)<<" "<<deformedSegmentation<<" "<<newIdea<<std::endl;

		//-log( p(X,A|T))
		log_p_XA_T=m_intensWeight*(log_p_XA_T>10000000?10000:log_p_XA_T);
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =m_segmentationWeight*1000* (segmentationLabel!=deformedSegmentation);
		result+=log_p_XA_T+log_p_SA_TSX;
		if (m_posteriorWeight>0){
	#if 0
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
			double tissueProb=1;
			if (m_segmentationWeight) tissueProb=m_segmentationPosteriorProbs[deformedSegmentation+int(imageIntensity/255)*2+int(movingIntensity/255)*2*255];
			double segmentationPenalty2;
			double threshold=5.5;
			double p_SA_AXT=m_segmentationLikelihoodProbs[int(imageIntensity/255)];
			if (m_segmentationWeight)	p_SA_AXT*=m_segmentationLikelihoodProbs[int(movingIntensity/255)];
			if (segmentationLabel){
				segmentationPenalty2=tissueProb>threshold?1.0:tissueProb;
				if (ooB){
					segmentationPenalty2=0;
				}
			}else{
				segmentationPenalty2=(1-tissueProb)>threshold?1.0:(1-tissueProb);
			}
			threshold=0.55;
			//if we weight pairwise segmentation, then we compute p_SA_AXT, if no pairwise weight is present then we assume we only weight segmentation and therefore compute p_SX_X
			if (m_segmentationWeight && deformedSegmentation || !m_segmentationWeight && segmentationLabel){
				p_SA_AXT=p_SA_AXT>threshold?1.0:p_SA_AXT;
			}else{
				p_SA_AXT=(1-p_SA_AXT)>threshold?1.0:(1-p_SA_AXT);
			}
	#if 0
			p_SA_AXT*=p_SA_AXT;
			segmentationPenalty2*=segmentationPenalty2;
	#endif
			//			std::cout<<segmentationLabel<<" "<<imageIntensity<<" "<<p_SA_AXT<<std::endl;
			if (!m_segmentationWeight) segmentationPenalty2=1;
			//			std::cout<<movingIntensity<<" "<<segmentationLabel<<" "<<p_SA_AXT<<" "<<segmentationPenalty2<<std::endl;
			//m_movingSegmentationProbabilityInterpolator->EvaluateAtContinuousIndex(idx2)[deformedSegmentation]*m_segmentationProbabilities->GetPixel(fixedIndex)[deformedSegmentation];
			//			double log_p_SA_AXT=-1000*log();
			//			std::cout<<(int)movingIntensity/255<<" "<<(int)imageIntensity/255<<" "<<deformedSegmentation<<" "<<tissueProb<<" "<<segmentationPenalty<<" "<<segmentationPenalty2<<std::endl;
			double segmentationPosterior=m_posteriorWeight*1000*-log(segmentationPenalty2+0.0000001);

			double log_p_SX_X = 0;//m_posteriorWeight*segmentationLabel*500*(-log(segmentationPenalty+0.000000001));
			double log_p_SA_AT = m_posteriorWeight*10000*(-log(p_SA_AXT+0.000000001));
			//			std::cout<<deformedSegmentation<<" "<<movingIntensity<<" "<<p_SA_AXT<<" "<<tissueProb<<" "<<log_p_XA_T<<" "<<log_p_SA_TSX<<std::endl;

			//			if (segmentationLabel){
			//				log_p_SX_X/=10;
			//			}

			//		std::cout<<"UNARIES: "<<imageIntensity<<" "<<movingIntensity<<" "<<segmentationLabel<<" "<<deformedSegmentation<<" "<<log_p_XA_T<<" "<<log_p_SA_TSX<<" "<<log_p_SX_XASAT<<" "<<log_p_SX_X<<std::endl;
			result+=+log_p_SX_XASAT+log_p_SX_X+segmentationPosterior+log_p_SA_AT;//+newIdea;
	#else

			double p_SX_X;
			if (segmentationLabel){
				p_SX_X=imageIntensity>25000?1:(25000-imageIntensity)/65000;
			}
			else{
				p_SX_X=imageIntensity<20000?1:(imageIntensity-20000)/65000;
			}
			double p_SA_A;
			if (deformedSegmentation){
				p_SA_A=movingIntensity>25000?1:(25000-movingIntensity)/65000;
			}
			else{
				p_SA_A=movingIntensity<20000?1:(movingIntensity-20000)/65000;
			}
			//			std::cout<<imageIntensity<<" "<<segmentationLabel<<" "<<p_SX_X<<" "<<movingIntensity<<" "<<deformedSegmentation<<" "<<p_SA_A<<std::endl;
			result+=m_posteriorWeight*1000*-log(p_SX_X*p_SA_A+0.000001);
	#endif

		}
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		return oobFactor*result;//*m_segmentationProbabilities->GetPixel(fixedIndex)[1];
	}
};
//#include "SRSPotential.cxx"
}//namespace
#endif /* POTENTIALS_H_ */
