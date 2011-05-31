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
#include <itkNearestNeighborInterpolateImageFunction.h>

#include <iostream>
#include "ImageUtils.h"
using namespace std;
namespace itk{


template<class TLabelMapper,class TImage>
class SegmentationRegistrationUnaryPotential : public RegistrationUnaryPotential<TLabelMapper,TImage>{
public:
	//itk declarations
	typedef SegmentationRegistrationUnaryPotential            Self;
	typedef RegistrationUnaryPotential<TLabelMapper,TImage>                    Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;
	typedef typename ImageType::SpacingType SpacingType;
	typedef LinearInterpolateImageFunction<ImageType> ImageInterpolatorType;
	typedef typename ImageInterpolatorType::Pointer InterpolatorPointerType;
	typedef typename ImageInterpolatorType::ContinuousIndexType ContinuousIndexType;
	typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
	typedef typename SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;
	typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
	//	typedef pairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef truePairwiseSegmentationClassifier<ImageType> pairwiseSegmentationClassifierType;
	typedef segmentationClassifier<ImageType> segmentationClassifierType;

	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::LinearInterpolateImageFunction<ProbImageType> FloatImageInterpolatorType;
	typedef typename FloatImageInterpolatorType::Pointer FloatImageInterpolatorPointerType;
	typedef typename itk::ConstNeighborhoodIterator<ImageType>::RadiusType RadiusType;
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
	RadiusType m_radius;
	int nIntensities;
public:
	ImagePointerType m_groundTruthImage;
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
		nIntensities=256;
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
	void setRadius(RadiusType rad){
		m_radius=rad;
	}
	void setRadius(SpacingType rad){
		for (int d=0;d<ImageType::ImageDimension;++d){
			m_radius[d]=rad[d]-1;
			if (m_radius[d]<0)
				m_radius[d]=0;
		}

	}
	RadiusType getRadius(){
		return m_radius;
	}

	void SetWeights(double intensWeight, double posteriorWeight, double segmentationWeight)
	{
		m_intensWeight=(intensWeight);
		m_posteriorWeight=(posteriorWeight);
		m_segmentationWeight=(segmentationWeight);
	}


	virtual ImagePointerType trainSegmentationClassifier(string filename){
		if (m_posteriorWeight>0){

			assert(this->m_movingImage);
			assert(this->m_movingSegmentation);
			m_segmentationLikelihoodProbs=new float[2*nIntensities];
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
			ofstream myFileL (filename.c_str(), ios::out | ios::binary);
			myFileL.write ((char*)m_segmentationLikelihoodProbs,2*nIntensities*sizeof(float) );
		}
		return NULL;
	}
	virtual ImagePointerType loadSegmentationProbs(string filename){
		m_segmentationLikelihoodProbs=new float[2*nIntensities];
		ifstream myFileL(filename.c_str(), ios::in | ios::binary);
		if (myFileL){
			myFileL.read((char*)m_segmentationLikelihoodProbs,2*nIntensities *sizeof(float));
			std::cout<<" read posterior m_segmentationLikelihoodProbs from disk"<<std::endl;
		}else{
			std::cout<<" error reading m_segmentationLikelihoodProbs"<<std::endl;
			exit(0);

		}
		typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
		typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
		duplicator->SetInputImage(this->m_movingImage);
		duplicator->Update();
		ImagePointerType returnImage=duplicator->GetOutput();
		itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator(returnImage,returnImage->GetLargestPossibleRegion());
//		for (ImageIterator.GoToBegin();!ImageIterator.IsAtEnd();++ImageIterator){
//			//			LabelImageIterator.Set(predictions[i]*65535);
//			double bone=m_segmentationLikelihoodProbs[1+2*(int)ImageIterator.Get()/255];
//			//				int label=this->m_movingSegmentation->GetPixel(ImageIterator.GetIndex())>0;
//			//				if (!label)
//			//					tissue=1-tissue;
//			//				tissue=tissue>0.5?1.0:tissue;
//			//	tissue*=tissue;
//			ImageIterator.Set(bone*65535);
//		}
//		ImageUtils<ImageType>::writeImage("moving-segmentaitonProbs.png",returnImage);
		return NULL;
	}
	virtual ImagePointerType trainPairwiseClassifier(string filename){
		m_segmentationPosteriorProbs=new float[2*2*nIntensities*nIntensities];
		m_pairwiseSegmenter.setData(this->m_movingImage,this->m_movingSegmentation);
		m_pairwiseSegmenter.train();
		std::cout<<"trained."<<std::endl;
		m_pairwiseSegmenter.eval(m_segmentationPosteriorProbs, nIntensities);
		std::cout<<"finished2"<<std::endl;

		ofstream myFile (filename.c_str(), ios::out | ios::binary);
		myFile.write ((char*)m_segmentationPosteriorProbs,2*2*nIntensities*nIntensities*sizeof(float) );
		m_pairwiseSegmenter.freeMem();
		return NULL;
	}
	virtual ImagePointerType loadPairwiseProbs(string filename){
		m_segmentationPosteriorProbs=new float[2*2*nIntensities*nIntensities];
		ifstream myFile (filename.c_str(), ios::in | ios::binary);
		if (myFile){
			myFile.read((char*)m_segmentationPosteriorProbs,2*2*nIntensities*nIntensities *sizeof(float));
			std::cout<<" read posterior m_segmentationPosteriorProbs from disk"<<std::endl;
		}else{
			std::cout<<" error reading m_segmentationPosteriorProbs"<<std::endl;
			exit(0);

		}
		return NULL;
	}
	virtual ImagePointerType trainPairwiseLikelihood(string filenale){return NULL;}
	ContinuousIndexType getMovingIndex(IndexType fixedIndex){
		ContinuousIndexType result;

		for (int d=0;d<ImageType::ImageDimension;++d){
			result[d]=1.0*this->m_fixedSize[d]*1.0*fixedIndex[d]/this->m_movingSize[d];
		}
		return result;
	}

	virtual double getPotential(IndexType fixedIndex, LabelType label){

		bool zero=false;
		for (int d=0;d<ImageType::ImageDimension;++d){
			if (m_radius[d]<0.1){
				zero=true;
				break;
			}
		}
		if (zero){
			double t1,t2;
			return getLocalPotential(fixedIndex,label,t1,t2);
		}

		typename itk::ConstNeighborhoodIterator<ImageType> nIt(m_radius,this->m_fixedImage, this->m_fixedImage->GetLargestPossibleRegion());
		nIt.SetLocation(fixedIndex);
		double res=0.0;
		double count=0;

		int totalCount=0;

		double segCount=0.0;
		double keep=1;
		double segWeightSum=0.0;
		double segmentationCosts=0,registrationCosts=0,segmentationCostSum=0,registrationCostSum=0;;
		//		m_intensWeight*=pow(keep,ImageType::ImageDimension);
		int centralIntens=nIt.GetCenterPixel();
		for (unsigned int i=0;i<nIt.Size();++i){
			bool inBounds;
			int neighborIntensity=nIt.GetPixel(i,inBounds);
			if (inBounds){
				IndexType neighborIndex=nIt.GetIndex(i);
				//this should be weighted somehow
				double weight=1.0;
				double maxD=0.0;
				double dist=0.0;
				for (int d=0;d<ImageType::ImageDimension;++d){
					double tmp=1.0*fabs(neighborIndex[d]-fixedIndex[d]);
					dist+=tmp*tmp;
					maxD+=m_radius[d]*m_radius[d];
				}
				double eucWeight=1-(dist/maxD);

				res+=eucWeight*getLocalPotential(neighborIndex,label,segmentationCosts, registrationCosts);

				if (eucWeight>1-keep || i==nIt.Size()/2){
					double intensitySimilarity=exp(-fabs(centralIntens-neighborIntensity)/1200);
					segmentationCostSum+=eucWeight*intensitySimilarity*segmentationCosts;
					segCount+=eucWeight*intensitySimilarity;
				}
				registrationCostSum+=eucWeight*registrationCosts;

				count+=eucWeight;
				totalCount++;
			}
		}
		//		std::cout<<1.0*segCount/totalCount<<std::endl;

		if (count>0){
//			std::cout<<registrationCostSum<<" "<<count<<" "<<registrationCostSum/count<<endl;
			return registrationCostSum/count+segmentationCostSum/segCount;
			return res/count;
		}
		else return 999999;
	}
	virtual double getLocalPotential(IndexType fixedIndex, LabelType label, double &segCosts, double & regCosts ){
		double result=0;
		//get index in moving image/segmentation
		ContinuousIndexType idx2=getMovingIndex(fixedIndex);
		//current discrete discplacement
		itk::Vector<float,ImageType::ImageDimension> disp=
				LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		idx2+=disp;
		//if in a multiresolution scheme, also add displacement from former iterations
		itk::Vector<float,ImageType::ImageDimension> baseDisp=
				LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
		idx2+=baseDisp;

		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
		bool ooB=false;
		int oobFactor=1;
		//check outofbounds and clip deformation
		bool inside=this->m_movingInterpolator->IsInsideBuffer(idx2);
		if (!inside){
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
		}
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
		double log_p_XA_T;
		//		if (imageIntensity<10000 ){
		//			log_p_XA_T=0;
		//		}
		//		else{
		log_p_XA_T=fabs(imageIntensity-movingIntensity)/65535;
		//		}
		//		std::cout<<fixedIndex<<" "<<label<<" "<<idx2<<" "<<imageIntensity<<" "<<movingIntensity<<std::endl;
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		//		if (m_fixedSegmentation){
		//			segmentationLabel=LabelMapperType::getSegmentation(this->m_baseLabelMap->GetPixel(fixedIndex));
		//		}
		int deformedSegmentation=m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;
		//		segmentationLabel=deformedSegmentation;
		//		std::cout<<bla<<std::endl;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
		//		double newIdea=1000*-log(0.00001+fabs(m_segmentationProbabilities->GetPixel(fixedIndex)-deformedSegmentation));
		//		std::cout<<fixedIndex<<" "<<label<<" "<<m_segmentationProbabilities->GetPixel(fixedIndex)<<" "<<deformedSegmentation<<" "<<newIdea<<std::endl;

		//-log( p(X,A|T))
		log_p_XA_T=m_intensWeight*(log_p_XA_T>10000000?10000:log_p_XA_T);
		//		intensSum+=weightlog_p_XA_T;
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =m_segmentationWeight* (segmentationLabel!=deformedSegmentation);
		result+=log_p_XA_T+log_p_SA_TSX;
		regCosts=log_p_XA_T;
		segCosts=log_p_SA_TSX;
		if (m_posteriorWeight>0){
			double log_p_SX_XASAT = 0;//m_posteriorWeight*1000*(-log(m_pairwiseSegmentationProbs(probposition,segmentationLabel)));
			double segmentationProb=1;
#if 1
			//			segmentationProb=m_segmentationPosteriorProbs[segmentationLabel+2*segmentationLabel+int(imageIntensity/255)*2*2+int(imageIntensity/255)*2*2*255];
			//			segmentationProb=m_segmentationPosteriorProbs[segmentationLabel+2*segmentationLabel+int(movingIntensity/255)*2*2+int(movingIntensity/255)*2*2*255];
			segmentationProb=m_segmentationPosteriorProbs[segmentationLabel+2*deformedSegmentation+int(imageIntensity/255)*2*2+int(movingIntensity/255)*2*2*255];
			//			segmentationProb=m_segmentationPosteriorProbs[deformedSegmentation+2*segmentationLabel+int(movingIntensity/255)*2*2+int(imageIntensity/255)*2*2*255];
			//			double segmentationPosterior=m_posteriorWeight*-log(segmentationProb+0.0000001);
			double segmentationPosterior=m_posteriorWeight*(1-segmentationProb);
#else
			double segmentationPosterior=m_posteriorWeight*(segmentationLabel!=(m_groundTruthImage->GetPixel(fixedIndex)>0));
#endif

			result+=segmentationPosterior;
			segCosts+=segmentationPosterior;
			//			segSum+=weight*(segmentationPosterior+log_p_SA_TSX);
			//			segCount+=weight*deformedSegmentation;
		}
		//result+=log_p_SA_A;
		//		result+=-log(m_segmentationProbs(m_labelConverter->getIntegerImageIndex(fixedIndex),segmentationLabel));//m_segmenter.posterior(imageIntensity,segmentationLabel));
		return oobFactor*result;//*m_segmentationProbabilities->GetPixel(fixedIndex)[1];
	}
};
//#include "SRSPotential.cxx"
}//namespace
#endif /* POTENTIALS_H_ */
