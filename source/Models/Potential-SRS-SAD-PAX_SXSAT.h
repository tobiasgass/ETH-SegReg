/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef ABX
#define ABX
//#define _SRSPOTENTIALS-PAX_SASXT_H_
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
namespace itk{


template<class TLabelMapper,class TImage>
class SegmentationRegistrationUnaryPotentialPosteriorPAXSASXT : public RegistrationUnaryPotential<TLabelMapper,TImage>{
public:
	//itk declarations
	typedef SegmentationRegistrationUnaryPotentialPosteriorPAXSASXT            Self;
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
	typedef relativeFrequencyPairwiseIntensityLikelihood<ImageType> pairwiseSegmentationClassifierType;
	typedef segmentationClassifier<ImageType> segmentationClassifierType;
	//	typedef intensityLikelihoodClassifier<ImageType> segmentationClassifierType;
	typedef pairwiseIntensityLikelihood<ImageType> pairWiseIntensityLikelihoodType;
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
	pairWiseIntensityLikelihoodType m_pairwiseLikelihood;
	matrix<float> m_segmentationProbs,m_pairwiseSegmentationProbs;
	ImagePointerType m_movingSegmentation;
	ProbImagePointerType m_segmentationProbabilities,m_movingSegmentationProbabilities;
	FloatImageInterpolatorPointerType m_movingSegmentationProbabilityInterpolator;
	float *m_segmentationPosteriorProbs,*m_segmentationLikelihoodProbs,*m_pairwiseIntensityLikelihood;

	bool m_fixedSegmentation;
	RadiusType m_radius;
	int nIntensities;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(SegmentationRegistrationUnaryPotentialPosteriorPAXSASXT, Object);

	SegmentationRegistrationUnaryPotentialPosteriorPAXSASXT(){
		m_segmentationWeight=1.0;
		m_intensWeight=1.0;
		m_posteriorWeight=1.0;
		m_segmenter=segmentationClassifierType();
		m_pairwiseSegmenter=pairwiseSegmentationClassifierType();
		m_pairwiseLikelihood=pairWiseIntensityLikelihoodType();
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
	void setRadius(RadiusType rad){m_radius=rad;}
	void setRadius(SpacingType rad){
		for (int d=0;d<ImageType::ImageDimension;++d){
			m_radius[d]=rad[d];
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

		return NULL;
	}
	virtual ImagePointerType loadSegmentationProbs(string filename){
		return NULL;
	}
	virtual ImagePointerType trainPairwiseClassifier(string filename){
		m_segmentationPosteriorProbs=new float[2*2*nIntensities*nIntensities];
		m_pairwiseSegmenter.setData(this->m_movingImage,this->m_movingSegmentation);
		m_pairwiseSegmenter.train();
		m_pairwiseSegmenter.eval( nIntensities,m_segmentationPosteriorProbs);
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
	virtual ImagePointerType trainPairwiseLikelihood(string filename){

		return NULL;
	}


	ContinuousIndexType getMovingIndex(IndexType fixedIndex){
		ContinuousIndexType result;

		for (int d=0;d<ImageType::ImageDimension;++d){
			result[d]=1.0*this->m_fixedSize[d]*1.0*fixedIndex[d]/this->m_movingSize[d];
		}
		return result;
	}

	virtual double getPotential(IndexType fixedIndex, LabelType label){


		typename itk::ConstNeighborhoodIterator<ImageType> nIt(m_radius,this->m_fixedImage, this->m_fixedImage->GetLargestPossibleRegion());
		nIt.SetLocation(fixedIndex);
		double res=0.0;
		double count=0;

		for (unsigned int i=0;i<nIt.Size();++i){
			bool inBounds;
			nIt.GetPixel(i,inBounds);
			if (inBounds){
				IndexType neighborIndex=nIt.GetIndex(i);
				//this should be weighted somehow
				double weight=1.0;
				for (int d=0;d<ImageType::ImageDimension;++d){
					weight*=1-(1.0*fabs(neighborIndex[d]-fixedIndex[d]))/m_radius[d];

				}

				res+=weight*getLocalPotential(neighborIndex,label);
				count+=weight;
			}
		}
		if (count>0)
			return res/count;
		else return 999999;
	}
	virtual double getLocalPotential(IndexType fixedIndex, LabelType label){
		double result=0;
		//get index in moving image/segmentation
		ContinuousIndexType idx2=getMovingIndex(fixedIndex);
		//current discrete discplacement
		itk::Vector<float,ImageType::ImageDimension> disp=
				LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		idx2+= disp;
		//if in a multiresolution scheme, also add displacement from former iterations
		itk::Vector<float,ImageType::ImageDimension> baseDisp=
				LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(fixedIndex));
		idx2+=baseDisp;

		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
		bool ooB=false;
		int oobFactor=1;
		//check outofbounds and clip deformation
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
		}
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(idx2);
		double log_p_XA_T;

		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		int deformedSegmentation=m_segmentationInterpolator->EvaluateAtContinuousIndex(idx2)>0;

		double segmentationProb=1;

		int index=deformedSegmentation+2*(segmentationLabel+int(movingIntensity/nIntensities)*2+int(imageIntensity/nIntensities)*2*nIntensities);
		//		int index=segmentationLabel+2*(deformedSegmentation+int(imageIntensity/nIntensities)*2+int(movingIntensity/nIntensities)*2*nIntensities);

		segmentationProb=m_segmentationPosteriorProbs[index];
		//		std::cout<<deformedSegmentation<<" "<<segmentationLabel<<" "<<int(imageIntensity/nIntensities)<<" "<<int(movingIntensity/nIntensities)<<" "<<segmentationProb<<std::endl;

		double log_p_SA_XASXT = 0;//m_posteriorWeight*1000*(-log(m_pairwiseSegmentationProbs(probposition,segmentationLabel)));
		log_p_SA_XASXT=m_posteriorWeight*-log(segmentationProb+0.0000001);
		//			std::cout<<m_posteriorWeight<<" "<<-log(segmentationPenalty2+0.0000001)<<" "<<segmentationPenalty2<<" "<<tissueProb<<std::endl;
		result+=+log_p_SA_XASXT;


		//		result+=m_segmentationWeight*(-log(movingSegLikelihood+0.000000001));
		result+=m_segmentationWeight*(deformedSegmentation!=segmentationLabel);

		//		std::cout<<m_segmentationWeight<<" "<<m_posteriorWeight<<" "<<m_intensWeight<<std::endl;
		return oobFactor*result;//*m_segmentationProbabilities->GetPixel(fixedIndex)[1];
	}
};

}
#endif /* POTENTIALS_H_ */
