/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _NCCSRSPOTENTIAL_H_
#define _NCCSRSPOTENTIAL_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "itkVector.h"
#include "SRSPotential.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
namespace itk{

template<class TLabelMapper,class TImage>
class NCCSRSUnaryPotential : public SegmentationRegistrationUnaryPotential<TLabelMapper,TImage>{
public:
	//itk declarations
	typedef NCCSRSUnaryPotential            Self;
	typedef SegmentationRegistrationUnaryPotential<TLabelMapper,TImage>       Superclass;
	typedef SmartPointer<Self>        Pointer;
	typedef SmartPointer<const Self>  ConstPointer;

	typedef	TImage ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::SizeType SizeType;
	typedef typename ImageType::SpacingType SpacingType;
	typedef NearestNeighborInterpolateImageFunction<ImageType> SegmentationInterpolatorType;
	typedef typename  SegmentationInterpolatorType::Pointer SegmentationInterpolatorPointerType;


	typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
	typedef typename InterpolatorType::Pointer InterpolatorPointerType;
	typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
	typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
public:
	/** Method for creation through the object factory. */
	itkNewMacro(Self);
	/** Standard part of every itk Object. */
	itkTypeMacro(NCCSRSUnaryPotential, Object);

	NCCSRSUnaryPotential(){
	}
	virtual double getPotential(IndexType fixedIndex, LabelType label){
		typename itk::ConstNeighborhoodIterator<ImageType> nIt(this->m_radius,this->m_fixedImage, this->m_fixedImage->GetLargestPossibleRegion());
		nIt.SetLocation(fixedIndex);
		double count=0, count2=0.00000000001;
		double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
		itk::Vector<float,ImageType::ImageDimension> disp=LabelMapperType::getDisplacement(LabelMapperType::scaleDisplacement(label,this->m_displacementFactor));
		double sum=0.0;
		double result=0;
		double labelWeight=0.0;
		for (unsigned int i=0;i<nIt.Size();++i){

			bool inBounds;
			double f=nIt.GetPixel(i,inBounds);
			if (inBounds){

				IndexType neighborIndex=nIt.GetIndex(i);
				ContinuousIndexType movingIndex(neighborIndex);
				double weight=1.0;
				for (int d=0;d<ImageType::ImageDimension;++d){
					weight*=1-(1.0*fabs(neighborIndex[d]-fixedIndex[d]))/(this->m_radius[d]+0.0000000001);
					movingIndex[d]+=disp[d];
				}

				itk::Vector<float,ImageType::ImageDimension> baseDisp=
						LabelMapperType::getDisplacement(this->m_baseLabelMap->GetPixel(neighborIndex));
				movingIndex+=baseDisp;
				double m;
				if (!this->m_movingInterpolator->IsInsideBuffer(movingIndex)){
					continue;
					for (int d=0;d<ImageType::ImageDimension;++d){
						if (movingIndex[d]>=this->m_movingInterpolator->GetEndContinuousIndex()[d]){
							movingIndex[d]=this->m_movingInterpolator->GetEndContinuousIndex()[d]-0.5;
						}
						else if (movingIndex[d]<this->m_movingInterpolator->GetStartContinuousIndex()[d]){
							movingIndex[d]=this->m_movingInterpolator->GetStartContinuousIndex()[d]+0.5;
						}
					}
				}
				m=this->m_movingInterpolator->EvaluateAtContinuousIndex(movingIndex);
				sff+=f*f;
				smm+=m*m;
				sfm+=f*m;
				sf+=f;
				sm+=m;
				count+=1;//weight;
				sum+=weight*getLocalPotential(neighborIndex,movingIndex,label);
				count2+=weight;
//				double localSeg=getLocalSegmentationProbability(neighborIndex);
//				std::cout<<label<<" "<<getLocalPotential(neighborIndex,movingIndex,label)<<std::endl;
//				//				localSeg=localSeg<0.3?0:localSeg;
//				//				labelWeight+=weight*localSeg;
//				labelWeight+=localSeg;
			}

		}
		if (count){
			sff -= ( sf * sf / count );
			smm -= ( sm * sm / count );
			sfm -= ( sf * sm / count );
			if (smm*sff){
				result=1-(1.0*sfm/sqrt(smm*sff))/2;
			}
			else if (sfm>0)result=0;
			else result=1;
//			result*=labelWeight/count;
		}
		//no correlation whatsoever
		else result=0.5;
		result=this->m_intensWeight*result;
		//sum and norm
		result=(result+sum/count2);//(this->m_intensWeight+this->m_segmentationWeight+2*this->m_posteriorWeight);
		//std::cout<<labelWeight/count2<<std::endl;
		return result;
	}
	virtual double getLocalSegmentationProbability(IndexType fixedIndex)
	{
		if (this->m_segmentationWeight){
			double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
			double p_SX_X=this->m_segmentationLikelihoodProbs[int(imageIntensity/255)];
			return p_SX_X;
		}
		else return 1.0;

	}
	virtual double getLocalPotential(IndexType fixedIndex, ContinuousIndexType movingIndex, LabelType label){
		double result=0;
		//get index in moving image/segmentation
		double imageIntensity=this->m_fixedImage->GetPixel(fixedIndex);
		double movingIntensity=this->m_movingInterpolator->EvaluateAtContinuousIndex(movingIndex);
		int segmentationLabel=LabelMapperType::getSegmentation(label)>0;
		if (this->m_fixedSegmentation){
			segmentationLabel=LabelMapperType::getSegmentation(this->m_baseLabelMap->GetPixel(fixedIndex));
		}
		int deformedSegmentation=this->m_segmentationInterpolator->EvaluateAtContinuousIndex(movingIndex)>0;
		//registration based on similarity of label and labelprobability
		//segProbs holds the probability that the fixedPixel is tissue
		//so if the prob. of tissue is high and the deformed pixel is also tissue, then the log should be close to zero.
		//if the prob of tissue os high and def. pixel is bone, then the term in the brackets becomes small and the neg logarithm large
		//-log( p(X,A|T))
		//-log( p(S_a|T,S_x) )
		double log_p_SA_TSX =this->m_segmentationWeight* (segmentationLabel!=deformedSegmentation);
		if (this->m_posteriorWeight>0){
			//-log(  p(S_x|X,A,S_a,T) )
			//for each index there are nlables/nsegmentation probabilities
			//		long int probposition=fixedIntIndex*this->m_labelConverter->nLabels()/2;
			//we are then interested in the probability of the displacementlabel only, disregarding the segmentation
			//-log( p(S_a|A) )
			double segmentationProb=1;
			if (this->m_segmentationWeight) segmentationProb=this->m_segmentationPosteriorProbs[segmentationLabel+2*(deformedSegmentation+int(imageIntensity/255)*2+int(movingIntensity/255)*2*255)];

			double threshold=5.5;
			double p_SA_AXT=this->m_segmentationLikelihoodProbs[int(imageIntensity/255)];
			if (this->m_segmentationWeight)	p_SA_AXT*=this->m_segmentationLikelihoodProbs[int(movingIntensity/255)];

			threshold=0.55;
			//if we weight pairwise segmentation, then we compute p_SA_AXT, if no pairwise weight is present then we assume we only weight segmentation and therefore compute p_SX_X
			if ( (this->m_segmentationWeight && deformedSegmentation )|| (!this->m_segmentationWeight && segmentationLabel)){
				p_SA_AXT=p_SA_AXT>threshold?1.0:p_SA_AXT;
			}else{
				p_SA_AXT=(1-p_SA_AXT)>threshold?1.0:(1-p_SA_AXT);
			}
			double segmentationPosterior=this->m_posteriorWeight*-log(segmentationProb+0.0000001);

			double log_p_SA_AT = this->m_posteriorWeight*(-log(p_SA_AXT+0.000000001));
			result+=segmentationPosterior+log_p_SA_AT;//+newIdea;

		}
		result+=log_p_SA_TSX;
		return result;
	}

};//class

}//namespace
#endif /* POTENTIALS_H_ */
