/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <assert.h>
#include "itkVectorImage.h"
#include "itkConstNeighborhoodIterator.h"
#include <itkVectorResampleImageFilter.h>
#include <itkResampleImageFilter.h>
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"


/*
 * Isotropic Graph
 * Returns current/next position in a grid based on size and resolution
 */

template<class TUnaryFunction,class TLabelMapper, class TImage>
class GraphModel{
public:
	typedef TUnaryFunction UnaryFunctionType;
	typedef typename UnaryFunctionType::Pointer UnaryFunctionPointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType OffsetType;
	typedef typename TImage::PointType PointType;

	typedef typename TImage::SizeType SizeType;
	typedef  TImage ImageType;
	typedef typename TImage::SpacingType SpacingType;
	typedef typename TImage::Pointer ImagePointerType;
	typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;

private:
	ImagePointerType m_fixedImage,m_fixedGradientImage;
	LabelImagePointerType m_labelImage;
	SizeType m_totalSize,m_gridSize,m_imageLevelDivisors;
	SpacingType m_spacing,m_labelSpacing;
	PointType m_origin;
	double m_DisplacementScalingFactor;
	static const unsigned int m_dim=TImage::ImageDimension;
	int m_nNodes,m_nVertices;
	//	ImageInterpolatorType m_ImageInterpolator,m_SegmentationInterpolator,m_BoneConfidenceInterploator;
	UnaryFunctionPointerType m_unaryFunction;
	bool verbose;
	double m_segmentationWeight, m_registrationWeight;
	bool m_haveLabelMap;

public:

	GraphModel(ImagePointerType fixedimage,UnaryFunctionPointerType unaryFunction, SpacingType res, double displacementScalingFactor, double segmentationWeight, double registrationWeight)
	:m_fixedImage(fixedimage),m_unaryFunction(unaryFunction),m_DisplacementScalingFactor(displacementScalingFactor), m_segmentationWeight(segmentationWeight),m_registrationWeight(registrationWeight)
	{
		m_haveLabelMap=false;
		verbose=false;
		assert(m_dim>1);
		assert(m_dim<4);
		m_totalSize=fixedimage->GetLargestPossibleRegion().GetSize();
		//		assert(m_totalSize==movingImage->GetLargestPossibleRegion().GetSize());
		m_spacing=res;
		m_nNodes=1;
		//		m_dblSpacing=m_spacing[0];

		if (LabelMapperType::nDisplacementSamples){
			m_labelSpacing=0.4*m_spacing/(LabelMapperType::nDisplacementSamples);
			if (verbose) std::cout<<m_spacing<<" "<<LabelMapperType::nDisplacementSamples<<" "<<m_labelSpacing<<std::endl;
		}
		for (int d=0;d<(int)m_dim;++d){
			if (verbose) std::cout<<"total size divided by spacing :"<<1.0*m_totalSize[d]/m_spacing[d]<<std::endl;
			m_origin[d]=0;//-int(m_spacing[d]/2);
			m_gridSize[d]=m_totalSize[d]/m_spacing[d]+1;
			m_nNodes*=m_gridSize[d];
			if (d>0){
				m_imageLevelDivisors[d]=m_imageLevelDivisors[d-1]*m_gridSize[d-1];
			}else{
				m_imageLevelDivisors[d]=1;
			}
		}

		if (verbose) std::cout<<"GridSize: "<<m_dim<<" ";
		if (m_dim>=2){
			if (verbose) std::cout<<m_gridSize[0]<<" "<<m_gridSize[1];
			m_nVertices=m_gridSize[1]*(m_gridSize[0]-1)+m_gridSize[0]*(m_gridSize[1]-1);
		}
		if (m_dim==3){
			std::cout<<" "<<m_gridSize[0];
			m_nVertices+=(m_gridSize[2]-1)*m_gridSize[1]*m_gridSize[0];
		}
		if (verbose) std::cout<<" "<<m_nNodes<<" "<<m_nVertices<<" "<<LabelMapperType::nLabels<<std::endl;
		//		m_ImageInterpolator.SetInput(m_movingImage);
	}

	typename ImageType::DirectionType getDirection(){return m_fixedImage->GetDirection();}
	void setLabelImage(LabelImagePointerType limg){m_labelImage=limg;m_haveLabelMap=true;}
	void setGradientImage(ImagePointerType limg){m_fixedGradientImage=limg;}

	SpacingType getDisplacementFactor(){return m_labelSpacing*m_DisplacementScalingFactor;}
	SpacingType getSpacing(){return m_spacing;}
	PointType getOrigin(){return m_origin;}

	double getUnaryPotential(int gridIndex, int labelIndex){
		IndexType fixedIndex=gridToImageIndex(getGridPositionAtIndex(gridIndex));
		LabelType label=LabelMapperType::getLabel(labelIndex);

		return m_unaryFunction->getPotential(fixedIndex,label)/m_nNodes;
	}

	double getPairwisePotential(int LabelIndex,int LabelIndex2){
		LabelType l1=LabelMapperType::getLabel(LabelIndex);
		LabelType l2=LabelMapperType::getLabel(LabelIndex2);
		//	LabelType l=l1-l2;
		double result=0;
		for (unsigned int d=0;d<m_dim;++d){
			double tmp=(l1[d]-l2[d])*m_labelSpacing[d]*m_DisplacementScalingFactor;
			result+=tmp*tmp;
		}
		//		std::cout<<sqrt(result)<<std::endl;
		double trunc=5.0;


		return (sqrt(result)>trunc?trunc:sqrt(result));
	}

	double getPairwisePotential(int idx1,int idx2,int LabelIndex,int LabelIndex2,bool verbose=false){
		IndexType fixedIndex1=gridToImageIndex(getGridPositionAtIndex(idx1));
		IndexType fixedIndex2=gridToImageIndex(getGridPositionAtIndex(idx2));
		LabelType l1=LabelMapperType::getLabel(LabelIndex);
		LabelType l2=LabelMapperType::getLabel(LabelIndex2);
		return getPairwisePotential(fixedIndex1,fixedIndex2,l1,l2);
	}
	double getPairwisePotential(IndexType fixedIndex1,IndexType fixedIndex2,LabelType l1,LabelType l2,bool verbose=false){

		//segmentation smoothness
		double segmentationSmootheness=0;
		if (LabelMapperType::nSegmentations){
			segmentationSmootheness=fabs(LabelMapperType::getSegmentation(l1)-LabelMapperType::getSegmentation(l2));
			double segWeight=fabs(m_fixedImage->GetPixel(fixedIndex1)-m_fixedImage->GetPixel(fixedIndex2));
			segWeight=exp(-segWeight/3000);
			segmentationSmootheness*=segWeight*m_segmentationWeight;
		}

		//registration smoothness
		LabelType oldl1;//=m_labelImage->GetPixel(fixedIndex1);
		LabelType oldl2;//=m_labelImage->GetPixel(fixedIndex2);
		if (m_labelImage){
			oldl1=m_labelImage->GetPixel(fixedIndex1);
			oldl2=m_labelImage->GetPixel(fixedIndex2);
		}
		double registrationSmootheness=0;
		//		double constrainedViolatedPenalty=std::numeric_limits<double>::max()/(m_nNodes*1000);;
		double constrainedViolatedPenalty=100;//9999999999;
		bool constrainsViolated=false;
		//		std::cout<<"DeltaInit1: "<<fixedIndex1<<" "<<fixedIndex2<<std::endl;
		if (LabelMapperType::nDisplacements){
			double d1,d2;
			int delta;
			for (unsigned int d=0;d<m_dim;++d){
				//applying the labels to evaluate to neighboring pixels
				d1=(l1[d])*m_labelSpacing[d]*m_DisplacementScalingFactor;
				d2=(l2[d])*m_labelSpacing[d]*m_DisplacementScalingFactor;
				if (m_labelImage){
					d1+=oldl1[d];
					d2+=oldl2[d];
				}
				//				std::cout<<"DeltaInit2: "<<d1<<" "<<d2<<std::endl;

				delta=(fixedIndex2[d]-fixedIndex1[d]);

				double axisPositionDifference=1.0*(d2-d1)/(m_spacing[d]);
				//				std::cout<<"DeltaInit2: "<<(m_spacing[d])<<" "<<d1<<" "<<d2<<" "<<axisPositionDifference<<std::endl;

				double relativeAxisPositionDifference=1.0*(axisPositionDifference+(delta/m_spacing[d]));
				//				std::cout<<"DeltaInit3 :"<<axisPositionDifference<<" "<<delta<<" "<<m_spacing[d]<<" "<<relativeAxisPositionDifference<<std::endl;
				//we shall never tear the image!
				if (delta>0){
					if (relativeAxisPositionDifference<0){
						constrainsViolated=true;
						//						exit(0);
						//						break;
					}
				}
				else if (delta<0){
					if (relativeAxisPositionDifference>0){
						constrainsViolated=true;
						//						break;
					}
				}
				if (fabs(relativeAxisPositionDifference)>2){
					constrainsViolated=true;
					//					exit(0);
					//					break;
				}

				registrationSmootheness+=(axisPositionDifference)*(axisPositionDifference);
			}
			//			std::cout<<registrationSmootheness<<std::endl;
			registrationSmootheness*=m_registrationWeight;

		}

		if (constrainsViolated){
			if (false){
				std::cout<<l1<<"/"<<l2<<" "
						<<fixedIndex1<<" -> "<<oldl1<<"+"<<LabelMapperType::scaleDisplacement(l1,getDisplacementFactor())<<" vs: "
						<<fixedIndex2<<" -> "<<oldl2<<"+"<<LabelMapperType::scaleDisplacement(l2,getDisplacementFactor())<<std::endl;
			}
			//						return 	m_registrationWeight*constrainedViolatedPenalty;
			return 	constrainedViolatedPenalty;
		}
		//		std::cout<<registrationSmootheness<<std::endl;
		double result=(registrationSmootheness+segmentationSmootheness)/m_nNodes;
		//		std::cout<<oldl1<<" "<<l1<<" "<<oldl2<<" "<<l2<<" "<<result<<std::endl;
		double trunc=5.0;
		return result;
		return ((result)>trunc?trunc:(result));
	}
	double getWeight(int gridIndex1, int gridIndex2){
		IndexType fixedIndex1=gridToImageIndex(getGridPositionAtIndex(gridIndex1));
		IndexType fixedIndex2=gridToImageIndex(getGridPositionAtIndex(gridIndex2));
		double segWeight=fabs(m_fixedImage->GetPixel(fixedIndex1)-m_fixedImage->GetPixel(fixedIndex2));
		segWeight=exp(-segWeight/10000);
		segWeight*=m_segmentationWeight;
		return segWeight;
	}
	double getPairwisePotential2(int LabelIndex,int LabelIndex2){
		double segmentationSmoothness=fabs(LabelMapperType::getSegmentation(LabelIndex)-LabelMapperType::getSegmentation(LabelIndex2));
		return segmentationSmoothness*m_segmentationWeight;
	}


	IndexType gridToImageIndex(IndexType gridIndex){
		IndexType imageIndex;
		for (unsigned int d=0;d<m_dim;++d){
			int t=gridIndex[d]*m_spacing[d];//+m_spacing[d]/2;
			imageIndex[d]=t>0?t:0;
		}
		return imageIndex;
	}

	IndexType imageToGridIndex(IndexType imageIndex){
		IndexType gridIndex;
		for (int d=0;d<m_dim;++d){
			gridIndex[d]=(imageIndex[d])/m_spacing[d];
			//			gridIndex[d]=(imageIndex[d]-m_spacing[d]/2)/m_spacing[d];
		}
		return gridIndex;
	}
	IndexType getGridPositionAtIndex(int idx){
		IndexType position;
		//		std::cout<<" index :"<<idx;
		for ( int d=m_dim-1;d>=0;--d){
			position[d]=idx/m_imageLevelDivisors[d];
			//			std::cout<<" d:"<<d<<" "<<m_imageLevelDivisors[d]<<" ="<<position[d];
			idx-=position[d]*m_imageLevelDivisors[d];
			//			std::cout<<" "<<idx;
		}
		//		std::cout<<" position:"<<position<<std::endl;
		return position;
	}
	IndexType getImagePositionAtIndex(int idx){
		return gridToImageIndex(getGridPositionAtIndex(idx));
	}

	int  getIntegerIndex(IndexType gridIndex){
		int i=0;
		for (unsigned int d=0;d<m_dim;++d){
			i+=gridIndex[d]*m_imageLevelDivisors[d];
		}
		return i;
	}

	int nNodes(){return m_nNodes;}

	LabelType getResolution(){return m_spacing;}
	int nVertices(){return m_nVertices;}

	std::vector<int> getForwardNeighbours(int index){
		IndexType position=getGridPositionAtIndex(index);
		std::vector<int> neighbours;
		for ( int d=0;d<(int)m_dim;++d){
			OffsetType off;
			off.Fill(0);
			if ((int)position[d]<(int)m_gridSize[d]-1){
				off[d]+=1;
				neighbours.push_back(getIntegerIndex(position+off));
			}
		}
		return neighbours;
	}

	SizeType getTotalSize() const
	{
		return m_totalSize;
	}
	SizeType getGridSize() const
	{return m_gridSize;}

	void setResolution(LabelType m_resolution)
	{
		this->m_resolution = m_resolution;
	}
	ImagePointerType getFixedImage(){
		return m_fixedImage;
	}
	void checkConstraints(LabelImagePointerType labelImage){
		ImagePointerType costMap=ImageType::New();
		costMap->SetRegions(labelImage->GetLargestPossibleRegion());
		costMap->Allocate();
		int vCount=0,totalCount=0;
		for (int n=0;n<m_nNodes;++n){
			std::cout<<n<<" "<<getImagePositionAtIndex(n)<<" "<<getGridPositionAtIndex(n)<<std::endl;
			IndexType idx=getImagePositionAtIndex(n);
			LabelType l1=labelImage->GetPixel(idx);
//			std::cout<<idx<<std::endl;
			//			int labelIndex=LabelMapperType::getIndex(labelImage->GetPixel(idx));
			std::vector<int> nb=getForwardNeighbours(n);
			double localSum=0.0;
			for (int i=0;i<nb.size();++i){
				IndexType idx2=getImagePositionAtIndex(nb[i]);
				LabelType l2=labelImage->GetPixel(idx2);

//				int nBLabel=LabelMapperType::getIndex(labelImage->GetPixel(idx2));
//				double pp=getPairwisePotential(n,nb[i],labelIndex,nBLabel,true);
				double pp=getPairwisePotential(idx,idx2,l1,l2,true);
				if (pp>99999999 ){
					vCount++;
				}
				totalCount++;
				localSum+=pp;
			}
			if (nb.size()) localSum/=nb.size();
			IndexType idx3=getGridPositionAtIndex(n);

			costMap->SetPixel(idx3,localSum);

		}
		ImageUtils<ImageType>::writeImage("costs.png",costMap);
		std::cout<<1.0*vCount/totalCount<<" ratio of violated constraints"<<std::endl;
	}
	LabelImagePointerType getFullLabelImage(LabelImagePointerType labelImg){
#if 1
		const unsigned int SplineOrder = 3;

		typedef typename itk::Image<float,ImageType::ImageDimension> ParamImageType;
		typedef typename itk::ResampleImageFilter<ParamImageType,ParamImageType> ResamplerType;
		typedef typename itk::BSplineResampleImageFunction<ParamImageType,double> FunctionType;
		typedef typename itk::BSplineDecompositionImageFilter<ParamImageType,ParamImageType>			DecompositionType;
		typedef typename  itk::ImageRegionIterator<ParamImageType> Iterator;
		typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;

		std::vector<typename ParamImageType::Pointer> newImages(ImageType::ImageDimension+1);
		for ( unsigned int k = 0; k < ImageType::ImageDimension+1; k++ )
		{
			//			std::cout<<k<<" setup"<<std::endl;
			typename ParamImageType::Pointer paramsK=ParamImageType::New();
			paramsK->SetRegions(labelImg->GetLargestPossibleRegion());
			paramsK->SetOrigin(labelImg->GetOrigin());
			paramsK->SetSpacing(labelImg->GetSpacing());
			paramsK->SetDirection(labelImg->GetDirection());
			paramsK->Allocate();
			Iterator itCoarse( paramsK, paramsK->GetLargestPossibleRegion() );
			LabelIterator itOld(labelImg,labelImg->GetLargestPossibleRegion());
			for (itCoarse.GoToBegin(),itOld.GoToBegin();!itCoarse.IsAtEnd();++itOld,++itCoarse){
				itCoarse.Set(itOld.Get()[k]);
			}
			//			std::cout<<k<<" setCoarse"<<std::endl;
			typename ResamplerType::Pointer upsampler = ResamplerType::New();
			typename FunctionType::Pointer function = FunctionType::New();
			upsampler->SetInput( paramsK );
			upsampler->SetInterpolator( function );
			//			upsampler->SetTransform( identityTransform );
			upsampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize() );
			upsampler->SetOutputSpacing( m_fixedImage->GetSpacing() );
			upsampler->SetOutputOrigin( m_fixedImage->GetOrigin());

			typename DecompositionType::Pointer decomposition = DecompositionType::New();

			decomposition->SetSplineOrder( SplineOrder );
			decomposition->SetInput( upsampler->GetOutput() );
			//			std::cout<<k<<" sampler"<<std::endl;
			decomposition->Update();
			//			std::cout<<k<<" decomp"<<std::endl;

			newImages[k] = decomposition->GetOutput();
		}
		std::vector< Iterator*> iterators(ImageType::ImageDimension+1);
		for ( unsigned int k = 0; k < ImageType::ImageDimension+1; k++ )
		{
			iterators[k]=new Iterator(newImages[k],newImages[k]->GetLargestPossibleRegion());
			iterators[k]->GoToBegin();
		}
		LabelImagePointerType fullLabelImage=LabelImageType::New();
		fullLabelImage->SetRegions(m_fixedImage->GetLargestPossibleRegion());
		fullLabelImage->SetOrigin(m_fixedImage->GetOrigin());
		fullLabelImage->SetSpacing(m_fixedImage->GetSpacing());
		fullLabelImage->SetDirection(m_fixedImage->GetDirection());
		fullLabelImage->Allocate();
		LabelIterator lIt(fullLabelImage,fullLabelImage->GetLargestPossibleRegion());
		lIt.GoToBegin();
		for (;!lIt.IsAtEnd();++lIt){
			LabelType l;
			for ( unsigned int k = 0; k < ImageType::ImageDimension+1; k++ ){
				l[k]=iterators[k]->Get();
				++(*(iterators[k]));
			}
			lIt.Set(l);
		}

		return fullLabelImage;

#else
		typedef typename itk::VectorLinearInterpolateImageFunction<LabelImageType, double> LabelInterpolatorType;
		typedef typename LabelInterpolatorType::Pointer LabelInterpolatorPointerType;
		typedef typename itk::VectorResampleImageFilter< LabelImageType , LabelImageType>	LabelResampleFilterType;
		LabelInterpolatorPointerType labelInterpolator=LabelInterpolatorType::New();
		labelInterpolator->SetInputImage(labelImg);
		//initialise resampler

		typename LabelResampleFilterType::Pointer resampler = LabelResampleFilterType::New();
		//resample deformation field to fixed image dimension
		resampler->SetInput( labelImg );
		resampler->SetInterpolator( labelInterpolator );
		resampler->SetOutputOrigin(getOrigin());//targetImage->GetOrigin());
		resampler->SetOutputSpacing ( m_fixedImage->GetSpacing() );
		resampler->SetOutputDirection ( m_fixedImage->GetDirection() );
		resampler->SetSize ( m_fixedImage->GetLargestPossibleRegion().GetSize() );
		if (verbose) std::cout<<"interpolating deformation field"<<std::endl;
		resampler->Update();
		return resampler->GetOutput();
#endif
	}
};



#endif /* GRIm_dim_H_ */
