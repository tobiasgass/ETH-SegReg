/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef GRAPHITK_H
#define GRAPHITK_H

#include <vector>
#include <assert.h>
#include "itkVectorImage.h"
#include "itkConstNeighborhoodIterator.h"
#include <itkVectorResampleImageFilter.h>
#include <itkResampleImageFilter.h>
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"
#include "Graph.h"


/*
 * Isotropic Graph
 * Returns current/next position in a grid based on size and resolution
 */

template<class TUnaryFunction,class TLabelMapper, class TImage>
class ITKGraphModel: public GraphModel<TUnaryFunction,TLabelMapper, TImage>{
public:
	typedef TUnaryFunction UnaryFunctionType;
	typedef typename UnaryFunctionType::Pointer UnaryFunctionPointerType;
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType OffsetType;
	typedef typename TImage::PointType PointType;
	typedef  GraphModel<TUnaryFunction,TLabelMapper, TImage> Superclass;
	typedef typename TImage::SizeType SizeType;
	typedef  TImage ImageType;
	typedef typename TImage::SpacingType SpacingType;
	typedef typename TImage::Pointer ImagePointerType;
	typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;


public:

	ITKGraphModel(ImagePointerType fixedimage,UnaryFunctionPointerType unaryFunction, int divisor, double displacementScalingFactor, double segmentationWeight, double registrationWeight)
	:Superclass(fixedimage,unaryFunction, divisor, displacementScalingFactor, segmentationWeight, registrationWeight)
	{
		//		assert(this->m_totalSize==movingImage->GetLargestPossibleRegion().GetSize());
		this->m_nNodes=1;
		//		this->m_dblSpacing=this->m_spacing[0];
		setSpacing(divisor);
		for (int d=0;d<(int)this->m_dim;++d){
			if (this->verbose) std::cout<<"total size divided by spacing :"<<1.0*this->m_totalSize[d]/this->m_spacing[d]<<std::endl;
			this->m_origin[d]=(this->m_spacing[d]/2-0.5);
			this->m_gridSize[d]=1.0*this->m_totalSize[d]/((int)this->m_spacing[d]);
			this->m_nNodes*=this->m_gridSize[d];
			if (d>0){
				this->m_imageLevelDivisors[d]=this->m_imageLevelDivisors[d-1]*this->m_gridSize[d-1];
			}else{
				this->m_imageLevelDivisors[d]=1;
			}
		}

		if (this->verbose) std::cout<<"GridSize: "<<this->m_dim<<" ";
		if (this->m_dim>=2){
			if (this->verbose) std::cout<<this->m_gridSize[0]<<" "<<this->m_gridSize[1];
			this->m_nVertices=this->m_gridSize[1]*(this->m_gridSize[0]-1)+this->m_gridSize[0]*(this->m_gridSize[1]-1);
		}
		if (this->m_dim==3){
			std::cout<<" "<<this->m_gridSize[0];
			this->m_nVertices+=(this->m_gridSize[2]-1)*this->m_gridSize[1]*this->m_gridSize[0];
		}
		if (this->verbose) std::cout<<" "<<this->m_nNodes<<" "<<this->m_nVertices<<" "<<LabelMapperType::nLabels<<std::endl;
		//		this->m_ImageInterpolator.SetInput(this->m_movingImage);
	}

	virtual void setSpacing(int divisor){
		SpacingType spacing;
		int minSpacing=999999;
		for (int d=0;d<ImageType::ImageDimension;++d){
			if(this->m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/(divisor) < minSpacing){
				minSpacing=(this->m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/(divisor));
			}
		}
		minSpacing=minSpacing>=1?minSpacing:1.0;
		for (int d=0;d<ImageType::ImageDimension;++d){
			int div=this->m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/minSpacing;
			div=div>0?div:1;
			spacing[d]=(1.0*this->m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/div);
			std::cout<<spacing[d]<<" "<<div<<" "<<this->m_fixedImage->GetLargestPossibleRegion().GetSize()[d]<<" "<<minSpacing<<std::endl;
		}
		this->m_spacing=spacing;
	}

	virtual IndexType gridToImageIndex(IndexType gridIndex){
		IndexType imageIndex;
		for (unsigned int d=0;d<this->m_dim;++d){
			int t=gridIndex[d]*this->m_spacing[d]+this->m_origin[d];
			imageIndex[d]=t>0?t:0;
		}
		return imageIndex;
	}

	virtual IndexType imageToGridIndex(IndexType imageIndex){
		IndexType gridIndex;
		for (unsigned int d=0;d<this->m_dim;++d){
			gridIndex[d]=(imageIndex[d]-this->m_origin[d])/this->m_spacing[d];
		}
		return gridIndex;
	}

};



#endif /* GRIthis->m_dithis->m_H_ */
