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
	typedef typename TImage::SizeType SizeType;
	typedef  TImage ImageType;
	typedef typename TImage::Pointer ImagePointerType;
	typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;

private:
	ImagePointerType m_fixedImage;
	LabelImagePointerType m_labelImage;
	SizeType m_totalSize,m_gridSize,m_imageLevelDivisors,m_spacing;
	static const unsigned int m_dim=TImage::ImageDimension;
	int m_nNodes,m_nVertices;
//	ImageInterpolatorType m_ImageInterpolator,m_SegmentationInterpolator,m_BoneConfidenceInterploator;
	UnaryFunctionPointerType m_unaryFunction;
public:

	GraphModel(ImagePointerType fixedimage,UnaryFunctionPointerType unaryFunction, SizeType res):m_fixedImage(fixedimage),m_unaryFunction(unaryFunction){
		assert(m_dim>1);
		assert(m_dim<4);
		m_totalSize=fixedimage->GetLargestPossibleRegion().GetSize();
//		assert(m_totalSize==movingImage->GetLargestPossibleRegion().GetSize());
		m_spacing=res;
		m_nNodes=1;
		for (int d=0;d<m_dim;++d){
			m_gridSize[d]=m_totalSize[d]/m_spacing[d];
			m_nNodes*=m_gridSize[d];
			if (d>0){
				m_imageLevelDivisors[d]=m_imageLevelDivisors[d-1]*m_gridSize[d-1];
			}else{
				m_imageLevelDivisors[d]=1;
			}
		}

		std::cout<<"GridSize: "<<m_dim<<" ";
		if (m_dim>=2){
			std::cout<<m_gridSize[0]<<" "<<m_gridSize[1];
			m_nVertices=m_gridSize[1]*(m_gridSize[0]-1)+m_gridSize[0]*(m_gridSize[1]-1);
		}
		if (m_dim==3){
			std::cout<<" "<<m_gridSize[0];
			m_nVertices+=(m_gridSize[2]-1)*m_gridSize[1]*m_gridSize[0];
		}
		std::cout<<" "<<m_nNodes<<" "<<m_nVertices<<std::endl;
//		m_ImageInterpolator.SetInput(m_movingImage);
		m_labelImage=LabelImageType::New();
		m_labelImage->SetRegions(m_fixedImage->GetLargestPossibleRegion());
		m_labelImage->SetSpacing(1.0);
		m_labelImage->SetNumberOfComponentsPerPixel(LabelMapperType::k);
		m_labelImage->Allocate();
	}

	double getUnaryPotential(int gridIndex, int labelIndex){
		IndexType fixedIndex=gridToImageIndex(getGridPositionAtIndex(gridIndex));
		LabelType label=LabelMapperType::getLabel(labelIndex);
//		ContinuousIndexType movingIndex=fixedIndex+label.getDisplacement();
		itk::Vector<double> test(3);
		int count=0;
//		int radius=m_spacing/2;
		double res=0.0;
		OffsetType off;//=m_spacing/2;
//		off.Fill(-radius);
#if 0
		for (int d=0;d<m_dim;++d){
			off[0]++;
			for (int e=0;e<m_dim;++e){
				if (off[e]>radius){
					off[e]-radius;
					off[e+1]++;
				}
#endif
//				if (fixedIndex+off<m_totalSize )//&& movingIndex+off<m_movingSize)
				res+=m_unaryFunction->getPotential(fixedIndex,label);
				++count;
#if 0
			}
		}
#endif
		if (count>0)
			return res/count;
		else return 999999;
	}
	double getPairwisePotential(int LabelIndex,int LabelIndex2){
		LabelType l1=LabelMapperType::getLabel(LabelIndex);
		LabelType l2=LabelMapperType::getLabel(LabelIndex2);
	//	LabelType l=l1-l2;
		double result=0;
		for (unsigned int d=0;d<m_dim;++d){
			double tmp=l1[d]-l2[d];
			result+=tmp*tmp;
		}
		return sqrt(result);
	}
	double getWeight(int gridIndex1, int gridIndex2){return 1.0;}
	double getPairwisePotential2(int LabelIndex,int LabelIndex2){ return 0;}


	IndexType gridToImageIndex(IndexType gridIndex){
		IndexType imageIndex;
		for (int d=0;d<m_dim;++d){
			imageIndex[d]=gridIndex[d]*m_spacing[d];
		}
		return imageIndex;
	}

	IndexType imageToGridIndex(IndexType imageIndex){
		IndexType gridIndex;
		for (int d=0;d<m_dim;++d){
			gridIndex[d]=imageIndex[d]/m_spacing[d];
		}
		return gridIndex;
	}
	IndexType getGridPositionAtIndex(int idx){
		IndexType position;
		for (int d=m_dim-1;d>=0;--d){
			position[d]=idx/m_imageLevelDivisors[d];
			idx-=position[d]*m_imageLevelDivisors[d];
		}
		return position;
	}
	IndexType getImagePositionAtIndex(int idx){
		return gridToImageIndex(getGridPositionAtIndex(idx));
	}

	int  getIntegerIndex(IndexType gridIndex){
		int i=0;
		for (int d=0;d<m_dim;++d){
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
		for (int d=0;d<m_dim;++d){
			OffsetType off;
			off.Fill(0);
			if (position[d]<m_gridSize[d]-1){
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

	void setResolution(LabelType m_resolution)
	{
		this->m_resolution = m_resolution;
	}
	ImagePointerType getFixedImage(){
		return m_fixedImage;
	}

};



#endif /* GRIm_dim_H_ */
