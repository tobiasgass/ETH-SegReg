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

public:

	GraphModel(ImagePointerType fixedimage,UnaryFunctionPointerType unaryFunction, SpacingType res, double displacementScalingFactor, double segmentationWeight, double registrationWeight)
	:m_fixedImage(fixedimage),m_unaryFunction(unaryFunction),m_DisplacementScalingFactor(displacementScalingFactor), m_segmentationWeight(segmentationWeight),m_registrationWeight(registrationWeight)
	{
		verbose=true;
		assert(m_dim>1);
		assert(m_dim<4);
		m_totalSize=fixedimage->GetLargestPossibleRegion().GetSize();
		//		assert(m_totalSize==movingImage->GetLargestPossibleRegion().GetSize());
		m_spacing=res;
		m_nNodes=1;
		//		m_dblSpacing=m_spacing[0];

		if (LabelMapperType::nDisplacementSamples){
			m_labelSpacing=0.4*m_spacing/(LabelMapperType::nDisplacementSamples+1);
			std::cout<<m_spacing<<" "<<LabelMapperType::nDisplacementSamples+1<<" "<<m_labelSpacing<<std::endl;
		}
		for (int d=0;d<(int)m_dim;++d){
			if (verbose) std::cout<<"total size divided by spacing :"<<1.0*m_totalSize[d]/m_spacing[d]<<std::endl;
			m_origin[d]=-int(m_spacing[d]/2);
			m_gridSize[d]=m_totalSize[d]/m_spacing[d];
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
		if (verbose) std::cout<<" "<<m_nNodes<<" "<<m_nVertices<<std::endl;
		//		m_ImageInterpolator.SetInput(m_movingImage);
	}

	void setLabelImage(LabelImagePointerType limg){m_labelImage=limg;}
	void setGradientImage(ImagePointerType limg){m_fixedGradientImage=limg;}

	SpacingType getDisplacementFactor(){return m_labelSpacing*m_DisplacementScalingFactor;}
	SpacingType getSpacing(){return m_spacing;}
	PointType getOrigin(){return m_origin;}
	double getUnaryPotential(int gridIndex, int labelIndex){
		IndexType fixedIndex=gridToImageIndex(getGridPositionAtIndex(gridIndex));
		LabelType label=LabelMapperType::getLabel(labelIndex);
		typename itk::ConstNeighborhoodIterator<ImageType>::RadiusType radius;
		for (unsigned int d=0;d<m_dim;++d){
			radius[d]=m_spacing[d]*0.45;
		}

		typename itk::ConstNeighborhoodIterator<ImageType> nIt(radius,m_fixedImage, m_fixedImage->GetLargestPossibleRegion());
		nIt.SetLocation(fixedIndex);
		double res=0.0;
		double count=0;
		for (unsigned int i=0;i<nIt.Size();++i){
			bool inBounds;
			nIt.GetPixel(i,inBounds);
			if (inBounds){
				res+=m_unaryFunction->getPotential(nIt.GetIndex(i),label);
				++count;
			}
		}
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
			double tmp=(l1[d]-l2[d])*m_labelSpacing[d]*m_DisplacementScalingFactor;
			result+=tmp*tmp;
		}
		//		std::cout<<sqrt(result)<<std::endl;
		double trunc=5.0;
		return (sqrt(result)>trunc?trunc:sqrt(result));
	}

	double getPairwisePotential(int idx1,int idx2,int LabelIndex,int LabelIndex2){
		IndexType fixedIndex1=gridToImageIndex(getGridPositionAtIndex(idx2));
		IndexType fixedIndex2=gridToImageIndex(getGridPositionAtIndex(idx1));
#if 1
		LabelType l1=LabelMapperType::getLabel(LabelIndex);
		LabelType l2=LabelMapperType::getLabel(LabelIndex2);
		LabelType oldl1=m_labelImage->GetPixel(fixedIndex1);
		LabelType oldl2=m_labelImage->GetPixel(fixedIndex2);
		double registrationSmootheness=0;
		double segmentationSmootheness=0;
		if (LabelMapperType::nSegmentations){
			segmentationSmootheness=fabs(LabelMapperType::getSegmentation(l1)-LabelMapperType::getSegmentation(l2));
#if 1
			double segWeight=fabs(m_fixedImage->GetPixel(fixedIndex1)-m_fixedImage->GetPixel(fixedIndex2));
			segWeight=exp(-segWeight/3000);
#else
			assert(m_fixedGradientImage);
			//if gradient changes, we would also like a segmentation label change
			double segWeight=1-fabs(m_fixedGradientImage->GetPixel(fixedIndex1) || m_fixedGradientImage->GetPixel(fixedIndex2));
#endif
			segmentationSmootheness*=segWeight*m_segmentationWeight;
		}
		double constrainedViolatedPenalty=99999;//std::numeric_limits<double>::max()/(m_nNodes*1000);;
		bool constrainsViolated=false;
		//		std::cout<<"DeltaInit: "<<fixedIndex1<<" "<<fixedIndex2<<" "<<l1+oldl1<<" "<<l2+oldl2<<std::endl;
		if (LabelMapperType::nDisplacements){
			double d1,d2;
			int delta;
			for (unsigned int d=0;d<m_dim;++d){
				//applying the labels to evaluate to neighboring pixels
				d1=(l1[d])*m_labelSpacing[d]*m_DisplacementScalingFactor+oldl1[d];
				d2=(l2[d])*m_labelSpacing[d]*m_DisplacementScalingFactor+oldl2[d];
				delta=(fixedIndex2[d]-fixedIndex1[d]);

				double relativeAxisPositionDifference=1.0*(d2+delta-d1)/m_spacing[d];
//				std::cout<<"Delta :"<<delta<<" "<<m_spacing[d]<<" "<<relativeAxisPositionDifference<<std::endl;
				//we shall never tear the image!
				if (delta>0){
					if (relativeAxisPositionDifference<0){
						constrainsViolated=true;
//						exit(0);
						break;
					}
				}
				else if (delta<0){
					if (relativeAxisPositionDifference>0){
						constrainsViolated=true;
						break;
					}
				}
				if (fabs(relativeAxisPositionDifference)>2){
					constrainsViolated=true;
//					exit(0);
					break;
				}

				registrationSmootheness+=(relativeAxisPositionDifference)*(relativeAxisPositionDifference);
			}
			registrationSmootheness*=m_registrationWeight;

		}

		if (constrainsViolated){
			return 	m_registrationWeight*constrainedViolatedPenalty;
		}
//		std::cout<<registrationSmootheness<<std::endl;
		double result=registrationSmootheness+segmentationSmootheness;
		//		std::cout<<oldl1<<" "<<l1<<" "<<oldl2<<" "<<l2<<" "<<result<<std::endl;
#else
		typename itk::ConstNeighborhoodIterator<LabelImageType>::RadiusType radius;
		assert(m_labelImage);
		for (unsigned int d=0;d<m_dim;++d){
			radius[d]=m_spacing[d]*0.45;
		}
		typename itk::ConstNeighborhoodIterator<LabelImageType> nIt(radius,m_labelImage, m_labelImage->GetLargestPossibleRegion());
		typename itk::ConstNeighborhoodIterator<LabelImageType> nIt2(radius,m_labelImage, m_labelImage->GetLargestPossibleRegion());
		nIt.SetLocation(fixedIndex1);
		nIt2.SetLocation(fixedIndex2);
		int count=0;
		double result=0;
		LabelType l1=LabelMapperType::getLabel(LabelIndex);
		LabelType l2=LabelMapperType::getLabel(LabelIndex2);
		for (unsigned int i=0;i<nIt.Size();++i){
			bool inBounds1, inBounds2;
			LabelType oldl1=nIt.GetPixel(i,inBounds1);
			LabelType oldl2=nIt2.GetPixel(i,inBounds2);
			if (inBounds1&&inBounds2){


				//	LabelType l=l1-l2;
				for (unsigned int d=0;d<m_dim;++d){
					//applying the labels to evaluate to neighboring pixels
					double tmp=(l1[d]-l2[d])*m_labelSpacing[d]*m_DisplacementScalingFactor;
					double tmp2=oldl1[d]-oldl2[d];
					result+=fabs(tmp+tmp2);
				}
				++count;
			}
		}
		result/=count;
		//		std::cout<<sqrt(result)<<std::endl;
#endif
		double trunc=5.0;
		return result;
		return ((result)>trunc?trunc:(result));
	}
	double getWeight(int gridIndex1, int gridIndex2){return 1.0;}
	double getPairwisePotential2(int LabelIndex,int LabelIndex2){ return 0;}


	IndexType gridToImageIndex(IndexType gridIndex){
		IndexType imageIndex;
		for (unsigned int d=0;d<m_dim;++d){
			int t=gridIndex[d]*m_spacing[d]+m_spacing[d]/2;
			imageIndex[d]=t>0?t:0;
		}
		return imageIndex;
	}

	IndexType imageToGridIndex(IndexType imageIndex){
		IndexType gridIndex;
		for (int d=0;d<m_dim;++d){
			gridIndex[d]=(imageIndex[d]-m_spacing[d]/2)/m_spacing[d];
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

};



#endif /* GRIm_dim_H_ */
