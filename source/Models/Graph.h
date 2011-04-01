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

protected:
	ImagePointerType m_fixedImage,m_fixedGradientImage,m_backProjFixedImage;
	LabelImagePointerType m_fullLabelImage,m_backProjLabelImage;
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

	GraphModel(ImagePointerType fixedimage,UnaryFunctionPointerType unaryFunction, int divisor, double displacementScalingFactor, double segmentationWeight, double registrationWeight)
	:m_fixedImage(fixedimage),m_unaryFunction(unaryFunction),m_DisplacementScalingFactor(displacementScalingFactor), m_segmentationWeight(segmentationWeight),m_registrationWeight(registrationWeight)
	{
		m_haveLabelMap=false;
		verbose=true;
		assert(m_dim>1);
		assert(m_dim<4);
		m_totalSize=fixedimage->GetLargestPossibleRegion().GetSize();
		m_nNodes=1;
		setSpacing(divisor);
		if (LabelMapperType::nDisplacementSamples){
			m_labelSpacing=0.4*m_spacing/(LabelMapperType::nDisplacementSamples);
			if (verbose) std::cout<<"Spacing :"<<m_spacing<<" "<<LabelMapperType::nDisplacementSamples<<" labelSpacing :"<<m_labelSpacing<<std::endl;
		}
		for (int d=0;d<(int)m_dim;++d){
			if (verbose) std::cout<<"total size divided by spacing :"<<1.0*m_totalSize[d]/m_spacing[d]<<std::endl;
			m_origin[d]=0;//-int(m_spacing[d]/2);
			m_gridSize[d]=m_totalSize[d]/m_spacing[d];
			if (m_spacing!=1.0)
				m_gridSize[d]++;
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
	virtual void setSpacing(int divisor){
		SpacingType spacing;
		int minSpacing=999999;
		for (int d=0;d<ImageType::ImageDimension;++d){
			if(m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/(divisor-1) < minSpacing){
				minSpacing=(m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/(divisor-1)-1);
			}
		}
		minSpacing=minSpacing>=1?minSpacing:1.0;
		for (int d=0;d<ImageType::ImageDimension;++d){
			int div=m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/minSpacing;
			div=div>0?div:1;
			spacing[d]=(1.0*m_fixedImage->GetLargestPossibleRegion().GetSize()[d]/div);
			if (spacing[d]>1.0) spacing[d]-=1.0/(div);
		}
		m_spacing=spacing;
	}
	typename ImageType::DirectionType getDirection(){return m_fixedImage->GetDirection();}
	void setLabelImage(LabelImagePointerType limg){m_fullLabelImage=limg;m_haveLabelMap=true;}
	void setGradientImage(ImagePointerType limg){m_fixedGradientImage=limg;}

	void setDisplacementFactor(double fac){m_DisplacementScalingFactor=fac;}
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

	virtual double getPairwisePotential(int idx1,int idx2,int LabelIndex,int LabelIndex2,bool verbose=false){
		IndexType fixedIndex1=gridToImageIndex(getGridPositionAtIndex(idx1));
		IndexType fixedIndex2=gridToImageIndex(getGridPositionAtIndex(idx2));
		LabelType l1=LabelMapperType::getLabel(LabelIndex);
		LabelType l2=LabelMapperType::getLabel(LabelIndex2);
		return getPairwisePotential(fixedIndex1,fixedIndex2,l1,l2);
	}
	virtual double getPairwisePotential(IndexType fixedIndex1,IndexType fixedIndex2,LabelType l1,LabelType l2,bool verbose=false){

		double edgeWeight=fabs(m_backProjFixedImage->GetPixel(imageToGridIndex(fixedIndex1))-m_backProjFixedImage->GetPixel(imageToGridIndex(fixedIndex2)));
		//		std::cout<<edgeWeight<<" "<<exp(-edgeWeight)<<std::endl;
		edgeWeight=1;//exp(-edgeWeight);

		//segmentation smoothness
		double segmentationSmootheness=0;
		if (LabelMapperType::nSegmentations){
			segmentationSmootheness=fabs(LabelMapperType::getSegmentation(l1)-LabelMapperType::getSegmentation(l2));
			//this weight should rather depend on the interpolated regions
			segmentationSmootheness*=m_segmentationWeight;
		}

		double registrationSmootheness=0;
		if (LabelMapperType::nDisplacements){
#if 0
			LabelType oldl1=m_backProjLabelImage->GetPixel(imageToGridIndex(fixedIndex1));
			LabelType oldl2=m_backProjLabelImage->GetPixel(imageToGridIndex(fixedIndex2));
#else
			LabelType oldl1=m_fullLabelImage->GetPixel((fixedIndex1));
			LabelType oldl2=m_fullLabelImage->GetPixel((fixedIndex2));
#endif
			double constrainedViolatedPenalty=65535;//9999999999;
			bool constrainsViolated=false;
			double d1,d2;
			int delta;
			LabelType displacement1=LabelMapperType::scaleDisplacement(l1,getDisplacementFactor());//+oldl1;
			LabelType displacement2=LabelMapperType::scaleDisplacement(l2,getDisplacementFactor());//+oldl2;
#if 1
			displacement1+=oldl1;
			displacement2+=oldl2;
#endif
			for (unsigned int d=0;d<m_dim;++d){

				d1=displacement1[d];
				d2=displacement2[d];
				delta=(fixedIndex2[d]-fixedIndex1[d]);

				double axisPositionDifference=1.0*(d2-d1);//(m_spacing[d]);

				double relativeAxisPositionDifference=1.0*(axisPositionDifference+(1.0*delta/m_spacing[d]));

				//we shall never tear the image!
				if (delta>0){
					if (relativeAxisPositionDifference<0.2){
						constrainsViolated=true;
					}
				}
				else if (delta<0){
					if (relativeAxisPositionDifference>-0.20){
						constrainsViolated=true;
					}
				}
				if (fabs(relativeAxisPositionDifference)>1.5){
					constrainsViolated=true;
				}

				registrationSmootheness+=(axisPositionDifference)*(axisPositionDifference);
			}
			registrationSmootheness*=m_registrationWeight;
			if (false){
				std::cout<<"DeltaInit1: "<<fixedIndex1<<" ->"<<oldl1<<"+"<<displacement1<<" ,"<<fixedIndex2<<" ->"<<oldl2<<"+"<<displacement2<<" :"<<registrationSmootheness<<std::endl;
			}
			if (constrainsViolated){
//								return 	constrainedViolatedPenalty;
			}
		}
		double result=edgeWeight*(registrationSmootheness+segmentationSmootheness)/m_nVertices;
		return result;
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


	virtual IndexType gridToImageIndex(IndexType gridIndex){
		IndexType imageIndex;
		for (unsigned int d=0;d<m_dim;++d){
			int t=gridIndex[d]*m_spacing[d];
			imageIndex[d]=t>0?t:0;
		}
		return imageIndex;
	}

	virtual IndexType imageToGridIndex(IndexType imageIndex){
		IndexType gridIndex;
		for (int d=0;d<m_dim;++d){
			gridIndex[d]=(imageIndex[d])/m_spacing[d];
		}
		return gridIndex;
	}
	virtual IndexType getGridPositionAtIndex(int idx){
		IndexType position;
		for ( int d=m_dim-1;d>=0;--d){
			position[d]=idx/m_imageLevelDivisors[d];
			idx-=position[d]*m_imageLevelDivisors[d];
		}
		return position;
	}
	virtual IndexType getImagePositionAtIndex(int idx){
		return gridToImageIndex(getGridPositionAtIndex(idx));
	}

	virtual int  getIntegerIndex(IndexType gridIndex){
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
	void checkConstraints(LabelImagePointerType labelImage, std::string filename='costs.png'){
		ImagePointerType costMap=ImageType::New();
		costMap->SetRegions(labelImage->GetLargestPossibleRegion());
		costMap->Allocate();
		int vCount=0,totalCount=0;
		for (int n=0;n<m_nNodes;++n){
			IndexType imageIdx=getImagePositionAtIndex(n);
			IndexType gridIdx=getGridPositionAtIndex(n);
			LabelType l1=labelImage->GetPixel(gridIdx);
			std::vector<int> nb=getForwardNeighbours(n);
			double localSum=0.0;
			for (unsigned int i=0;i<nb.size();++i){
				IndexType neighbGridIdx=getGridPositionAtIndex(nb[i]);
				IndexType neighbImageIdx=gridToImageIndex(neighbGridIdx);
				LabelType l2=labelImage->GetPixel(neighbGridIdx);
				double pp=getPairwisePotential(imageIdx,neighbImageIdx,l1,l2,true);
				if (pp>99 ){
					vCount++;
				}
				totalCount++;
				localSum+=pp;
			}
			if (nb.size()) localSum/=nb.size();
			costMap->SetPixel(gridIdx,localSum);
			//			std::cout<<gridIdx<<" "<<65535*localSum<<std::endl;

		}
		ImageUtils<ImageType>::writeImage(filename,costMap);

		std::cout<<1.0*vCount/(totalCount+0.0000000001)<<" ratio of violated constraints"<<std::endl;
	}
	LabelImagePointerType getFullLabelImage(LabelImagePointerType labelImg){
		typedef typename  itk::ImageRegionIterator<LabelImageType> LabelIterator;
#if 1
		const unsigned int SplineOrder = 3;

		typedef typename itk::Image<float,ImageType::ImageDimension> ParamImageType;
		typedef typename itk::ResampleImageFilter<ParamImageType,ParamImageType> ResamplerType;
		typedef typename itk::BSplineResampleImageFunction<ParamImageType,double> FunctionType;
		typedef typename itk::BSplineDecompositionImageFilter<ParamImageType,ParamImageType>			DecompositionType;
		typedef typename  itk::ImageRegionIterator<ParamImageType> Iterator;

		std::vector<typename ParamImageType::Pointer> newImages(ImageType::ImageDimension+1);
		//interpolate deformation
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
			if (k<ImageType::ImageDimension){
				typename ResamplerType::Pointer upsampler = ResamplerType::New();
				typename FunctionType::Pointer function = FunctionType::New();
				upsampler->SetInput( paramsK );
				upsampler->SetInterpolator( function );
				upsampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize() );
				upsampler->SetOutputSpacing( m_fixedImage->GetSpacing() );
				upsampler->SetOutputOrigin( m_fixedImage->GetOrigin());
				typename DecompositionType::Pointer decomposition = DecompositionType::New();
				decomposition->SetSplineOrder( SplineOrder );
				decomposition->SetInput( upsampler->GetOutput() );
				decomposition->Update();
				newImages[k] = decomposition->GetOutput();
			}
			else{
				typedef typename itk::LinearInterpolateImageFunction<ParamImageType, double> InterpolatorType;
				typedef typename InterpolatorType::Pointer InterpolatorPointerType;
				typedef typename itk::ResampleImageFilter< ParamImageType , ParamImageType>	ParamResampleFilterType;
				InterpolatorPointerType interpolator=InterpolatorType::New();
				interpolator->SetInputImage(paramsK);
				typename ParamResampleFilterType::Pointer resampler = ParamResampleFilterType::New();
				//resample deformation field to fixed image dimension
				resampler->SetInput( paramsK );
				resampler->SetInterpolator( interpolator );
				resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
				resampler->SetOutputSpacing ( m_fixedImage->GetSpacing() );
				resampler->SetOutputDirection ( m_fixedImage->GetDirection() );
				resampler->SetSize ( m_fixedImage->GetLargestPossibleRegion().GetSize() );
				resampler->Update();
				newImages[k] = resampler->GetOutput();
			}
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
			lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
			//			lIt.Set(l);
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
		resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
		resampler->SetOutputSpacing ( m_fixedImage->GetSpacing() );
		resampler->SetOutputDirection ( m_fixedImage->GetDirection() );
		resampler->SetSize ( m_fixedImage->GetLargestPossibleRegion().GetSize() );
		if (verbose) std::cout<<"interpolating deformation field"<<std::endl;
		resampler->Update();
		LabelImagePointerType fullLabelImage=resampler->GetOutput();
		LabelIterator lIt(fullLabelImage,fullLabelImage->GetLargestPossibleRegion());
		lIt.GoToBegin();
		for (;!lIt.IsAtEnd();++lIt){
			LabelType l=lIt.Get();
			lIt.Set(LabelMapperType::scaleDisplacement(l,getDisplacementFactor()));
		}
		return fullLabelImage;
#endif
	}

	void calculateBackProjections(){
		m_backProjFixedImage=ImageType::New();
		typename ImageType::RegionType imRegion;
		imRegion.SetSize(m_gridSize);
		m_backProjFixedImage->SetOrigin(m_origin);
		m_backProjFixedImage->SetRegions(imRegion);
		m_backProjFixedImage->Allocate();
		m_backProjLabelImage=LabelImageType::New();
		typename LabelImageType::RegionType region;
		region.SetSize(m_gridSize);
		m_backProjLabelImage->SetOrigin(m_origin);
		m_backProjLabelImage->SetRegions(region);
		m_backProjLabelImage->Allocate();
		bool inBounds;
		typename itk::ConstNeighborhoodIterator<LabelImageType> nIt(this->m_unaryFunction->getRadius(),this->m_fullLabelImage, this->m_fullLabelImage->GetLargestPossibleRegion());
		typename itk::ConstNeighborhoodIterator<ImageType> fixedIt(this->m_unaryFunction->getRadius(),this->m_fixedImage, this->m_fixedImage->GetLargestPossibleRegion());

		for (int i=0;i<m_nNodes;++i){

			IndexType gridIndex=getGridPositionAtIndex(i);
			IndexType imageIndex=gridToImageIndex(gridIndex);
			nIt.SetLocation(imageIndex);
			fixedIt.SetLocation(imageIndex);

			double weightSum=0.0;
			LabelType labelSum;
			float valSum=0.0;
			for (unsigned int n=0;n<nIt.Size();++n){
				LabelType l=nIt.GetPixel(n,inBounds);
				float val=fixedIt.GetPixel(n);
				//				std::cout<<val<<std::endl;
				if (inBounds){
					IndexType neighborIndex=nIt.GetIndex(n);
					double weight=1.0;
					for (int d=0;d<ImageType::ImageDimension;++d){
						weight*=1-(1.0*fabs(neighborIndex[d]-imageIndex[d]))/(m_spacing[d]); //uhuh radius==spacing?
						//weight+=w/ImageType::ImageDimension;
					}
					//					std::cout<<n<<" "<<neighborIndex<<" "<<imageIndex<<" "<<weight<<" "<<val<<std::endl;
					labelSum+=l*weight;
					valSum+=val*weight;
					weightSum+=weight;
				}

			}
			labelSum/=weightSum;
			m_backProjLabelImage->SetPixel(gridIndex,labelSum);
			//			std::cout<<gridIndex<<" "<<imageIndex<<" "<<valSum<<" "<<weightSum<<" "<<valSum/weightSum<<std::endl;
			valSum/=weightSum;
			m_backProjFixedImage->SetPixel(gridIndex,valSum);
		}
	}
};



#endif /* GRIm_dim_H_ */
