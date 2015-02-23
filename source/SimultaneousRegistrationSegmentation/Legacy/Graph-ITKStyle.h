#include "Log.h"
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
namespace itk{
template<class TImage, 
         class TUnaryRegistrationFunction, 
         class TPairwiseRegistrationFunction, 
         class TUnarySegmentationFunction, 
         class TPairwiseSegmentationFunction,
         class TPairwiseSegmentationRegistrationFunction,
         class TLabelMapper>
class ITKGraphModel: public  GraphModel<TImage,
                                        TLabelMapper>
{
public:
	typedef TLabelMapper LabelMapperType;
	typedef typename LabelMapperType::LabelType LabelType;
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType OffsetType;
	typedef typename TImage::PointType PointType;
   typedef GraphModel<TImage,
                           TLabelMapper> Superclass;
	typedef typename TImage::SizeType SizeType;
	typedef  TImage ImageType;
	typedef typename TImage::SpacingType SpacingType;
	typedef typename TImage::Pointer ImagePointerType;
	typedef typename itk::Image<LabelType,ImageType::ImageDimension> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
    typedef typename itk::ConstNeighborhoodIterator<ImageType> ConstImageNeighborhoodIteratorType;

private:
	SpacingType m_offset;
public:

    void initGraph(int nGraphNodesPerEdge){
        assert(this->m_fixedImage);
        
        //image size
        this->m_imageSize=this->m_fixedImage->GetLargestPossibleRegion().GetSize();
        this->m_imageSpacing=this->m_fixedImage->GetSpacing();
        LOG<<"Full image resolution: "<<this->m_imageSize<<endl;
        this->m_nSegmentationNodes=1;
        this->m_nRegistrationNodes=1;
        //calculate graph spacing
        setSpacing(nGraphNodesPerEdge);
        
        if (LabelMapperType::nDisplacementSamples){
            
            this->m_labelSpacing=0.4*this->m_gridPixelSpacing/(LabelMapperType::nDisplacementSamples);
            if (this->verbose) LOG<<"Spacing :"<<this->m_gridPixelSpacing<<" "<<LabelMapperType::nDisplacementSamples<<" labelSpacing :"<<this->m_labelSpacing<<std::endl;
        }
        for (int d=0;d<(int)this->m_dim;++d){
            if (this->verbose) LOG<<"total size divided by spacing :"<<1.0*this->m_imageSize[d]/this->m_gridPixelSpacing[d]<<std::endl;

            //origin is original origin
			this->m_origin[d]=this->m_fixedImage->GetOrigin()[d]+(this->m_gridSpacing[d]/2-0.5);
            //
            this->m_gridSize[d]=this->m_imageSize[d]/this->m_gridPixelSpacing[d];
         
            this->m_nRegistrationNodes*=this->m_gridSize[d];
            this->m_nSegmentationNodes*=this->m_imageSize[d];

            //level divisors are used to simplify the calculation of image indices from integer indices
            if (d>0){
                this->m_imageLevelDivisors[d]=this->m_imageLevelDivisors[d-1]*this->m_imageSize[d-1];
                this->m_graphLevelDivisors[d]=this->m_graphLevelDivisors[d-1]*this->m_gridSize[d-1];
            }else{
                this->m_imageLevelDivisors[d]=1;
                this->m_graphLevelDivisors[d]=1;
            }
        }
        this->m_nNodes=this->m_nRegistrationNodes+this->m_nSegmentationNodes;
        if (this->verbose) LOG<<"GridSize: "<<this->m_dim<<" ";
        
        //nvertices is not used!?
        if (this->m_dim>=2){
            if (this->verbose) LOG<<this->m_gridSize[0]<<" "<<this->m_gridSize[1];
            this->m_nRegEdges=this->m_gridSize[1]*(this->m_gridSize[0]-1)+this->m_gridSize[0]*(this->m_gridSize[1]-1);
            this->m_nSegEdges=this->m_imageSize[1]*(this->m_imageSize[0]-1)+this->m_imageSize[0]*(this->m_imageSize[1]-1);
        }
        if (this->m_dim==3){
            LOG<<" "<<this->m_gridSize[2];
            this->m_nRegEdges=this->m_nRegEdges*this->m_gridSize[2]+(this->m_gridSize[2]-1)*this->m_gridSize[1]*this->m_gridSize[0];
            this->m_nSegEdges=this->m_nSegEdges*this->m_imageSize[2]+(this->m_imageSize[2]-1)*this->m_imageSize[1]*this->m_imageSize[0];
        }

        typename ConstImageNeighborhoodIteratorType::RadiusType r;
        //controls the size of the neighborhood for registration-to-segmentation edges
        double reductionFactor=1;
        for (int d=0;d<(int)this->m_dim;++d){
            r[d]=(this->m_gridPixelSpacing[d]/(2*reductionFactor));
        }
        this->m_fixedNeighborhoodIterator=ConstImageNeighborhoodIteratorType(r,this->m_fixedImage,this->m_fixedImage->GetLargestPossibleRegion());
        this->m_nSegRegEdges=this->m_nSegmentationNodes/pow(reductionFactor,this->m_dim);
        this->m_nEdges=this->m_nRegEdges+this->m_nSegEdges+this->m_nSegRegEdges;
        if (this->verbose) LOG<<" nodes:"<<this->m_nNodes<<" totalEdges:"<<this->m_nRegEdges+this->m_nSegEdges+this->m_nSegRegEdges<<" labels:"<<LabelMapperType::nLabels<<std::endl;
        if (this->verbose) LOG<<" Segnodes:"<<this->m_nSegmentationNodes<<"\t SegEdges :"<<this->m_nSegEdges<<std::endl
                              <<" Regnodes:"<<this->m_nRegistrationNodes<<"\t\t RegEdges :"<<this->m_nRegEdges<<std::endl
                              <<" SegRegEdges:"<<this->m_nSegRegEdges<<std::endl;
                         
        
       
        if (this->verbose) LOG<<" finished graph init" <<std::endl;
    }
    
  
	virtual void setSpacing(int divisor){
		SpacingType spacing;
		int minSpacing=999999;
		for (int d=0;d<ImageType::ImageDimension;++d){
			if((int)this->m_imageSize[d]/(divisor) < minSpacing){
				minSpacing=(this->m_imageSize[d]/(divisor));
			}
		}
		minSpacing=minSpacing>=1?minSpacing:1.0;
		for (int d=0;d<ImageType::ImageDimension;++d){
			int div=this->m_imageSize[d]/minSpacing;
			div=div>0?div:1;
			spacing[d]=(1.0*this->m_imageSize[d]/div);
            this->m_gridPixelSpacing[d]=spacing[d];
			this->m_gridSpacing[d]=spacing[d]*this->m_fixedImage->GetSpacing()[d];
		}

	}
    //return position in full image from coarse graph node index
    virtual IndexType  getImageIndexFromCoarseGraphIndex(int idx){
        IndexType position;
        for ( int d=this->m_dim-1;d>=0;--d){
            //position[d] is now the index in the coarse graph (image)
                position[d]=idx/this->m_graphLevelDivisors[d];
                idx-=position[d]*this->m_graphLevelDivisors[d];
                //now calculate the fine image index from the coarse graph index
                position[d]*=this->m_gridSpacing[d]/this->m_imageSpacing[d];
                position[d]+=0.5*this->m_gridSpacing[d]/this->m_imageSpacing[d];
        }
        assert(this->m_fixedImage->GetLargestPossibleRegion().IsInside(position));
        return position;
    }

  

};//class

}//namespace



#endif /* GRIthis->m_dithis->m_H_ */
