#include "Log.h"
/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef SEGGRAPH_H
#define SEGGRAPH_H
 
#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

/*
 * Isotropic Graph
 * Returns current/next position in a grid based on size and resolution
 */
namespace SRS{
    template<class TImage, 
             class TUnarySegmentationFunction,
             class TPairwiseSegmentationFunction,
             class TLabelMapper>
    class SegmentationGraphModel: public itk::Object{
    public:
        typedef SegmentationGraphModel Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        itkNewMacro(Self);

       
        typedef TImage ImageType;
        typedef typename TImage::IndexType IndexType;
        typedef typename TImage::OffsetType OffsetType;
        typedef typename TImage::PointType PointType;
        typedef typename TImage::SizeType SizeType;
        typedef typename TImage::SpacingType SpacingType;
        typedef typename TImage::Pointer ImagePointerType;
        typedef typename TImage::ConstPointer ConstImagePointerType;

        typedef typename itk::ConstNeighborhoodIterator<ImageType> ConstImageNeighborhoodIteratorType;

        typedef TUnarySegmentationFunction UnarySegmentationFunctionType;
        typedef typename UnarySegmentationFunctionType::Pointer UnarySegmentationFunctionPointerType;
         typedef TPairwiseSegmentationFunction PairwiseSegmentationFunctionType;
        typedef typename PairwiseSegmentationFunctionType::Pointer PairwiseSegmentationFunctionPointerType;
        typedef int SegmentationLabelType;
        typedef typename itk::Image<SegmentationLabelType,ImageType::ImageDimension> SegmentationLabelImageType;
        typedef typename SegmentationLabelImageType::Pointer SegmentationLabelImagePointerType;
        
        typedef TLabelMapper LabelMapperType;
    
    protected:
    
        SizeType m_totalSize,m_imageLevelDivisors,m_graphLevelDivisors,m_gridSize, m_imageSize;
    
        //grid spacing in unit pixels
        SpacingType m_gridPixelSpacing;
        //grid spacing in mm
        SpacingType m_gridSpacing, m_imageSpacing;
    
        PointType m_origin;
        double m_DisplacementScalingFactor;
        static const unsigned int m_dim=TImage::ImageDimension;
        int m_nNodes,m_nVertices;
        int m_nEdges;
        UnarySegmentationFunctionPointerType m_unarySegFunction;
        PairwiseSegmentationFunctionPointerType m_pairwiseSegFunction;

        bool verbose;
        bool m_haveLabelMap;
        ConstImagePointerType m_fixedImage;
        ConstImageNeighborhoodIteratorType * m_fixedNeighborhoodIterator;

    public:

        SegmentationGraphModel(){
            assert(m_dim>1);
            assert(m_dim<4);
            m_haveLabelMap=false;
            verbose=true;
            m_fixedImage=NULL;
        };
        ~SegmentationGraphModel(){
            delete m_fixedNeighborhoodIterator;
        }

        void setFixedImage(ConstImagePointerType fixedImage){
            m_fixedImage=fixedImage;
        }
        void initGraph(int dummy=1){
            assert(m_fixedImage);

            //image size
            m_imageSize=m_fixedImage->GetLargestPossibleRegion().GetSize();
            m_imageSpacing=m_fixedImage->GetSpacing();
            LOG<<"Full image resolution: "<<m_imageSize<<endl;
            m_nNodes=1;
	
            for (int d=0;d<(int)m_dim;++d){
                //origin is original origin
                m_origin[d]=m_fixedImage->GetOrigin()[d];
                m_nNodes*=m_imageSize[d];

                //level divisors are used to simplify the calculation of image indices from integer indices
                if (d>0){
                    m_imageLevelDivisors[d]=m_imageLevelDivisors[d-1]*m_imageSize[d-1];
                }else{
                    m_imageLevelDivisors[d]=1;
                }
            }
            LOGV(1)<<"GridSize: "<<m_dim<<" ";
        
            //nvertices is not used!?
            if (m_dim>1){
                m_nEdges=m_imageSize[1]*(m_imageSize[0]-1)+m_imageSize[0]*(m_imageSize[1]-1);
            }
            if (m_dim==3){
                m_nEdges=m_nEdges*this->m_imageSize[2]+(this->m_imageSize[2]-1)*this->m_imageSize[1]*this->m_imageSize[0];
            }
	

            LOGV(1)<<" nodes:"<<m_nNodes<<" totalEdges:"<<m_nEdges<<std::endl;
            
            LOG<<std::flush;
        
        //     typename ConstImageNeighborhoodIteratorType::RadiusType r;
        //     for (int d=0;d<(int)m_dim;++d){
        //         r[d]=(m_gridPixelSpacing[d]/2);
        //     }
        //     m_fixedNeighborhoodIterator=new ConstImageNeighborhoodIteratorType(r,m_fixedImage,m_fixedImage->GetLargestPossibleRegion());

            LOGV(1)<<" finished graph init" <<std::endl;
        }

        virtual int  getImageIntegerIndex(IndexType imageIndex){
            int i=0;
            for (unsigned int d=0;d<m_dim;++d){
                i+=imageIndex[d]*m_imageLevelDivisors[d];
            }
            return i;
        }

        //return position in full image depending on fine graph nodeindex
        virtual IndexType getImageIndex(int idx){
            IndexType position;
            for ( int d=m_dim-1;d>=0;--d){
                position[d]=idx/m_imageLevelDivisors[d];
                idx-=position[d]*m_imageLevelDivisors[d];
            }
            return position;
        }
        double getUnaryPotential(int nodeIndex,int labelIndex){
            IndexType imageIndex=getImageIndex(nodeIndex);
            //Segmentation:labelIndex==segmentationlabel
            double result=m_unarySegFunction->getPotential(imageIndex,labelIndex)/m_nNodes;
            if (result<0)
                LOG<<imageIndex<<" " <<result<<std::endl;
            return result;
        };
        double getPairwisePotential(int nodeIndex1, int nodeIndex2,int labelIndex1, int labelIndex2){
            IndexType imageIndex1=getImageIndex(nodeIndex1);
            IndexType imageIndex2=getImageIndex(nodeIndex2);
            return (labelIndex1!=labelIndex2)*m_pairwiseSegFunction->getPotential(imageIndex1,imageIndex2,labelIndex1, labelIndex2)/m_nEdges;
        }
        double getWeight(int nodeIndex1, int nodeIndex2){
            IndexType imageIndex1=getImageIndex(nodeIndex1);
            IndexType imageIndex2=getImageIndex(nodeIndex2);
            double result=m_unarySegFunction->getWeight(imageIndex1,imageIndex2)/m_nEdges;
            if (result<0)
                LOG<<imageIndex1<<" "<<imageIndex2<<" "<<result<<std::endl;
            return result;
        }
   
        std::vector<int> getForwardNeighbours(int index){
            IndexType position=getImageIndex(index);
            std::vector<int> neighbours;
            for ( int d=0;d<(int)m_dim;++d){
                OffsetType off;
                off.Fill(0);
                if ((int)position[d]<(int)m_imageSize[d]-1){
                    off[d]+=1;
                    neighbours.push_back(getImageIntegerIndex(position+off));
                }
            }
            return neighbours;
        }

        ImagePointerType getSegmentationImage(std::vector<int> labels){
            ImagePointerType result=ImageType::New();
            result->SetRegions(m_fixedImage->GetLargestPossibleRegion());
            result->SetSpacing(m_fixedImage->GetSpacing());
            result->SetDirection(m_fixedImage->GetDirection());
            result->SetOrigin(m_fixedImage->GetOrigin());
            result->Allocate();
            typename itk::ImageRegionIterator<ImageType> it(result,result->GetLargestPossibleRegion());
            int i=0;
            for (it.GoToBegin();!it.IsAtEnd();++it,++i){
                assert(i<labels.size());
                it.Set(labels[i]);
            }
            assert(i==(labels.size()));
            return result;
        }
        SizeType getImageSize() const
        {
            return m_imageSize;
        }
        SizeType getGridSize() const
        {return m_gridSize;}

        int nNodes(){return m_nNodes;}
        int nLabels(){return LabelMapperType::nSegmentations;}
        int nEdges(){return m_nEdges;}

        ImagePointerType getFixedImage(){
            return m_fixedImage;
        }
        void setUnarySegmentationFunction(UnarySegmentationFunctionPointerType func){
            m_unarySegFunction=func;
        }
        void setPairwiseSegmentationFunction(PairwiseSegmentationFunctionPointerType func){
            m_pairwiseSegFunction=func;
        }
    
        typename ImageType::DirectionType getDirection(){return m_fixedImage->GetDirection();}
       
        SpacingType getSpacing(){return m_gridSpacing;}	
        SpacingType getPixelSpacing(){return m_gridPixelSpacing;}
        PointType getOrigin(){return m_origin;}
        int nSegNodes(){
            return m_nNodes;
        }
        int nSegLabels(){
            //return LabelMapperType::nSegmentations;
            return 2;
        }
    }; //GraphModel

}//namespace

#endif /* GRIm_dim_H_ */
