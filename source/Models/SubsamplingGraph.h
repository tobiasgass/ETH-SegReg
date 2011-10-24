/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef SSGRAPH_H
#define SSGRAPH_H
 
#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include <limits>
#include "Graph.h"
using namespace std;
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
    class SubsamplingGraphModel: 
        public GraphModel<TImage,
                          TUnaryRegistrationFunction,
                          TPairwiseRegistrationFunction,
                          TUnarySegmentationFunction, 
                          TPairwiseSegmentationFunction,
                          TPairwiseSegmentationRegistrationFunction,
                          TLabelMapper>
    {
    public:
        typedef SubsamplingGraphModel Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef GraphModel<TImage,
                          TUnaryRegistrationFunction,
                          TPairwiseRegistrationFunction,
                          TUnarySegmentationFunction, 
                          TPairwiseSegmentationFunction,
                          TPairwiseSegmentationRegistrationFunction,
                          TLabelMapper> Superclass;
        itkNewMacro(Self);

        //    typedef  itk::ImageToimageFilter<TImage,TImage> Superclass;
        typedef TImage ImageType;
        typedef typename TImage::IndexType IndexType;
        typedef typename TImage::PixelType PixelType;
        typedef typename TImage::OffsetType OffsetType;
        typedef typename TImage::PointType PointType;
        typedef typename TImage::SizeType SizeType;
        typedef typename TImage::SpacingType SpacingType;
        typedef typename TImage::Pointer ImagePointerType;
        typedef typename TImage::ConstPointer ConstImagePointerType;

        typedef typename itk::ConstNeighborhoodIterator<ImageType> ConstImageNeighborhoodIteratorType;

        typedef TUnaryRegistrationFunction UnaryRegistrationFunctionType;
        typedef typename UnaryRegistrationFunctionType::Pointer UnaryRegistrationFunctionPointerType;
        typedef TPairwiseRegistrationFunction PairwiseRegistrationFunctionType;
        typedef typename PairwiseRegistrationFunctionType::Pointer PairwiseRegistrationFunctionPointerType;
        typedef TUnarySegmentationFunction UnarySegmentationFunctionType;
        typedef typename UnarySegmentationFunctionType::Pointer UnarySegmentationFunctionPointerType;
        typedef TPairwiseSegmentationFunction PairwiseSegmentationFunctionType;
        typedef typename PairwiseSegmentationFunctionType::Pointer PairwiseSegmentationFunctionPointerType;
        typedef TPairwiseSegmentationRegistrationFunction PairwiseSegmentationRegistrationFunctionType;
        typedef typename PairwiseSegmentationRegistrationFunctionType::Pointer PairwiseSegmentationRegistrationFunctionPointerType;
    
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType RegistrationLabelType;
        typedef typename itk::Image<RegistrationLabelType,ImageType::ImageDimension> RegistrationLabelImageType;
        typedef typename RegistrationLabelImageType::Pointer RegistrationLabelImagePointerType;

        typedef int SegmentationLabelType;
        typedef typename itk::Image<SegmentationLabelType,ImageType::ImageDimension> SegmentationLabelImageType;
        typedef typename SegmentationLabelImageType::Pointer SegmentationLabelImagePointerType;
    
    protected:
        //here we store the necessary information per node
        struct nodeInformation {
            //mapping
            std::vector<RegistrationLabelType> indexToSubsampledDisplacementMapping;
            //costs
            std::vector<double> subsampledNodeCosts;
        };
        //std::pair< std::vector<RegistrationLabelType>, std::vector<double> >
        //information for all nodes
        std::vector< nodeInformation > m_nodeMappingInfo;
    public:

        void initGraph(int nGraphNodesPerEdge){
            Superclass::initGraph(nGraphNodesPerEdge);
            m_nodeMappingInfo= std::vector<nodeInformation>(this->m_nDisplacementLabels);
            //computing new subsampled labels and label costs
            for (int r=0;r<this->m_nRegistrationNodes;++r){
                //computing original label costs
                std::vector<double> originalRegistrationCosts(this->m_nDisplacementLabels);
                for (int l=0;l<this->m_nDisplacementLabels;++l){
                    originalRegistrationCosts[l]=Superclass::getUnaryRegistrationPotential(r,l);
                }
                
                // now compute novel mapping and costs and store in m_nodeMappingInfo
                //for the beginning it might be safe to leave m_nRegistrationLabels at its old value. otherwise we'd need to do some copying/overwritng to be safe
                m_nodeMappingInfo[r].indexToSubsampledDisplacementMapping= std::vector< RegistrationLabelType>(this->m_nDisplacementLabels);
                m_nodeMappingInfo[r].subsampledNodeCosts=std::vector<double>(this->m_nDisplacementLabels);
                
                //actually fill these arrays :D
            }
        }

      
        //this is now all interfacing stuff i needed to rewrite because of the optimizer, don't bother :)
        double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            double result=m_nodeMappingInfo[nodeIndex].subsampledNodeCosts[labelIndex];
            return result/this->m_nRegistrationNodes;
        }
     
        double getPairwiseRegistrationPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int labelIndex2){
            IndexType graphIndex1=this->getImageIndexFromCoarseGraphIndex(nodeIndex1);
            RegistrationLabelType l1=m_nodeMappingInfo[nodeIndex1].indexToSubsampledDisplacementMapping[labelIndex1];
            l1=LabelMapperType::scaleDisplacement(l1,this->getDisplacementFactor());
            IndexType graphIndex2=this->getImageIndexFromCoarseGraphIndex(nodeIndex2);
            RegistrationLabelType l2=m_nodeMappingInfo[nodeIndex2].indexToSubsampledDisplacementMapping[labelIndex2];
            l2=LabelMapperType::scaleDisplacement(l2,this->getDisplacementFactor());
            return this->m_pairwiseRegFunction->getPotential(graphIndex1, graphIndex2, l1,l2)/this->m_nRegEdges;
        };
      
        //#define MULTISEGREGNEIGHBORS
        inline double getPairwiseRegSegPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int segmentationLabel){
    
            IndexType imageIndex=this->getImageIndex(nodeIndex2);
            //compute distance between center index and patch index
            double weight=1.0;
            IndexType graphIndex=this->getImageIndexFromCoarseGraphIndex(nodeIndex1);
            double dist=1;
            for (unsigned int d=0;d<this->m_dim;++d){
                dist*=1.0-fabs((1.0*graphIndex[d]-imageIndex[d])/(this->m_gridPixelSpacing[d]));
            }
            weight=dist;
            RegistrationLabelType registrationLabel=m_nodeMappingInfo[nodeIndex1].indexToSubsampledDisplacementMapping[labelIndex1];
            registrationLabel=LabelMapperType::scaleDisplacement(registrationLabel,this->getDisplacementFactor());
            return weight*this->m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel)/this->m_nSegRegEdges;
        }

        
       
   
    }; //GraphModel

}//namespace

#endif /* GRIm_dim_H_ */
