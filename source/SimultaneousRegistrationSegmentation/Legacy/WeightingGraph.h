#include "Log.h"
/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef WGRAPH_H
#define WGRAPH_H
 
#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include <limits>
#include "Graph.h"
#include "Graph-ITKStyle.h"
#include <vnl/vnl_bignum.h>


namespace SRS{
    template<class TImage, 
             class TUnaryRegistrationFunction, 
             class TPairwiseRegistrationFunction, 
             class TUnarySegmentationFunction, 
             class TPairwiseSegmentationFunction,
             class TPairwiseSegmentationRegistrationFunction,
             class TLabelMapper>
    class WeightedGraphModel: 
        public GraphModel<TImage,
                          TUnaryRegistrationFunction,
                          TPairwiseRegistrationFunction,
                          TUnarySegmentationFunction, 
                          TPairwiseSegmentationFunction,
                          TPairwiseSegmentationRegistrationFunction,
                          TLabelMapper>
    {
    public:
        typedef WeightedGraphModel Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
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

        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType RegistrationLabelType;
        typedef typename itk::Image<RegistrationLabelType,ImageType::ImageDimension> RegistrationLabelImageType;
        typedef typename RegistrationLabelImageType::Pointer RegistrationLabelImagePointerType;

        typedef int SegmentationLabelType;
        typedef typename itk::Image<SegmentationLabelType,ImageType::ImageDimension> SegmentationLabelImageType;
        typedef typename SegmentationLabelImageType::Pointer SegmentationLabelImagePointerType;
    
        static const int Dimension=ImageType::ImageDimension;
    private:
        std::vector<double> m_gridNodeWeights;
    public:
        virtual void Init(){
            m_gridNodeWeights=std::vector<double>(this->m_nRegistrationNodes,0.0);
            for (int i=0;i<this->m_nRegistrationNodes;++i){
                IndexType imageIndex=this->getImageIndexFromCoarseGraphIndex(i);
                double weight=this->m_unaryRegFunction->GetOverlapRatio(imageIndex);
                m_gridNodeWeights[i]=weight;
            }
        }
        double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            double result=Superclass::getUnaryRegistrationPotential(nodeIndex,labelIndex);
            return result*m_gridNodeWeights[nodeIndex];
        }

    };//class

}//namespace

#endif//includeguard
