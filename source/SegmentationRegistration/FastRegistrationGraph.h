#include "Log.h"
/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef FRGRAPH_H
#define FRGRAPH_H
 
#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include <limits>
#include "Graph.h"
#include <vnl/vnl_bignum.h>
#include <vnl/vnl_random.h>

// Method of center fitting
#define ISFITCENTER      0
#define CENTERFITHACK    0
#define SCALECENTER      10

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
    class FastRegistrationGraphModel: 
        public GraphModel<TImage,
                          TLabelMapper>
    {
    public:
        typedef FastRegistrationGraphModel Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef GraphModel<TImage,
      
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
    
        static const int Dimension=ImageType::ImageDimension;
    protected:
        int nRegistrationLabels;
        std::vector<std::vector<double> > m_registrationCosts;
    public:

        virtual void Init(){
            m_registrationCosts=std::vector<std::vector<double> >(this->m_nDisplacementLabels);
            for (int l=0;l<this->m_nDisplacementLabels;++l){
                m_registrationCosts[l]=std::vector<double>(this->m_nRegistrationNodes);
                RegistrationLabelType l1=LabelMapperType::getLabel(l);
                l1=LabelMapperType::scaleDisplacement(l1,this->getDisplacementFactor());
                this->m_unaryRegFunction->shiftMovingImage(l1);
                for (int n=0;n<this->m_nRegistrationNodes;++n){
                    IndexType imageIndex=this->getImageIndexFromCoarseGraphIndex(n);
                    m_registrationCosts[l][n]= this->m_unaryRegFunction->getLocalPotential(imageIndex);
                }
            }
        }
        virtual double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            return m_registrationCosts[labelIndex][nodeIndex]/this->m_nRegistrationNodes;
        }
    }; //GraphModel
}//namespace

#endif /* GRIm_dim_H_ */
