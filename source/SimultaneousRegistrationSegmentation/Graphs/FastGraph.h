#pragma once

#include "Graph.h"
#include "Potential-Registration-Unary.h"
#include "Potential-Registration-Pairwise.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Segmentation-Pairwise.h"
#include "Potential-Coherence-Pairwise.h"
#include "BaseLabel.h"
#include "Log.h"

namespace SRS{

    ///Graph which allows for registration potential functions with caching
    template<class TImage,
             class TUnaryRegistrationFunction=FastUnaryPotentialRegistrationNCC<TImage>,
             class TPairwiseRegistrationFunction= PairwisePotentialRegistration<TImage>,
             class TUnarySegmentationFunction=UnaryPotentialSegmentation<TImage>,
             class TPairwiseSegmentationFunction=PairwisePotentialSegmentation<TImage>,
             class TPairwiseCoherenceFunction=PairwisePotentialCoherence<TImage> >
    class FastGraphModel: public GraphModel<TImage,TUnaryRegistrationFunction,TPairwiseRegistrationFunction,TUnarySegmentationFunction,TPairwiseSegmentationFunction,TPairwiseCoherenceFunction>
    {
    public:
        typedef FastGraphModel Self;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
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
        typedef typename TransfUtils<ImageType>::DisplacementType RegistrationLabelType;
    public:
         void Init(){
            //#define moarcaching
            this->m_unaryRegFunction->setCoarseImage(this->m_coarseGraphImage);
            TIME(this->m_unaryRegFunction->initCaching());
            
#ifdef moarcaching
            std::vector<RegistrationLabelType> displacementList(this->m_nDisplacementLabels);
            for (int n=0;n<this->m_nDisplacementLabels;++n){
                LOGV(25)<<"Caching unary registration potentials for label "<<n<<endl;
                displacementList[n]=this->m_labelMapper->scaleDisplacement(this->m_labelMapper->getLabel(n),this->getDisplacementFactor());
            }
            this->m_unaryRegFunction->setDisplacements(displacementList);
            this->m_unaryRegFunction->compute();
#endif
            
        }
         void cacheRegistrationPotentials(int labelIndex){
#ifndef moarcaching
            LOGV(25)<<"Caching unary registration function for label " << labelIndex<<endl;
            
            this->m_unaryRegFunction->cachePotentials(this->m_labelMapper->scaleDisplacement(this->m_labelMapper->getLabel(labelIndex),this->getDisplacementFactor()));
#endif
        }
        inline double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            IndexType index=this->getGraphIndex(nodeIndex);
            LOGV(90)<<VAR(index);
#ifdef moarcaching
            double result=  this->m_unaryRegFunction->getPotential(index,labelIndex);//this->m_nRegistrationNodes;
#else
            double result=  this->m_unaryRegFunction->getPotential(index);//this->m_nRegistrationNodes;
#endif

            if (this->m_normalizePotentials) result/=this->m_nRegistrationNodes;
            return result;

        }


    };
}
