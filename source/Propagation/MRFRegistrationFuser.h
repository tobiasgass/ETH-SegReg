#pragma once


#include <limits.h>
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include "itkGaussianImage.h"
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"

template<class ImageType>
class MRFRegistrationFuser : public GaussianEstimatorVectorImage<ImageType>{

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename TransfUtils<ImageType>::DisplacementType DeformationType;
    typedef typename TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationImageIteratorType;
    static const int D=ImageType::ImageDimension;
    typedef TypeGeneral TRWType;
    typedef MRFEnergy<TRWType> MRFType;
    typedef typename TRWType::REAL Real;
    typedef typename MRFType::NodeId NodeType;
    typedef typename MRFType::EdgeId EdgeType;
private:
    std::vector<DeformationFieldPointerType> m_lowResDeformations;
    std::vector<FloatImagePointerType> m_lowResLocalWeights;
    double m_gridSpacing,m_pairwiseWeight,m_alpha;
    DeformationFieldPointerType m_lowResResult, m_result;
    FloatImagePointerType m_gridImage,m_highResGridImage;
    int m_count;
    bool m_hardConstraints;
public:
    MRFRegistrationFuser(){
        m_gridSpacing=8;
        m_pairwiseWeight=1.0;
        m_gridImage=NULL;
        m_count=0;
        m_hardConstraints=false;
        m_alpha=1.0;
    }
    void setPairwiseWeight(double w){m_pairwiseWeight=w;}
    void setAlpha(double a){m_alpha=a;}
    void setGridSpacing(double s){m_gridSpacing=s;}
    void setHardConstraints(bool b){m_hardConstraints=b;}
    void addImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        if (!m_gridImage.IsNotNull()){
            if (m_gridSpacing<=0){
                LOG<<VAR(m_gridSpacing)<<endl;
                exit(0);
            }
            //initialize
            m_highResGridImage=TransfUtils<FloatImageType>::createEmptyImage(img);
            m_gridImage=FilterUtils<FloatImageType>::NNResample(m_highResGridImage,
                                                                1.0/m_gridSpacing,
                                                                false);
        }            
        if (weights.IsNotNull()){
            m_lowResLocalWeights.push_back(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
        }
        //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
        m_lowResDeformations.push_back(TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage));
        ++m_count;
    }
    double finalize(ImagePointerType & labelImage=NULL){
        int nRegLabels=m_count;
         
        //build MRF
       
        MRFType * m_optimizer;
        m_optimizer= new MRFType(TRWType::GlobalSize());
        TRWType::REAL D1[nRegLabels];
        //
        SizeType size=m_gridImage->GetLargestPossibleRegion().GetSize();
        int nRegNodes=m_gridImage->GetLargestPossibleRegion().GetNumberOfPixels();
        vector<NodeType> regNodes(nRegNodes,NULL);

        //iterate coarse grid for unaries
        FloatImageIteratorType gridIt(m_gridImage,m_gridImage->GetLargestPossibleRegion());
        bool buff;
        gridIt.GoToBegin();
        for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
            
            IndexType idx=gridIt.GetIndex();
            for (int l1=0;l1<nRegLabels;++l1) {
                if (m_lowResLocalWeights.size()==m_count){
                    //D1[l1]=-log(m_lowResLocalWeights[l1]->GetPixel(idx));
                    D1[l1]=1.0-(m_lowResLocalWeights[l1]->GetPixel(idx));
                    LOGV(3)<<l1<<" "<<VAR(D1[l1])<<" "<<m_lowResLocalWeights[l1]->GetPixel(idx)<<endl;
                }else
                    D1[l1]=1;
            }

            d=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
            regNodes[d] = 
                m_optimizer->AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
        }
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::OffsetType OffsetType;
        //iterate coarse grid for pairwises
        gridIt.GoToBegin();
        TRWType::REAL Vreg[nRegLabels*nRegLabels];
        for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
            IndexType idx=gridIt.GetIndex();
            std::vector<DeformationType> displacements(nRegLabels);
            for (int l1=0;l1<nRegLabels;++l1){
                displacements[l1]=m_lowResDeformations[l1]->GetPixel(idx);
            }
            PointType point,neighborPoint;
            m_gridImage->TransformIndexToPhysicalPoint(idx,point);
            d=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
            for (int i=0;i<D;++i){
                 OffsetType off;
                 off.Fill(0);
                 off[i]=1;
                 IndexType neighborIndex=idx+off;
                 if (m_gridImage->GetLargestPossibleRegion().IsInside(neighborIndex)){
                     for (int l1=0;l1<nRegLabels;++l1){
                         for (int l2=0;l2<nRegLabels;++l2){
                             DeformationType neighborDisplacement=m_lowResDeformations[l2]->GetPixel(neighborIndex);
                             m_gridImage->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);
                             double distanceNormalizer=(point-neighborPoint).GetNorm();
                             DeformationType displacementDifference=(displacements[l1]-neighborDisplacement);
                             double weight;
                             bool checkFolding=point[i]-neighborPoint[i] + displacementDifference[i] >= 0;
                             if (m_hardConstraints && checkFolding ){
                                 //folding!
                                 // (p1+d1)-(p2+d2) > 0
                                 LOGV(3)<<VAR(point[i])<<" "<<VAR(neighborPoint[i])<<" "<<VAR(displacements[l1][i])<< " " <<VAR(neighborDisplacement[i])<<endl;
                                 weight=100000;
                             }else{
                                 weight=m_pairwiseWeight*(displacementDifference.GetSquaredNorm()/distanceNormalizer) + (m_alpha)*(l1!=l2);
                                 //weight=m_pairwiseWeight*(l1!=l2);
                             }
                             Vreg[l1+l2*nRegLabels]=weight;
                         }
                     }
                     int d2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,buff);
                     m_optimizer->AddEdge(regNodes[d], regNodes[d2], TRWType::EdgeData(TRWType::GENERAL,Vreg));
                 }         
             }
        }

        //solve MRF
        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy=-1, lowerBound=-1;
        options.m_iterMax = 50; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=2;
        options.m_eps=-1;
        clock_t opt_start=clock();
        m_optimizer->Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        float t = (float) ((double)(finish - opt_start) / CLOCKS_PER_SEC);
        LOGV(2)<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound <<std::endl;

        //get output and upsample

        m_lowResResult=ImageUtils<DeformationFieldType>::duplicate(m_lowResDeformations[0]);
        labelImage=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(m_gridImage);
        DeformationImageIteratorType resIt(m_lowResResult, m_lowResResult->GetLargestPossibleRegion());
        resIt.GoToBegin();
        for (;!resIt.IsAtEnd();++resIt){
            IndexType idx=resIt.GetIndex();
            int linearIndex=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
            int label=m_optimizer->GetSolution(regNodes[linearIndex]);
            resIt.Set(m_lowResDeformations[label]->GetPixel(idx));
            labelImage->SetPixel(resIt.GetIndex(),label);
        }
        
        m_result=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(m_lowResResult,m_highResGridImage);
        //m_result=TransfUtils<FloatImageType>::computeDeformationFieldFromBSplineTransform(m_lowResResult,m_highResGridImage);
        delete m_optimizer;
        return energy;
    }

    DeformationFieldPointerType getMean(){return m_result;}
    DeformationFieldPointerType getVariance(){return NULL;}
    DeformationFieldPointerType getStdDev(){return NULL;}

    FloatImagePointerType getLikelihood(DeformationFieldPointerType img, double s=1.0){
        FloatImagePointerType result=TransfUtils<FloatImageType>::createEmptyFloat(img);
        return result;
    }
    
};//class

