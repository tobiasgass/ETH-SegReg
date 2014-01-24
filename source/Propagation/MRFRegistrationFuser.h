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
    typedef typename ImageType::SpacingType SpacingType;
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
    double m_relativeLB;
    SpacingType m_gridSpacings;
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
            m_gridSpacings=m_gridImage->GetSpacing();
        }            
        if (weights.IsNotNull()){
            m_lowResLocalWeights.push_back(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
        }
        //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
        m_lowResDeformations.push_back(TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage));
        //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
        ++m_count;
    }
    double finalize(ImagePointerType & labelImage=NULL){
        int nRegLabels=m_count;
        bool useAuxLabel=false;
        if (useAuxLabel){
            nRegLabels++;
            LOG<<"Adding auxiliary lables to avoid problems with over-constrained hard constraints for positive jacobians"<<endl;
        }
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
                if (l1<m_count){
                    //D1[l1]=-log(m_lowResLocalWeights[l1]->GetPixel(idx));
                    D1[l1]=1.0-(m_lowResLocalWeights[l1]->GetPixel(idx));
                    LOGV(7)<<l1<<" "<<VAR(D1[l1])<<" "<<m_lowResLocalWeights[l1]->GetPixel(idx)<<endl;
                }else{
                    //penalty for aux label
                    D1[l1]=2;
                }
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
                if (l1<m_lowResLocalWeights.size()){
                    displacements[l1]=m_lowResDeformations[l1]->GetPixel(idx);
                }
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
                             double weight=0.0;
                             if (l1<m_count && l2<m_count){
                                 DeformationType neighborDisplacement=m_lowResDeformations[l2]->GetPixel(neighborIndex);
                                 m_gridImage->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);
                                 double distanceNormalizer=(point-neighborPoint).GetNorm();
                                 DeformationType displacementDifference=(displacements[l1]-neighborDisplacement);
                                 bool checkFolding=false;
                                 double k=1.0/D+0.0000001;
                                 double K=0.5;


#if 1

                                 
                                 for (int d2=0;d2<D;++d2){
                                     if (d2!=i){
                                         checkFolding = checkFolding || fabs(displacementDifference[d2])>m_gridSpacings[d2]*k;
                                         //checkFolding = checkFolding || (fabs(displacementDifference[d2]) > fabs(point[i]-neighborPoint[i]));
                                     } else{
                                         //checkFolding = checkFolding || displacementDifference[d2] +point[i]-neighborPoint[i] >= 0;
                                         //checkFolding = checkFolding || (fabs(displacementDifference[d2]) > 2*fabs(point[i]-neighborPoint[i]));
                                         checkFolding = checkFolding 
                                             || (-displacementDifference[d2])  < -1.0*k*m_gridSpacings[d2]
                                             ||  (-displacementDifference[d2]) > K*m_gridSpacings[d2];
                                         
                                     }
                                     
                                 }
                                 
                                 
                                 
                                 
                                 if (m_hardConstraints && checkFolding ){
                                     //folding!
                                     // (p1+d1)-(p2+d2) > 0
                                     LOGV(6)<<VAR(point[i])<<" "<<VAR(neighborPoint[i])<<" "<<VAR(displacements[l1][i])<< " " <<VAR(neighborDisplacement[i])<<endl;
                                     weight=10000000;
                                     //weight=m_pairwiseWeight*(log(1.0+1.0/(2*pow(m_alpha,2.0))*displacementDifference.GetSquaredNorm()/(distanceNormalizer*distanceNormalizer)));
                                     
                                 }else{
                                     
                                     weight=m_pairwiseWeight*(displacementDifference.GetSquaredNorm()/(distanceNormalizer)) ;//+ (m_alpha)*(l1!=l2);
                                     
                                     //student-t; fusion flow paper
                                     //weight=m_pairwiseWeight*(log(1.0+1.0/(2*pow(m_alpha,2.0))*displacementDifference.GetSquaredNorm()/(distanceNormalizer*distanceNormalizer)));
                                     //weight=m_pairwiseWeight*(l1!=l2);
                                 }
                         
#else
                             
                                 for (int d2=0;d2<D;++d2){
                                     double dispDiff=-displacementDifference[d2] ;
                                     if ( dispDiff < -1.0*k*m_gridSpacings[d2] ){
                                         double pen=(dispDiff + 1.0*k*m_gridSpacings[d2] );
                                         weight+=m_pairwiseWeight*0.5*pen*pen;
                                     }
                                     if (dispDiff > K*m_gridSpacings[d2]){
                                         double pen=(displacementDifference[d2] - 1.0*K*m_gridSpacings[d2] );
                                         weight+=m_pairwiseWeight*0.5*pen*pen;
                                     }
                                 }
                                 
                                 
                                 
#endif
                             }else if (l1<m_count || l2<m_count){
                                 weight=0;
                             }else{
                                 //do not allow aux labels next to each other? or should one?
                                 weight=0;//1000000;
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
        options.m_iterMax = 1000; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=2;
        options.m_eps=1e-7;
        clock_t opt_start=clock();
        m_optimizer->Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        float t = (float) ((double)(finish - opt_start) / CLOCKS_PER_SEC);
        LOGV(2)<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound <<std::endl;

        m_relativeLB=lowerBound/energy;
        //get output and upsample

        m_lowResResult=ImageUtils<DeformationFieldType>::duplicate(m_lowResDeformations[0]);
        labelImage=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(m_gridImage);
        DeformationImageIteratorType resIt(m_lowResResult, m_lowResResult->GetLargestPossibleRegion());
        resIt.GoToBegin();
        int countAuxLabel=0;
        for (;!resIt.IsAtEnd();++resIt){
            IndexType idx=resIt.GetIndex();
            int linearIndex=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
            int label=m_optimizer->GetSolution(regNodes[linearIndex]);
            if (!useAuxLabel || label<m_count){
                resIt.Set(m_lowResDeformations[label]->GetPixel(idx));
            } else if (useAuxLabel){
                countAuxLabel++;
            }
            labelImage->SetPixel(resIt.GetIndex(),label);
        }
        if (useAuxLabel){
            LOGV(2)<<VAR(countAuxLabel)<<std::endl;

            //compute displacements for auxiliary labels by interpolation
            while (countAuxLabel>0){
                countAuxLabel=0;
                 resIt.GoToBegin();
                 for (;!resIt.IsAtEnd();++resIt){
                     IndexType idx=resIt.GetIndex();
                     int linearIndex=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
                     int label=m_optimizer->GetSolution(regNodes[linearIndex]);
                     //check auxLabel
                     if (label == nRegLabels-1){
                         //check if solved neighbors exist and accumulate their displacement
                         DeformationType disp;
                         disp.Fill(0.0);
                         int countValidNeighb=0;
                         //forward and backwards neighbor
                         for (int dir=-1;dir<2;dir+=2){
                             //axes
                             for (int i=0;i<D;++i){
                                 OffsetType off;
                                 off.Fill(0);
                                 off[i]=dir;
                                 IndexType neighborIndex=idx+off;
                                 //check inside
                                 if(m_gridImage->GetLargestPossibleRegion().IsInside(neighborIndex)){
                                     int linearIndexN=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,buff);
                                     int labelN=m_optimizer->GetSolution(regNodes[linearIndexN]);
                                     if (labelN<m_count){
                                         //neighbor is not labelled with aux label
                                         ++countValidNeighb;
                                         disp+=m_lowResDeformations[labelN]->GetPixel(neighborIndex);
                                     }
                                 }
                             }
                         }
                         if (countValidNeighb>0){
                             resIt.Set(disp*1.0/countValidNeighb);
                             labelImage->SetPixel(resIt.GetIndex(),-1);
                         }else{
                             countAuxLabel++;
                         }
                             
                     }//aux label found
                 }//resIt
                 LOGV(2)<<VAR(countAuxLabel)<<std::endl;
            }//while
        }//if hardCounstraints
        
        m_result=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(m_lowResResult,m_highResGridImage);
        //m_result=TransfUtils<FloatImageType>::linearInterpolateDeformationField(m_lowResResult,m_highResGridImage);
        //m_result=TransfUtils<FloatImageType>::computeDeformationFieldFromBSplineTransform(m_lowResResult,m_highResGridImage);
        delete m_optimizer;
        return energy;
    }

    DeformationFieldPointerType getMean(){return m_result;}
    DeformationFieldPointerType getVariance(){return NULL;}
    DeformationFieldPointerType getStdDev(){return NULL;}
    double getRelativeLB(){return m_relativeLB;}
    FloatImagePointerType getLikelihood(DeformationFieldPointerType img, double s=1.0){
        FloatImagePointerType result=TransfUtils<FloatImageType>::createEmptyFloat(img);
        return result;
    }
    
};//class

