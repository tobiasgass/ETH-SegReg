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
#include <itkVectorGradientMagnitudeImageFilter.h>
#include "itkGaussianImage.h"
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
    typedef typename itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationImageIteratorType;
    static const int D=ImageType::ImageDimension;
    typedef TypeGeneral TRWType;
    typedef MRFEnergy<TRWType> MRFType;
    typedef typename TRWType::REAL Real;
    typedef typename MRFType::NodeId NodeType;
    typedef typename MRFType::EdgeId EdgeType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::OffsetType OffsetType;
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
    bool m_anisotropicSmoothing;
    GaussianEstimatorScalarImage<FloatImageType> m_smoothingEstimator;
    ImagePointerType m_mask,m_labelImage;
public:
    MRFRegistrationFuser(){
        m_gridSpacing=8;
        m_pairwiseWeight=1.0;
        m_gridImage=NULL;
        m_count=0;
        m_hardConstraints=false;
        m_alpha=1.0;
        m_anisotropicSmoothing=false;
        m_mask=NULL;
    }
    void setPairwiseWeight(double w){m_pairwiseWeight=w;}
    void setAlpha(double a){m_alpha=a;}
    void setGridSpacing(double s){m_gridSpacing=s;}
    void setHardConstraints(bool b){m_hardConstraints=b;}
    void setAnisoSmoothing(bool a){m_anisotropicSmoothing=a;}
    void setMask(ImagePointerType mask){m_mask=mask;}
    
    //add deformation (with optinal weights)
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
      
        DeformationFieldPointerType def=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage);
        m_lowResDeformations.push_back(def);
        //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
        //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
        ++m_count;
        if (weights.IsNotNull()){
            m_lowResLocalWeights.push_back(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
        }
        if (m_anisotropicSmoothing){
            typedef  typename itk::VectorGradientMagnitudeImageFilter<DeformationFieldType> FilterType;
            typename FilterType::Pointer filter=FilterType::New();
            filter->SetInput(def);
            filter->Update();
            m_smoothingEstimator.addImage(filter->GetOutput());//FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
        }
    }
    void replaceFirstImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        
        DeformationFieldPointerType def=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage);
        m_lowResDeformations[0]=(def);
     
        if (weights.IsNotNull()){
            m_lowResLocalWeights[0]=(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
        }
     
    }

    void replaceLastImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        
        DeformationFieldPointerType def=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage);
        m_lowResDeformations[m_count-1]=(def);
     
        if (weights.IsNotNull()){
            m_lowResLocalWeights[m_count-1]=(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
        }
     
    }
    //build and solve graph
    double finalize(){
        m_lowResResult=ImageUtils<DeformationFieldType>::duplicate(m_lowResDeformations[0]);
        m_labelImage=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(m_gridImage);

    }
    double solve(){
        TRWType::REAL energy=-1, lowerBound=-1;

        
        int nRegLabels=m_count;
        bool useAuxLabel=false;
        if (useAuxLabel){
            nRegLabels++;
            LOG<<"Adding auxiliary lables to avoid problems with over-constrained hard constraints for positive jacobians"<<endl;
        }
        //build MRF
        FloatImagePointerType anisoSmoothingWeights;
        if (m_anisotropicSmoothing){
            m_smoothingEstimator.finalize();
            anisoSmoothingWeights=m_smoothingEstimator.getMean();
        }
            
        MRFType * m_optimizer;
        m_optimizer= new MRFType(TRWType::GlobalSize());
        TRWType::REAL D1[nRegLabels];
        //
        SizeType size=m_gridImage->GetLargestPossibleRegion().GetSize();
        int nRegNodes=m_gridImage->GetLargestPossibleRegion().GetNumberOfPixels();
        vector<NodeType> regNodes(nRegNodes,NULL);
        
        if (m_mask.IsNull()){
            m_mask=FilterUtils<FloatImageType,ImageType>::createEmpty(m_gridImage);
            m_mask->FillBuffer(1);
        }else{
            if (m_mask->GetLargestPossibleRegion().GetSize()!=m_gridImage->GetLargestPossibleRegion().GetSize())
                m_mask=FilterUtils<ImageType>::NNResample(m_mask,1.0/m_gridSpacing,false);
        }
        
        
        //iterate coarse grid for unaries
        FloatImageIteratorType gridIt(m_gridImage,m_gridImage->GetLargestPossibleRegion());
        ImageIteratorType * maskIt;
        maskIt=new ImageIteratorType(m_mask,m_mask->GetLargestPossibleRegion());
        maskIt->GoToBegin();

        bool buff;
        gridIt.GoToBegin();
        int countInside=0;
        for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
            
            bool insideMask=maskIt->Get()>0;
            ++(*maskIt);
            if (!insideMask)
                continue;
        
            ++countInside;
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
        LOGV(1)<<VAR(countInside)<<endl;
     
        //iterate coarse grid for pairwises
        gridIt.GoToBegin();
        if (m_mask.IsNotNull()){
            maskIt->GoToBegin();
        }
        TRWType::REAL Vreg[nRegLabels*nRegLabels];
        for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
            bool insideMask=maskIt->Get()>0;
            ++(*maskIt);
            if (!insideMask)
                continue;
            

            IndexType idx=gridIt.GetIndex();
            //LOGV(1)<<VAR(idx)<<endl;
            std::vector<DeformationType> displacements(nRegLabels);
            for (int l1=0;l1<nRegLabels;++l1){
                if (l1<m_lowResLocalWeights.size()){
                    displacements[l1]=m_lowResDeformations[l1]->GetPixel(idx);
                }
            }
            PointType point,neighborPoint;
            m_gridImage->TransformIndexToPhysicalPoint(idx,point);
            d=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
            double normalizer=-1.0;
            if (anisoSmoothingWeights.IsNotNull()){
                normalizer=anisoSmoothingWeights->GetPixel(idx);
            }
            
            for (int i=0;i<D;++i){
                OffsetType off;
                off.Fill(0);
                off[i]=1;
                IndexType neighborIndex=idx+off;
                //LOGV(2)<<m_mask->GetLargestPossibleRegion().GetSize()<<" "<<VAR(neighborIndex)<<" "<<(m_mask->GetPixel(neighborIndex)>0)<<" "<<m_mask->GetPixel(neighborIndex)<<" "<<m_mask->GetPixel(idx)<<" "<<VAR((m_mask->GetPixel(idx)>0))<<endl;
                if (!m_mask->GetPixel(neighborIndex)>0){
                    //skip edges where neighbor pixel is outside mask, if mask is available
                    continue;
                }
                if (m_gridImage->GetLargestPossibleRegion().IsInside(neighborIndex)){
                    //LOGV(1)<<VAR(neighborIndex)<<endl;

                    for (int l1=0;l1<nRegLabels;++l1){
                        for (int l2=0;l2<nRegLabels;++l2){
                            double weight=0.0;
                            DeformationType neighborDisplacement=m_lowResDeformations[l2]->GetPixel(neighborIndex);
                            m_gridImage->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);
                            double distanceNormalizer=(point-neighborPoint).GetNorm();
                            DeformationType displacementDifference=(displacements[l1]-neighborDisplacement);
                            if (l1<m_count && l2<m_count){
                                
                         
                                if (m_hardConstraints){
                                    //fessler penalty 
                                    bool checkFolding=false;
                                    double k=0.0;//1.0/D+0.0000001;
                                    double K=0.0;//0.5;
                                    for (int d2=0;d2<D;++d2){
                                        double dispDiff=-displacementDifference[d2] ;
                                        if ( dispDiff < -1.0*k*m_gridSpacings[d2] ){
                                            double pen=(dispDiff + 1.0*k*m_gridSpacings[d2] );
                                            weight+=0.5*pen*pen;
                                        }
                                        if (dispDiff > K*m_gridSpacings[d2]){
                                            double pen=(displacementDifference[d2] - 1.0*K*m_gridSpacings[d2] );
                                            weight+=fabs(0.5*pen);
                                        }
                                    }
                                }
                                else{
                                    //weight=(displacementDifference-(point-neighborPoint)).GetSquaredNorm();///(distanceNormalizer)) ;//+ (m_alpha)*(l1!=l2);
                                    weight=(displacementDifference.GetSquaredNorm()/(distanceNormalizer));//+ (m_alpha)*(l1!=l2);
                                    //for (int d2=0;d2<D;++d2){
                                    //weight+=fabs(displacementDifference[d2])/sqrt(distanceNormalizer);
                                    //}

                                    //student-t; fusion flow paper
                                    //weight=m_pairwiseWeight*(log(1.0+1.0/(2*pow(m_alpha,2.0))*displacementDifference.GetSquaredNorm()/(distanceNormalizer*distanceNormalizer)));
                                    //weight=m_pairwiseWeight*(l1!=l2);
                                }
                         
                            }else if (l1<m_count || l2<m_count){
                                weight=0;
                            }else{
                                //do not allow aux labels next to each other? or should one?
                                weight=0;//1000000;
                            }
                            if (normalizer>0){
                                weight=(weight*distanceNormalizer- (normalizer*normalizer));
                                weight*=weight;
                                         
                            }
                            Vreg[l1+l2*nRegLabels]=m_pairwiseWeight*weight;
                           
                        }
                    }
                    int d2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,buff);
                    m_optimizer->AddEdge(regNodes[d], regNodes[d2], TRWType::EdgeData(TRWType::GENERAL,Vreg));
                }         
            }
        }

        //solve MRF
        MRFEnergy<TRWType>::Options options;
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

          
        DeformationImageIteratorType resIt(m_lowResResult, m_lowResResult->GetLargestPossibleRegion());
        resIt.GoToBegin();
        int countAuxLabel=0;
        maskIt->GoToBegin();
        for (;!resIt.IsAtEnd();++resIt){
            IndexType idx=resIt.GetIndex();
            int linearIndex=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
            bool insideMask=true;
            insideMask=maskIt->Get()>0;
            ++(*maskIt);
            int label;
            if (insideMask){
                label=m_optimizer->GetSolution(regNodes[linearIndex]);
                //resIt.Set(m_lowResDeformations[label]->GetPixel(idx));
                m_lowResResult->SetPixel(idx,m_lowResDeformations[label]->GetPixel(idx));
                m_labelImage->SetPixel(idx,label);
            }
        }
        delete maskIt;
      
     
        //m_result=TransfUtils<FloatImageType>::linearInterpolateDeformationField(m_lowResResult,m_highResGridImage);
        //m_result=TransfUtils<FloatImageType>::computeDeformationFieldFromBSplineTransform(m_lowResResult,m_highResGridImage);
        delete m_optimizer;
        return energy;
    }

    DeformationFieldPointerType getMean(){return    m_result=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(m_lowResResult,m_highResGridImage);return m_result;}
    ImagePointerType getLabelImage(){return m_labelImage;}
    DeformationFieldPointerType getVariance(){return NULL;}
    DeformationFieldPointerType getStdDev(){return NULL;}
    double getRelativeLB(){return m_relativeLB;}
    FloatImagePointerType getLikelihood(DeformationFieldPointerType img, double s=1.0){
        FloatImagePointerType result=TransfUtils<FloatImageType>::createEmptyFloat(img);
        return result;
    }

    double solveUntilPosJacDet(int maxIter,double increaseSmoothing,bool useMask){
        double energy;
        double minJac=-1;
        finalize();
        double mmJac=0;
        int iter=0;
        for (;iter<maxIter;++iter){
            //FloatImagePointerType jacDets=TransfUtils<ImageType>::getJacDets(m_lowResResult);
            FloatImagePointerType jacDets=TransfUtils<ImageType>::getJacDets(getMean());
            minJac=FilterUtils<FloatImageType>::getMin(jacDets);
        
            
            LOGV(2)<<VAR(iter)<<" "<<VAR(minJac)<<endl;
            double fessler=fesslerTest();
            LOGV(2)<< VAR(( (fessler<0) == minJac<0 ))<<" "<<VAR(fessler)<<" "<<VAR(minJac)<<endl;
            double fac=1.0;
            if (minJac<0.0) fac=-1.0;
            if (minJac>mmJac)
                break;
            if (useMask){
                ImagePointerType mask= FilterUtils<FloatImageType,ImageType>::cast(FilterUtils<FloatImageType>::binaryThresholdingHigh(jacDets,mmJac));
                ImageUtils<ImageType>::writeImage("mask.nii",mask);
                LOGV(2)<<"dilating mask with a ball of "<<min(100.0,fac*20.0*minJac)<<" px."<<endl;
                mask=FilterUtils<ImageType>::dilation(mask,max(1.0*m_gridSpacings[0],min(100.0,fac*20.0*minJac)));
                ImageUtils<ImageType>::writeImage("mask-dilated.nii",mask);
                setMask(mask);
                                                        
            }
            energy=solve();
            m_pairwiseWeight*=increaseSmoothing;
                    
        }
        LOGV(1)<<"SSR iterations :"<<iter<<endl;
        return energy;

    }

    double fesslerTest(){
        
        double minDiff=100000000;
        DeformationImageIteratorType gridIt(m_lowResResult, m_lowResResult->GetLargestPossibleRegion());

        for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
          
            IndexType idx=gridIt.GetIndex();
            
            PointType point,neighborPoint;
            m_gridImage->TransformIndexToPhysicalPoint(idx,point);
            double normalizer=-1.0;
            DeformationType disp=gridIt.Get();
            for (int i=0;i<D;++i){
                OffsetType off;
                off.Fill(0);
                off[i]=1;
                IndexType neighborIndex=idx+off;
                if (m_gridImage->GetLargestPossibleRegion().IsInside(neighborIndex)){
                    DeformationType neighborDisplacement=m_lowResResult->GetPixel(neighborIndex);
                    m_gridImage->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);
                    double distanceNormalizer=(point-neighborPoint).GetNorm();
                    DeformationType displacementDifference=(disp-neighborDisplacement);
                    //fessler penalty 
                    bool checkFolding=false;
                    double k=1.0/D+0.0000001;
                    double K=0.0;//0.5;
                    for (int d2=0;d2<D;++d2){
                        double dispDiff=-displacementDifference[d2] ;
                        if ( dispDiff < -1.0*k*m_gridSpacings[d2] ){
                            double pen=(-dispDiff + 1.0*k*m_gridSpacings[d2] );
                            if (pen<minDiff)minDiff=pen;
                        }
                    }
                    
                }         
            }
        }

        return minDiff;
    }
    
};//class

