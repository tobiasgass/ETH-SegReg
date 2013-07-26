#pragma once

#include <stdio.h>
#include <iostream>
#include "GCoptimization.h"
#include "argstream.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "bgraph.h"
#include <sstream>
#include "argstream.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "itkConstNeighborhoodIterator.h"
#include "itkDisplacementFieldTransform.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkImageAlgorithm.h>


using namespace std;

template <class ImageType>
class SegmentationFusion{
public:
    typedef typename ImageType::PixelType PixelType;
    static const unsigned int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
    typedef typename  ImageType::IndexType IndexType;
    typedef typename  ImageType::PointType PointType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename  ImageType::SizeType SizeType;
    typedef typename  ImageType::ConstPointer ImageConstPointerType;
    typedef typename  ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename  FloatImageType::Pointer FloatImagePointerType;

    typedef  TransfUtils<ImageType,double> TransfUtilsType;

    typedef typename  TransfUtilsType::DisplacementType DisplacementType; 
    typedef typename  TransfUtilsType::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename  itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename  itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
    typedef typename  itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
    typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
    typedef typename  ImageNeighborhoodIteratorType::RadiusType RadiusType;

    typedef itk::VectorImage<float,D> ProbabilisticVectorImageType;
    typedef typename ProbabilisticVectorImageType::PixelType ProbPixelType;
    typedef typename ProbabilisticVectorImageType::Pointer ProbabilisticVectorImagePointerType;
    typedef typename itk::ImageRegionIterator<ProbabilisticVectorImageType> ProbImageIteratorType;


    typedef std::vector<std::pair<string,ImagePointerType> > ImageListType;

private:
    int m_count;
    std::map<int, int> m_labelMapping,m_inverseLabelMapping;
    int m_nLabels;
    ProbabilisticVectorImagePointerType m_accumulator,m_normalizer;
    
public:
    SegmentationFusion(){
        m_count=0;
    }
  
   
    ImagePointerType getFusedSegmentation(){
        if (m_count==0){
            return NULL;
        }
        ImagePointerType result=ImageType::New();
        result->SetDirection(m_accumulator->GetDirection());
        result->SetOrigin(m_accumulator->GetOrigin());
        result->SetSpacing(m_accumulator->GetSpacing());
        result->SetRegions(m_accumulator->GetLargestPossibleRegion());
        result->Allocate();
        ImageIteratorType segIt(result,result->GetLargestPossibleRegion());
        ProbImageIteratorType accIt(m_accumulator,m_accumulator->GetLargestPossibleRegion());
        accIt.GoToBegin();
        for (segIt.GoToBegin();!segIt.IsAtEnd();++segIt,++accIt){
            ProbPixelType px=accIt.Get();
            int bestLabel=-1;
            double bestScore=-1;
            for (int n=0;n<m_nLabels;++n){
                if (px[n]>bestScore){
                    bestScore=px[n];
                    bestLabel=n;
                }
            }
            segIt.Set(m_inverseLabelMapping[bestLabel]);
        }
        return result;
    }
    ProbabilisticVectorImagePointerType segmentationToProbabilisticVector(ImagePointerType seg){
        ProbabilisticVectorImagePointerType result=ProbabilisticVectorImageType::New();
        result->SetNumberOfComponentsPerPixel(m_nLabels);
        result->SetDirection(seg->GetDirection());
        result->SetOrigin(seg->GetOrigin());
        result->SetSpacing(seg->GetSpacing());
        result->SetRegions(seg->GetLargestPossibleRegion());
        result->Allocate();
        ProbImageIteratorType probIt(result,result->GetLargestPossibleRegion());
        ImageIteratorType segIt(seg,seg->GetLargestPossibleRegion());
        segIt.GoToBegin();
        for (probIt.GoToBegin();!probIt.IsAtEnd();++segIt,++probIt){
            int label=m_labelMapping[segIt.Get()];
            ProbPixelType px(m_nLabels);
            px.Fill(0);
            px[label]=1;
            probIt.Set(px);
        }
        return result;

    }
    void addImage(ImagePointerType seg, FloatImagePointerType weights=NULL){
        if (m_count==0){
            computeLabelMapping(seg);
        }
        ProbabilisticVectorImagePointerType probSeg=segmentationToProbabilisticVector(seg);
        if (m_count==0){
            m_accumulator=ImageUtils<ProbabilisticVectorImageType>::duplicate(probSeg);
            m_normalizer=ImageUtils<ProbabilisticVectorImageType>::duplicate(probSeg);
            ProbPixelType px(m_nLabels);
            px.Fill(0.0);
            m_accumulator->FillBuffer(px);
            m_normalizer->FillBuffer(px);
        }
        ProbImageIteratorType probIt(probSeg,probSeg->GetLargestPossibleRegion());
        ProbImageIteratorType accIt(m_accumulator,m_accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType normIt(m_normalizer,m_normalizer->GetLargestPossibleRegion());
        FloatImageIteratorType weightIt;
        if (weights.IsNotNull()){
            weightIt=FloatImageIteratorType(weights,weights->GetLargestPossibleRegion());
            weightIt.GoToBegin();
        }
        probIt.GoToBegin();accIt.GoToBegin();normIt.GoToBegin();
        for (;!probIt.IsAtEnd();++probIt,++accIt,++normIt){
            ProbPixelType px=probIt.Get();
            double weight=1.0;
            if (weights.IsNotNull()){
                weight=weightIt.Get();
                ++weightIt;
            }
            ProbPixelType accPx=accIt.Get();
            ProbPixelType normPx=normIt.Get();
            for (int n=0;n<m_nLabels;++n){
                accPx[n]+=weight*px[n];
                normPx[n]+=weight;
            }
            accIt.Set(accPx);
            normIt.Set(normPx);
        }
        ++m_count;
    }
    void computeLabelMapping(ImagePointerType seg){
        ImageIteratorType segIt(seg,seg->GetLargestPossibleRegion());
        segIt.GoToBegin();
        m_nLabels=0;
        for(;!segIt.IsAtEnd();++segIt){
            int val=segIt.Get();
            if (m_labelMapping.find(val)==m_labelMapping.end()){
                m_labelMapping[val]=m_nLabels;
                LOGV(2)<<"mapping segmentation "<<val<<" to "<<m_nLabels<<endl;
                m_inverseLabelMapping[m_nLabels]=val;
                ++m_nLabels;
            }
        }
        
    }

};
#if 0  
    ProbabilisticVectorImagePointerType segmentationToProbabilisticVector(ImagePointerType img){
        ProbabilisticVectorImagePointerType result=createEmptyProbImageFromImage(img);
        ProbImageIteratorType probIt(result,result->GetLargestPossibleRegion());
        ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            ProbabilisticPixelType p;
            p.Fill(0.0);
            p[int(imgIt.Get())]=1;
            probIt.Set(p);
        }
        return result;
    }

    ImagePointerType probSegmentationToSegmentationLocal( ProbabilisticVectorImagePointerType img){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            float maxProb=-std::numeric_limits<float>::max();
            int maxLabel=0;
            ProbabilisticPixelType p = probIt.Get();
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                if (p[s]>maxProb){
                    maxLabel=s;
                    maxProb=p[s];
                }
            }
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        return result;
    }
    ImagePointerType probSegmentationToProbImageLocal( ProbabilisticVectorImagePointerType img){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            float maxProb=-std::numeric_limits<float>::max();
            int maxLabel=0;
            ProbabilisticPixelType p = probIt.Get();
            double sump=0.0;
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                if (p[s]>maxProb){
                    maxLabel=s;
                    maxProb=p[s];
                }
                sump+=p[s];
            }
            maxProb=p[1]/sump;
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*pow(maxProb,1.0));
            }else{
                imgIt.Set(maxProb*maxLabel);
            }
        }
        return result;
    }

    ImagePointerType probSegmentationToSegmentationGraphcut( ProbabilisticVectorImagePointerType img, double smooth){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        typedef BGraph<float,float,float> MRFType;
        typedef MRFType::node_id NodeType;
        MRFType* optimizer;
        SizeType size=img->GetLargestPossibleRegion().GetSize();
        int nNodes=1;
        for (unsigned int d=0;d<D;++d) {nNodes*=size[d];}
        int nEdges=D*nNodes;
        for (unsigned int d=0;d<D;++d) {nEdges-=size[d];}
        optimizer = new MRFType(nNodes,nEdges);
        optimizer->add_node(nNodes);
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        int i=0;
        for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++i){
            IndexType idx=probIt.GetIndex();
            ProbabilisticPixelType localProbs=probIt.Get();
            ProbabilisticPixelType energies;
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                energies[s]=0;
                for (unsigned int sprime=0;sprime<nSegmentationLabels;++sprime){
                    if (sprime!=s){
                        energies[s]+=localProbs[sprime];
                    }
                }
            }
            LOGV(7)<<VAR(i)<<" "<<VAR(energies)<<endl;
            optimizer->add_tweights(i,energies[0],energies[1]);
            for (unsigned  int d=0;d<D;++d){
                OffsetType off;
                off.Fill(0);
                off[d]+=1;
                IndexType neighborIndex=idx+off;
                bool inside2;
                int withinImageIndex2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,inside2);
                if (inside2){
                    optimizer -> add_edge(i,withinImageIndex2,smooth,smooth);
                }
            }
        }
        optimizer->maxflow();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        i=0;
        for (imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++i){
            int maxLabel=optimizer->what_segment(i)== MRFType::SOURCE ;
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        return result;
    }
    ProbabilisticVectorImagePointerType normalizeProbs(ProbabilisticVectorImagePointerType img){
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        ProbabilisticVectorImagePointerType result=ProbabilisticVectorImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ProbImageIteratorType resultIt(result,img->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++resultIt){
            ProbabilisticPixelType localProbs=probIt.Get();
            double sum=0.0;
            //compute normalizing sum to normalize probabilities(if they're not yet normalized)
            for (int s2=0;s2<nSegmentationLabels;++s2){
                sum+=localProbs[s2];
            }
            for (int s2=0;s2<nSegmentationLabels;++s2){
                localProbs[s2]/=sum;
            }
            resultIt.Set(localProbs);
        }
        return result;
    }
    ProbabilisticVectorImagePointerType normalizeProbsSimple(ProbabilisticVectorImagePointerType img, int nAtlases, int nImages){
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        ProbabilisticVectorImagePointerType result=ProbabilisticVectorImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ProbImageIteratorType resultIt(result,img->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++resultIt){
            ProbabilisticPixelType localProbs=probIt.Get();
          
            for (int s2=0;s2<nSegmentationLabels;++s2){
                localProbs[s2]/=nAtlases*(nImages-1);
            }
            resultIt.Set(localProbs);
        }
        return result;
    }
    ImagePointerType probSegmentationToSegmentationGraphcutMultiLabel( ProbabilisticVectorImagePointerType img, ImagePointerType segImg, double smooth, double sigma){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();

        typedef GCoptimizationGeneralGraph MRFType;
        //todo
        MRFType optimizer(result->GetBufferedRegion().GetNumberOfPixels(),nSegmentationLabels);
       
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
       
        //going to use sparse labels
        //iterate over labels
        for (unsigned int s=0;s<nSegmentationLabels;++s){
            std::vector<GCoptimization::SparseDataCost> costs(result->GetBufferedRegion().GetNumberOfPixels());
            //GCoptimization::SparseDataCost costs[result->GetBufferedRegion().GetNumberOfPixels()];
            int n = 0;
            int i=0;
            for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++i){
                ProbabilisticPixelType localProbs=probIt.Get();
                double prob=localProbs[s];
                double sum=0.0;
                //compute normalizing sum to normalize probabilities(if they're not yet normalized)
                for (int s2=0;s2<nSegmentationLabels;++s2){
                    sum+=localProbs[s2];
                }
                //only add site do label if prob > 0
                if (prob>0){
                    costs[n].site=i;
                    //LOGV(3)<<VAR(prob)<<" "<<VAR(sum)<<endl;
                    costs[n].cost=-log(prob/sum);
                    ++n;
                }
            }
            //resize to actual number of sites with label s
            costs.resize(n);
            optimizer.setDataCost(s,&costs[0],n);
        }
        float *smoothCosts = new float[nSegmentationLabels*nSegmentationLabels];
        for ( int l1 = 0; l1 < nSegmentationLabels; l1++ )
            for (int l2 = 0; l2 < nSegmentationLabels; l2++ )
                smoothCosts[l1+l2*nSegmentationLabels] =  (l1!=l2);
        optimizer.setSmoothCost(smoothCosts);
        int i=0;
        SizeType size=img->GetLargestPossibleRegion().GetSize();

        ImageIteratorType imageIt(segImg,segImg->GetLargestPossibleRegion());
        for (imageIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++i){
            IndexType idx=imageIt.GetIndex();
            double value=imageIt.Get();
            for (unsigned  int d=0;d<D;++d){
                OffsetType off;
                off.Fill(0);
                off[d]+=1;
                IndexType neighborIndex=idx+off;
                bool inside2;
                int withinImageIndex2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,inside2);
                if (inside2){
                    double neighborvalue=segImg->GetPixel(neighborIndex);
                    double weight= exp(-fabs(value-neighborvalue)/sigma);
                    optimizer.setNeighbors(i,withinImageIndex2,smooth*weight);
                }
            }
        }
        LOGV(1)<<"solving graph cut"<<endl;
        optimizer.setVerbosity(mylog.getVerbosity());
        try{
            optimizer.expansion(20);
        }catch (GCException e){
            e.Report();
            exit(-1);
        }
        LOGV(1)<<"done"<<endl;
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        i=0;
        for (imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++i){
            int maxLabel=optimizer.whatLabel(i) ;
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        delete smoothCosts;
        return result;
    }

    void updateProbabilisticSegmentationUniform(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight){
        ProbabilisticVectorImagePointerType deformedIncrement=increment;
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());
        for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight);
        }
    }

    void updateProbabilisticSegmentationGlobalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,MetricType metric ){
       
        double metricWeight=0;
        typedef typename itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
        interpolator->SetInputImage(movingImage);

        typedef typename itk::IdentityTransform<float,D> DTTransformType;
        typename DTTransformType::Pointer transf=DTTransformType::New();
      
#if 0
        switch(metric){
        case NCC:{
            typedef typename itk::NormalizedCorrelationImageToImageMetric<
                ImageType,
                ImageType >    NCCMetricType;
            typename NCCMetricType::Pointer ncc=NCCMetricType::New();
            ncc->SetTransform(transf);
            ncc->SetFixedImage(targetImage);
            ncc->SetMovingImage(movingImage);
            ncc->SetInterpolator(interpolator);
            ncc->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            metricWeight=(1.0-ncc->GetValue(transf->GetParameters()))/2;
            break;
        }
        case MSD:{
            typedef typename itk::MeanSquaresImageToImageMetric<
                ImageType,
                ImageType >    MSDMetricType;
            typename MSDMetricType::Pointer msd=MSDMetricType::New();
            msd->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            msd->SetTransform(transf);
            msd->SetFixedImage(targetImage);
            msd->SetMovingImage(movingImage);
            msd->SetInterpolator(interpolator);
            msd->Initialize();
            metricWeight=(msd->GetValue(transf->GetParameters()));
            metricWeight/=m_sigma*m_sigma;
            metricWeight=exp(-metricWeight);
            break;
        }
        case MAD:{
            metricWeight=globalMAD(targetImage,movingImage);
            break;
        }
        case MI:{
            typedef itk::MattesMutualInformationImageToImageMetric< 
                ImageType, 
                ImageType >    MIMetricType;
            typename MIMetricType::Pointer       MImetric      = MIMetricType::New();
            MImetric->ReinitializeSeed( 76926294 );
            MImetric->SetNumberOfHistogramBins( 50 );
            MImetric->SetNumberOfSpatialSamples( 100*50 );
            MImetric->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            MImetric->SetTransform(transf);
            MImetric->SetFixedImage(targetImage);
            MImetric->SetMovingImage(movingImage);
            MImetric->SetInterpolator(interpolator);
            MImetric->Initialize();
            metricWeight=-MImetric->GetValue(transf->GetParameters());
          
            break;
        }
        case NMI:{
            typedef itk::NormalizedMutualInformationHistogramImageToImageMetric< 
                ImageType, 
                ImageType >    NMIMetricType;
            typename NMIMetricType::Pointer      NMImetric     = NMIMetricType::New();
            
            typename NMIMetricType::HistogramType::SizeType histSize;
            histSize[0] = 50;
            histSize[1] = 50;
            NMImetric->SetHistogramSize(histSize);           
            NMImetric->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            NMImetric->SetTransform(transf);
            NMImetric->SetFixedImage(targetImage);
            NMImetric->SetMovingImage(movingImage);
            NMImetric->SetInterpolator(interpolator);
            NMImetric->Initialize();
            metricWeight=NMImetric->GetValue(transf->GetParameters());
            break;
        }
        }   //switch
#endif
        ProbabilisticVectorImagePointerType deformedIncrement=increment;
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());
     
        LOGV(10)<<VAR(metricWeight)<<endl;
        for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight*metricWeight);
        }
    }
    void updateProbabilisticSegmentationLocalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,MetricType metric ){
        ProbabilisticVectorImagePointerType deformedIncrement=increment;
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());


        std::pair<ImagePointerType,ImagePointerType> deformedMoving;
        
        
        ImageNeighborhoodIteratorPointerType tIt=new ImageNeighborhoodIteratorType(m_patchRadius,targetImage,targetImage->GetLargestPossibleRegion());
        ImageNeighborhoodIteratorPointerType aIt=new ImageNeighborhoodIteratorType(m_patchRadius,deformedMoving.first,deformedMoving.first->GetLargestPossibleRegion());
        ImageNeighborhoodIteratorPointerType mIt=new ImageNeighborhoodIteratorType(m_patchRadius,deformedMoving.second,deformedMoving.second->GetLargestPossibleRegion());
        accIt.GoToBegin();incIt.GoToBegin();tIt->GoToBegin();mIt->GoToBegin(); aIt->GoToBegin();
        for (;!accIt.IsAtEnd();++accIt,++incIt,++(*tIt),++(*mIt),++(*aIt)){
            double metricWeight=1;
            switch (metric){
            case MSD:
                metricWeight=localMSD(tIt,aIt,mIt);
                break;
            case MAD:
                metricWeight=localMAD(tIt,aIt,mIt);
                break;
            case NCC:
                metricWeight=localNCC(tIt,aIt,mIt);
                break;
            default:
                metricWeight=1;
            }
            LOGV(10)<<VAR(metricWeight)<<endl;

            accIt.Set(accIt.Get()+incIt.Get()*globalWeight*metricWeight);
        }
        delete tIt; delete aIt; delete mIt;
    }
    void updateProbabilisticSegmentationLocalMetricNew(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,MetricType metric ){
        ProbabilisticVectorImagePointerType deformedIncrement=increment;
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());


        std::pair<ImagePointerType,ImagePointerType> deformedMoving;
        
        deformedMoving.first=movingImage;
        
        FloatImagePointerType metricImage;
        switch (metric){
        case MSD:
            metricImage=FilterUtils<ImageType,FloatImageType>::LSSDAutoNorm(deformedMoving.first, targetImage,m_patchRadius[0],m_sigma);
            break;
        case MAD:
            metricImage=FilterUtils<ImageType,FloatImageType>::LSADAutoNorm(deformedMoving.first, targetImage,m_patchRadius[0],m_sigma);
            break;
        case NCC:
            metricImage=FilterUtils<ImageType,FloatImageType>::efficientLNCC(deformedMoving.first, targetImage,m_patchRadius[0], m_sigma);
            break;
        default:
            LOG<<"no valid metric, aborting"<<endl;
            exit(0);
        }
        LOGI(8,ImageUtils<FloatImageType>::writeImage("weightImage.nii",metricImage));
        FloatImageIteratorType weightIt(metricImage,metricImage->GetLargestPossibleRegion());
        weightIt.GoToBegin();
        accIt.GoToBegin();incIt.GoToBegin();
        for (;!accIt.IsAtEnd();++accIt,++incIt, ++weightIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight*weightIt.Get());
        }
    }
    ProbabilisticVectorImagePointerType createEmptyProbImageFromImage(ImagePointerType input){
        ProbabilisticVectorImagePointerType output=ProbabilisticVectorImageType::New();
        output->SetOrigin(input->GetOrigin());
        output->SetSpacing(input->GetSpacing());
        output->SetDirection(input->GetDirection());
        output->SetRegions(input->GetLargestPossibleRegion());
        output->Allocate();
        ProbabilisticPixelType p;
        p.Fill(0.0);
        output->FillBuffer(p);
        return output;
        
    }

    ProbabilisticVectorImagePointerType warpProbImage(ProbabilisticVectorImagePointerType input, DeformationFieldPointerType deformation){
        ProbabilisticVectorImagePointerType output=ProbabilisticVectorImageType::New();
        output->SetOrigin(deformation->GetOrigin());
        output->SetSpacing(deformation->GetSpacing());
        output->SetDirection(deformation->GetDirection());
        output->SetRegions(deformation->GetLargestPossibleRegion());
        output->Allocate();
        ProbImageIteratorType outIt(output,output->GetLargestPossibleRegion());
        typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
            ProbabilisticVectorImageType ,double> DefaultFieldInterpolatorType;
        typename DefaultFieldInterpolatorType::Pointer interpolator=DefaultFieldInterpolatorType::New();
        interpolator->SetInputImage(input);

        DeformationIteratorType deformationIt(deformation,deformation->GetLargestPossibleRegion());
        for (outIt.GoToBegin(),deformationIt.GoToBegin();!outIt.IsAtEnd();++outIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename DefaultFieldInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();
            PointType p;
            output->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            input->TransformPhysicalPointToContinuousIndex(p,idx);
            outIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
        }
        return output;
    }
    
    double localMAD(ImageNeighborhoodIteratorPointerType tIt, ImageNeighborhoodIteratorPointerType aIt,ImageNeighborhoodIteratorPointerType mIt){
        double result=0;
        int count=0;
        for (unsigned int i=0;i<tIt->Size();++i){
            if (mIt->GetPixel(i)){
                result+=fabs(tIt->GetPixel(i)-aIt->GetPixel(i));
                count++;
            }
        }
        if (!count)
            return 1.0;
        return exp(-result/count/m_sigma);
    }
    double localMSD(ImageNeighborhoodIteratorPointerType tIt, ImageNeighborhoodIteratorPointerType aIt,ImageNeighborhoodIteratorPointerType mIt){
        double result=0;
        int count=0;
        for (unsigned int i=0;i<tIt->Size();++i){
            if (mIt->GetPixel(i)){
                double tmp=(tIt->GetPixel(i)-aIt->GetPixel(i));
                result+=tmp*tmp;
                count++;
            }
        }
        if (!count)
            return 1.0;
        return  exp(-result/count/(m_sigma*m_sigma));
    }
    double localNCC(ImageNeighborhoodIteratorPointerType tIt, ImageNeighborhoodIteratorPointerType aIt,ImageNeighborhoodIteratorPointerType mIt){
        double result=0;
        int count=0;
        double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
        for (unsigned int i=0;i<tIt->Size();++i){
            if (mIt->GetPixel(i)){
                double f=tIt->GetPixel(i);
                double m= aIt->GetPixel(i);
                sff+=f*f;
                smm+=m*m;
                sfm+=f*m;
                sf+=f;
                sm+=m;
                count+=1;
            }
        }
        if (!count)
            return 0.5;
        else{
            double NCC=0;
            sff -= ( sf * sf / count );
            smm -= ( sm * sm / count );
            sfm -= ( sf * sm / count );
            if (smm*sff>0){
                NCC=1.0*sfm/sqrt(smm*sff);
            }
            result=(1.0+NCC)/2;
        }
        return result;
    }

    double globalMAD(ImagePointerType target, ImagePointerType moving, DeformationFieldPointerType deformation){
        std::pair<ImagePointerType,ImagePointerType> deformedMoving = TransfUtilsType::warpImageWithMask(moving,deformation);
        ImageIteratorType tIt(target,target->GetLargestPossibleRegion());
        ImageIteratorType mIt(deformedMoving.first,deformedMoving.first->GetLargestPossibleRegion());
        ImageIteratorType maskIt(deformedMoving.second,deformedMoving.second->GetLargestPossibleRegion());
        tIt.GoToBegin();mIt.GoToBegin();maskIt.GoToBegin();
        double result=0.0;int count=0;
        for (;!tIt.IsAtEnd();++tIt,++mIt,++maskIt){
            if (maskIt.Get()){
                result+=fabs(tIt.Get()-mIt.Get());
                count++;
            }
        }
        if (count)
            return exp(-result/count/m_sigma);
        else
            return 0.0;
    }
    
    double DICE(ImagePointerType seg1, ImagePointerType seg2){
        typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
        typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
        seg1=FilterUtils<ImageType>::binaryThresholdingLow(seg1,1.0);
        seg2=FilterUtils<ImageType>::binaryThresholdingLow(seg2,1.0);
        filter->SetSourceImage(seg1);
        filter->SetTargetImage(seg2);
        filter->Update();
        return filter->GetDiceCoefficient(1);
    }
};//class
#endif
