
#include <stdio.h>
#include <iostream>
#include "GCoptimization.h"
#include "ArgumentParser.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "bgraph.h"
#include <sstream>
#include "ArgumentParser.h"
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
using namespace std;

const int nSegmentationLabels = 2; 
typedef short PixelType;
const unsigned int D=3;
typedef itk::Image<PixelType,D> ImageType;
typedef   ImageType::Pointer ImagePointerType;
typedef   ImageType::IndexType IndexType;
typedef   ImageType::PointType PointType;
typedef   ImageType::OffsetType OffsetType;
typedef   ImageType::SizeType SizeType;
typedef   ImageType::ConstPointer ImageConstPointerType;
typedef   ImageUtils<ImageType>::FloatImageType FloatImageType;
typedef   FloatImageType::Pointer FloatImagePointerType;

typedef  TransfUtils<ImageType,double> TransfUtilsType;

typedef   TransfUtilsType::DisplacementType DisplacementType; 
typedef   TransfUtilsType::DeformationFieldType DeformationFieldType;
typedef   DeformationFieldType::Pointer DeformationFieldPointerType;
typedef   itk::ImageRegionIterator<ImageType> ImageIteratorType;
typedef   itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
typedef   itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
typedef   itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
typedef   ImageNeighborhoodIteratorType::RadiusType RadiusType;

typedef itk::Vector<float,nSegmentationLabels> ProbabilisticPixelType;
typedef itk::Image<ProbabilisticPixelType,D> ProbabilisticVectorImageType;
typedef  ProbabilisticVectorImageType::Pointer ProbabilisticVectorImagePointerType;
typedef  itk::ImageRegionIterator<ProbabilisticVectorImageType> ProbImageIteratorType;

  

ImagePointerType probSegmentationToSegmentationGraphcutMultiLabel( ProbabilisticVectorImagePointerType img, ImagePointerType segImg, double smooth, double sigma, int nSegmentationLabels){
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
                //costs[n].cost=-log(prob/sum);
                costs[n].cost=-log(prob);
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


int main(int argc, char ** argv){
    feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string probImageFilename="",imageFilename="",outImageFilename="";
    double smoothness=1.0;
    double m_graphCutSigma=10;
    int nSegmentations=2;
    bool graphCut=true;
    int verbose = 0  ; 
    as->parameter ("p",probImageFilename , "prob image file name", true);
    as->parameter ("i", imageFilename, " intensity image", true);
    as->parameter ("o", outImageFilename, " output segmentations image", true);
    as->parameter ("sigmaGC", m_graphCutSigma,"sigma for exp(- contrast/sigma) for graphcut smoothness",false);
    as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
    as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
    as->parameter ("nSegLabels", nSegmentations,"number of segmentation labels (including background)",false);
    as->parameter ("verbose", verbose,"get verbose output",false);
    as->parse();
    
      
    ImagePointerType img=ImageUtils<ImageType>::readImage(imageFilename);
    ProbabilisticVectorImagePointerType probs=ImageUtils<ProbabilisticVectorImageType>::readImage(probImageFilename);
    ImagePointerType result = probSegmentationToSegmentationGraphcutMultiLabel(probs,img,smoothness,m_graphCutSigma,nSegmentations);
        
    ImageUtils<ImageType>::writeImage(outImageFilename,result);
    return 1;
}//main

 
   
