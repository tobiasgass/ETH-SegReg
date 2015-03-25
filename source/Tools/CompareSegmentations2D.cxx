#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkCastImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include <map>
#include "itkConnectedComponentImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include <itkMinimumMaximumImageCalculator.h>
#include <itkHausdorffDistanceImageFilter.h>
#include <map>
#include "ArgumentParser.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include "mmalloc.h"
#include "SegmentationMapper.hxx"
using namespace std;

const unsigned int D=2;
typedef unsigned short Label;
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;
typedef  LabelImage::Pointer LabelImagePointerType;

LabelImagePointerType selectLabel(LabelImagePointerType img, Label l){
    LabelImagePointerType result=ImageUtils<LabelImage>::createEmpty(img);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(img,img->GetLargestPossibleRegion());
    IteratorType it2(result,img->GetLargestPossibleRegion());
    for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
        it2.Set(it1.Get()==l);
    }
    return result;
}


int main(int argc, char * argv [])
{

    
    ArgumentParser as(argc, argv);
	string groundTruth,segmentationFilename,outputFilename="";
    bool hausdorff=false;
    double threshold=1;
    int evalLabel=1;
    bool connectedComponent=false;
    int labelsToEvaluate=1;
	as.parameter ("g", groundTruth, "groundtruth image (file name)", true);
	as.parameter ("s", segmentationFilename, "segmentation image (file name)", true);
	as.parameter ("o", outputFilename, "output image (file name)", false);
    as.parameter ("t", threshold, "threshold segmentedImage (threshold)", false);
	as.parameter ("e", evalLabel, "label to evaluate", false);
	as.parameter ("labelsToEvaluate", labelsToEvaluate, "labels to evaluate", false);
    as.option ("h", hausdorff, "compute hausdorff distance(0,1)");
	as.option ("l", connectedComponent, "use largest connected component in segmentation");
	as.parse();
	

 
    LabelImage::Pointer groundTruthImg =
        ImageUtils<LabelImage>::readImage(groundTruth);
    LabelImage::Pointer segmentedImg =
        ImageUtils<LabelImage>::readImage(segmentationFilename);

    SegmentationMapper<LabelImage> segmentationMapper;
    groundTruthImg=segmentationMapper.FindMapAndApplyMap(groundTruthImg);
    segmentedImg=segmentationMapper.ApplyMap(segmentedImg);
 
    
  
    TRealImage::Pointer distancesOutsideTruthBone;
    TRealImage::Pointer distancesInsideTruthBone;
    TRealImage::Pointer distanceMap;
    float maxAbsDistance = 0;
    float mean=0;
    for (int l=0;l<labelsToEvaluate;++l){
        LabelImage::Pointer evalGroundTruthImage=            selectLabel(groundTruthImg,evalLabel);   
        LabelImage::Pointer evalSegmentedImage=            selectLabel(segmentedImg,evalLabel);   
       
        typedef LabelImage::ConstPointer ConstType;
        if (connectedComponent){  
            typedef itk::MinimumMaximumImageCalculator <LabelImage>
                ImageCalculatorFilterType;
            typedef itk::ConnectedComponentImageFilter<LabelImage,LabelImage>  ConnectedComponentImageFilterType;
            ConnectedComponentImageFilterType::Pointer filter =
                ConnectedComponentImageFilterType::New();
            filter->SetInput(evalSegmentedImage);
            filter->Update();
    
            typedef itk::LabelShapeKeepNObjectsImageFilter< LabelImage > LabelShapeKeepNObjectsImageFilterType;
            LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
            labelShapeKeepNObjectsImageFilter->SetInput( filter->GetOutput() );
            labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
            labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1);
            labelShapeKeepNObjectsImageFilter->SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
            labelShapeKeepNObjectsImageFilter->Update();
            evalSegmentedImage =  FilterUtils<LabelImage>::binaryThresholdingLow(labelShapeKeepNObjectsImageFilter->GetOutput(), 1);     //;//f->GetOutput();// FilterUtils<LabelImage>::binaryThresholding(labelShapeKeepNObjectsImageFilter->GetOutput(),1,10000);//filter->GetOutput(),1,1);
        }
   
   
        if (hausdorff){
            typedef itk::HausdorffDistanceImageFilter<LabelImage, LabelImage> HausdorffDistanceFilterType;
            typedef HausdorffDistanceFilterType::Pointer HDPointerType;
            HDPointerType hdFilter=HausdorffDistanceFilterType::New();;
            hdFilter->SetInput1(evalGroundTruthImage);
            hdFilter->SetInput2(evalSegmentedImage);
            hdFilter->SetUseImageSpacing(true);
            hdFilter->Update();
            mean=hdFilter->GetAverageHausdorffDistance();
            maxAbsDistance=hdFilter->GetHausdorffDistance();
        }

        typedef itk::LabelOverlapMeasuresImageFilter<LabelImage> OverlapMeasureFilterType;
        OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
        filter->SetSourceImage(evalGroundTruthImage);
        filter->SetTargetImage(evalSegmentedImage);
        filter->Update();
        double dice=filter->GetDiceCoefficient();
        std::cout<<"Label "<<segmentationMapper.GetInverseMappedLabel(evalLabel) ;
        std::cout<<" Dice " << dice ;
        if (hausdorff){
        std::cout<<" Mean "<< mean;
        std::cout<<" MaxAbs "<< maxAbsDistance<<" ";
        }
        std::cout<<endl;
        ++evalLabel;
    }
    std::cout<< std::endl;
    // std::cout<<"EvalG - % of bone segmented "<< float(glob.truePos) / ((glob.truePos + glob.falseNeg) / 100)<< std::endl;

    

  


	return EXIT_SUCCESS;
}

