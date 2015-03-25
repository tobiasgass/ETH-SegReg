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

const unsigned int D=3;
typedef int Label;
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;
typedef  LabelImage::Pointer LabelImagePointerType;

LabelImagePointerType selectLabel(LabelImagePointerType img, Label l, bool &present){
    LabelImagePointerType result=ImageUtils<LabelImage>::createEmpty(img);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(img,img->GetLargestPossibleRegion());
    IteratorType it2(result,img->GetLargestPossibleRegion());
    for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
        bool check=it1.Get()==l;
        it2.Set(check);
        if (check) {present=true;}
    }
    return result;
}
LabelImagePointerType differenceLabels(LabelImagePointerType img, LabelImagePointerType img2){
    LabelImagePointerType result=ImageUtils<LabelImage>::createEmpty(img);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(img,img->GetLargestPossibleRegion());
    IteratorType it3(img2,img2->GetLargestPossibleRegion());
    IteratorType it2(result,img->GetLargestPossibleRegion());
    for (it1.GoToBegin(),it2.GoToBegin(),it3.GoToBegin();!it1.IsAtEnd();++it1,++it2,++it3){
        it2.Set(it1.Get()!=it3.Get());
    }
    return result;
}

int main(int argc, char * argv [])
{

    
    ArgumentParser as(argc, argv);
	string groundTruth,segmentationFilename,outputFilename="",maskFilename="";
    bool hausdorff=false;
    double threshold=1;
    int evalLabel=-1;
    bool connectedComponent=false;
    int labelsToEvaluate=-1;
    int verbose=0;
    string labelList="";
    bool evalAll=false;
    bool resampleIfNeeded=false;
    bool excludeMissing=false;
	as.parameter ("g", groundTruth, "groundtruth image (file name)", true);
	as.parameter ("s", segmentationFilename, "segmentation image (file name)", true);
	as.parameter ("m", maskFilename, "Binary mask in which measures are to be computed (file name)", false);
	as.parameter ("o", outputFilename, "output image (file name)", false);
    as.parameter ("t", threshold, "threshold segmentedImage (threshold)", false);
	as.parameter ("e", evalLabel, "label to evaluate", false);
    
	as.parameter ("labelsToEvaluate", labelsToEvaluate, "labels to evaluate", false);
	as.parameter ("labelList", labelList, "list of labels to evaluate", false);
    as.option ("all", evalAll, "compute mean overlap [disables hausdorff]");
    as.option ("excludeMissing", excludeMissing, "exclude labels missing from segmentation estimate");
    as.option ("h", hausdorff, "compute hausdorff distance(0,1)");
	as.option ("l", connectedComponent, "use largest connected component in segmentation");
	as.option ("r", resampleIfNeeded, "resample input seg to GT seg if necessary");
	as.parameter ("v", verbose, "verbosity level", false);

    
	as.parse();
	
    logSetVerbosity(verbose);
 
    LabelImage::Pointer groundTruthImg =
        ImageUtils<LabelImage>::readImage(groundTruth);
    LabelImage::Pointer segmentedImg =
        ImageUtils<LabelImage>::readImage(segmentationFilename);
    if (resampleIfNeeded){
        if (groundTruthImg->GetSpacing()!=segmentedImg->GetSpacing() 
            || groundTruthImg->GetOrigin()!=segmentedImg->GetOrigin() 
            ||groundTruthImg->GetLargestPossibleRegion().GetSize()!=segmentedImg->GetLargestPossibleRegion().GetSize() ){
            segmentedImg=FilterUtils<LabelImage>::NNResample(segmentedImg,groundTruthImg,false);

        }

    }

    LabelImage::Pointer mask = NULL;
    if (maskFilename !=""){
         LabelImage::Pointer mask =
             ImageUtils<LabelImage>::readImage(maskFilename);
         groundTruthImg=ImageUtils<LabelImage>::multiplyImageOutOfPlace(groundTruthImg,mask);
         segmentedImg=ImageUtils<LabelImage>::multiplyImageOutOfPlace(segmentedImg,mask);
    }

    

    if (evalAll){
        typedef itk::LabelOverlapMeasuresImageFilter<LabelImage> OverlapMeasureFilterType;
        OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
        filter->SetSourceImage(groundTruthImg);
        filter->SetTargetImage(segmentedImg);
        filter->SetCoordinateTolerance(1e-3);
        filter->Update();
        double dice=filter->GetDiceCoefficient();
        std::cout<<"Label ALL Dice "<<dice<<std::endl;
        if (outputFilename!=""){
            LabelImage::Pointer differenceImage=differenceLabels(groundTruthImg,segmentedImg);
            ImageUtils<LabelImage>::writeImage(outputFilename,differenceImage);
        }
        return 1;
    }

    std::vector<int> listOfLabels;
    SegmentationMapper<LabelImage> segmentationMapper;
        
    if (labelList==""){
        groundTruthImg=segmentationMapper.FindMapAndApplyMap(groundTruthImg);
        segmentedImg=segmentationMapper.ApplyMap(segmentedImg);
        if (evalLabel>0){
            labelsToEvaluate=1;
        }else{
            evalLabel=1;
        }
    }else{
        ifstream ifs(labelList.c_str());
        labelsToEvaluate=0;
        do{
            int tmp;
            ifs>>tmp;
            listOfLabels.push_back(tmp);
            LOGV(1)<<VAR(labelsToEvaluate)<<" "<<VAR(tmp)<<endl;
            ++labelsToEvaluate;
        } while (!ifs.eof());
        ifs.close();
        --labelsToEvaluate;
    }
 
    
  
    TRealImage::Pointer distancesOutsideTruthBone;
    TRealImage::Pointer distancesInsideTruthBone;
    TRealImage::Pointer distanceMap;
    float maxAbsDistance = 0;
    float mean=0;
    if (labelsToEvaluate<0 && (labelList == "")){
        labelsToEvaluate=segmentationMapper.getNumberOfLabels()-1;
    }
    for (int l=0;l<labelsToEvaluate;++l){
        if (listOfLabels.size()){
            evalLabel=listOfLabels[l];
            LOGV(2)<<"Evaluating label "<<VAR(evalLabel)<<" "<<VAR(l)<<endl;
            //skip label 0
            if (evalLabel==0){
                continue;
            }
        }
        bool present;
        LabelImage::Pointer evalGroundTruthImage=            selectLabel(groundTruthImg,evalLabel,present);   
        present=false;
        LabelImage::Pointer evalSegmentedImage=            selectLabel(segmentedImg,evalLabel,present);   
        
        if (present || !excludeMissing ){
       
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
            if (outputFilename!=""){
                ImageUtils<LabelImage>::writeImage(outputFilename,evalSegmentedImage);
            }
        }
   
   
        if (hausdorff){
            typedef itk::HausdorffDistanceImageFilter<LabelImage, LabelImage> HausdorffDistanceFilterType;
            typedef HausdorffDistanceFilterType::Pointer HDPointerType;
            HDPointerType hdFilter=HausdorffDistanceFilterType::New();
            evalSegmentedImage=FilterUtils<LabelImage>::NNResample(evalSegmentedImage,evalGroundTruthImage,false);
            hdFilter->SetInput1(evalGroundTruthImage);
            hdFilter->SetInput2(evalSegmentedImage);
            hdFilter->SetUseImageSpacing(true);
            hdFilter->SetCoordinateTolerance(1e-3);
            hdFilter->SetDirectionTolerance(1e-3);
            hdFilter->Update();
            mean=hdFilter->GetAverageHausdorffDistance();
            maxAbsDistance=hdFilter->GetHausdorffDistance();
        }

        typedef itk::LabelOverlapMeasuresImageFilter<LabelImage> OverlapMeasureFilterType;
        OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
        filter->SetSourceImage(evalGroundTruthImage);
        filter->SetTargetImage(evalSegmentedImage);
        filter->SetCoordinateTolerance(1e-3);
        filter->Update();
        double dice=filter->GetDiceCoefficient();
        int label=evalLabel;
        if (labelList==""){
            label=segmentationMapper.GetInverseMappedLabel(evalLabel) ;
        }
        std::cout<<" Label "<< label;
        std::cout<<" Dice " << dice ;
        if (hausdorff){
            std::cout<<" Mean "<< mean;
            std::cout<<" MaxAbs "<< maxAbsDistance<<" ";
        }
        std::cout<<endl;
        }
        //std::cout<<endl;
        ++evalLabel;
    }
    std::cout<< std::endl;
    // std::cout<<"EvalG - % of bone segmented "<< float(glob.truePos) / ((glob.truePos + glob.falseNeg) / 100)<< std::endl;

    

  


	return EXIT_SUCCESS;
}

