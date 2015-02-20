#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.h"
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
#include "ChamferDistanceTransform.h"
#include <map>
#include "ArgumentParser.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>


using namespace std;

const unsigned int D=3;
typedef unsigned char Label;
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;




int main(int argc, char * argv [])
{


    ArgumentParser as(argc, argv);
	string groundTruth,segmentationFilename,outputFilename="";
    bool hausdorff=false;
    double threshold=-9999999;
    bool convertFromClassified=false;
    bool multilabel=false;
    bool connectedComponent=false;
	as.parameter ("g", groundTruth, "groundtruth image (file name)", true);
	as.parameter ("s", segmentationFilename, "segmentation image (file name)", true);
	as.parameter ("o", outputFilename, "output image (file name)", false);
	as.option ("h", hausdorff, "compute hausdorff distance(0,1)", false);
	as.parameter ("t", threshold, "threshold segmentedImage (threshold)", false);
	as.parameter ("c", convertFromClassified, "convert from classified segmentation (after normalization) (0,1)", false);
	as.option ("m", multilabel, "convert from multilabel segmentation");
	as.option ("l", connectedComponent, "use largest connected component in segmentation");
	as.parse();
	

 
    LabelImage::Pointer groundTruthImg =
        ImageUtils<LabelImage>::readImage(groundTruth);
    LabelImage::Pointer segmentedImg =
        ImageUtils<LabelImage>::readImage(segmentationFilename);

    
 
    unsigned totalPixels = 0;

    //    groundTruthImg=normalizeImage(groundTruthImg);
    convertToBinaryImage(groundTruthImg);
    TRealImage::Pointer distancesOutsideTruthBone;
    TRealImage::Pointer distancesInsideTruthBone;
    TRealImage::Pointer distanceMap;
    float maxAbsDistance = 0;
    float maxDistance = -std::numeric_limits<float>::max();
    double minSum=0,maxSum=0;
    int minDistance = std::numeric_limits<int>::max();
    double minCount=0, maxCount=0;
    int sum = 0;
    unsigned totalEdges = 0;
    float mean=0;
 
    if (multilabel){
        segmentedImg=convertToBinaryImageFromMultiLabel(segmentedImg);
    }
    else{
        if (threshold!=-9999999){
            segmentedImg= convertToBinaryImage (segmentedImg, threshold);         
        }else{
            segmentedImg= convertToBinaryImage ( segmentedImg) ;
        }
    }    
    typedef LabelImage::ConstPointer ConstType;
    if (connectedComponent){  
        typedef itk::MinimumMaximumImageCalculator <LabelImage>
            ImageCalculatorFilterType;
        typedef itk::ConnectedComponentImageFilter<LabelImage,LabelImage>  ConnectedComponentImageFilterType;
        ConnectedComponentImageFilterType::Pointer filter =
            ConnectedComponentImageFilterType::New();
        filter->SetInput(segmentedImg);
        filter->Update();
    
        typedef itk::LabelShapeKeepNObjectsImageFilter< LabelImage > LabelShapeKeepNObjectsImageFilterType;
        LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
        labelShapeKeepNObjectsImageFilter->SetInput( filter->GetOutput() );
        labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
        labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1);
        labelShapeKeepNObjectsImageFilter->SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
        labelShapeKeepNObjectsImageFilter->Update();
      
        
      
        segmentedImg = convertToBinaryImage (labelShapeKeepNObjectsImageFilter->GetOutput(), 0.1);     //;//f->GetOutput();// FilterUtils<LabelImage>::binaryThresholding(labelShapeKeepNObjectsImageFilter->GetOutput(),1,10000);//filter->GetOutput(),1,1);
    }
   
   
    if (hausdorff){
        typedef itk::HausdorffDistanceImageFilter<LabelImage, LabelImage> HausdorffDistanceFilterType;
        typedef HausdorffDistanceFilterType::Pointer HDPointerType;
        HDPointerType hdFilter=HausdorffDistanceFilterType::New();;
        hdFilter->SetInput1(groundTruthImg);
        hdFilter->SetInput2(segmentedImg);
        hdFilter->SetUseImageSpacing(true);
        hdFilter->Update();
        mean=hdFilter->GetAverageHausdorffDistance();
        maxAbsDistance=hdFilter->GetHausdorffDistance();
    }

    typedef itk::LabelOverlapMeasuresImageFilter<LabelImage> OverlapMeasureFilterType;
    OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
    filter->SetSourceImage(groundTruthImage);
    filter->SetTargetImage(segmentedImage);
    filter->Update();
    double dice=filter->GetDiceCoefficient();
    std::cout<<"Dice " << dice ;
    std::cout<<"  Mean "<< mean;
    std::cout<<" MaxAbs "<< maxAbsDistance;
    std::cout<< std::endl;
    // std::cout<<"EvalG - % of bone segmented "<< float(glob.truePos) / ((glob.truePos + glob.falseNeg) / 100)<< std::endl;

    

  


	return EXIT_SUCCESS;
}
