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
typedef short Label;
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
	string segmentationFilename,outputFilename="";
    int verbose=0;
    string labelList="";
    int targetLabel=-1;
    bool binary=false;
	as.parameter ("s", segmentationFilename, "segmentation image (file name)", true);
	as.parameter ("o", outputFilename, "output image (file name)", true);
	as.parameter ("labelList", labelList, "list of labels to evaluate", false);
	as.parameter ("label", targetLabel, "labels to evaluate", false);
	as.option ("bin", binary, "compute binary labelling");
	as.parameter ("v", verbose, "verbosity level", false);

	as.parse();
	
    logSetVerbosity(verbose);
 
 
    LabelImage::Pointer segmentedImg =
        ImageUtils<LabelImage>::readImage(segmentationFilename);

    std::vector<int> listOfLabels;
    if (labelList!=""){
    ifstream ifs(labelList.c_str());
    int old=-1;
    do{
        int tmp;
        ifs>>tmp;
        if (tmp<old){
            LOG<<VAR(tmp)<<" smaller "<<VAR(old)<<"; ordererd labels are required" <<std::endl;
            exit(0);
        }
        listOfLabels.push_back(tmp);
        LOGV(1)<<" "<<VAR(tmp)<<endl;
    } while (!ifs.eof());
    ifs.close();
    }
    LabelImagePointerType result=ImageUtils<LabelImage>::createEmpty(segmentedImg);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(segmentedImg,segmentedImg->GetLargestPossibleRegion());
    IteratorType it2(result,segmentedImg->GetLargestPossibleRegion());
    int count=0;
    for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
        int label=it1.Get();
        if (labelList !=""){
            //set label to zero if not in list; requires ordererd list of labels
            if ( ! binary_search(listOfLabels.begin(), listOfLabels.end(), label) ){
                label=0;
            }
        }else{
            label=label*(label==targetLabel);
        }
        if (binary)
            label=label!=0;
        count+=label>0;
        it2.Set(label);
    }
    
    if (count){
        ImageUtils<LabelImage>::writeImage(outputFilename,result);
    }else{
        LOG<<"Label count ZERO, not writing output"<<std::endl;
    }

	return EXIT_SUCCESS;
}

