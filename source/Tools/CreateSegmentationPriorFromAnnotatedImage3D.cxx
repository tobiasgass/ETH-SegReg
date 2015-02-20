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



int main(int argc, char * argv [])
{

    
    if (argc != 5){
        LOG<<"USAGE : convertPointToIndex3D image x y z " << endl;
    }

        
 
    LabelImage::Pointer image =        ImageUtils<LabelImage>::readImage(argv[1]);
    
    ImageType::PointType point;
    
    point[0]=atof(argv[2]);
    point[1]=atof(argv[3]);
    point[2]=atof(argv[4]);
    
    ImageType::IndexType idx;
    image->ConvertPhysicalPointToIndex(point,idx);

    LOG<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<endl;

    
    LabelImagePointerType result=ImageUtils<LabelImage>::createEmpty(segmentedImg);
    typedef itk::ImageRegionIterator<LabelImage> IteratorType;
    IteratorType it1(segmentedImg,segmentedImg->GetLargestPossibleRegion());
    IteratorType it2(result,segmentedImg->GetLargestPossibleRegion());
    for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
        int label=it1.Get();
        if (label==20000)
            label=2;
        else if (label==10000)
            label=1;
        else
            label=0;
        it2.Set(label);
    }
    
    ImageUtils<LabelImage>::writeImage(outputFilename,result);

	return EXIT_SUCCESS;
}

