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
#include "argstream.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include "mmalloc.h"
#include "SegmentationMapper.hxx"
using namespace std;

const unsigned int D=3;
typedef short Label;
typedef itk::Image< Label, D > ImageType;
typedef  ImageType::Pointer LabelImagePointerType;



int main(int argc, char * argv [])
{

    
    if (argc != 5){
        LOG<<"USAGE : convertPointToIndex3D image x y z " << endl;
        exit(0);
    }

        
 
    ImageType::Pointer image =        ImageUtils<ImageType>::readImage(argv[1]);
    
    ImageType::PointType point;
    
    point[0]=atof(argv[2]);
    point[1]=atof(argv[3]);
    point[2]=atof(argv[4]);
    
    ImageType::IndexType idx;
    bool inside = image->TransformPhysicalPointToIndex(point,idx);

    if (inside){
        std::cout<<idx[2]<<endl;
    }else{
        LOG<<VAR(idx)<<" not inside largest possible region of " << argv[1]<<endl;
    }

	return EXIT_SUCCESS;
}

