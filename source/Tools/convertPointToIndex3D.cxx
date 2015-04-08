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
typedef itk::Image< Label, D > ImageType;
typedef  ImageType::Pointer LabelImagePointerType;



int main(int argc, char * argv [])
{

    
    if (argc != 3){
        LOG<<"USAGE : convertPointToIndex3D image indexFile " << endl;
        exit(0);
    }

        
 
    ImageType::Pointer image =        ImageUtils<ImageType>::readImage(argv[1]);
    
    ImageType::PointType point;
    ImageType::IndexType idx;

    ifstream ifs(argv[2]);
    bool valid=true;
    while ( ! ifs.eof() ) {
        for (int d=0;d<D;++d){
            if (ifs.eof()){
                valid=false;
                break;
            }
            ifs>>idx[d];
            if (image->GetDirection()[d][d]==-1.0)
                idx[d]=image->GetBufferedRegion().GetSize()[d]-idx[d];
        }
        if (valid){
            image->TransformIndexToPhysicalPoint(idx,point);
            for (int d=0;d<D;++d){
                std::cout<<point[d]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    return EXIT_SUCCESS;
}

