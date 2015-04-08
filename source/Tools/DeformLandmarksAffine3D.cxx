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
#include "TransformationUtils.h"

using namespace std;
using namespace itk;

const unsigned int D=3;
typedef short Label;
typedef itk::Image< Label, D > ImageType;
typedef  ImageType::Pointer LabelImagePointerType;



int main(int argc, char * argv [])
{

    
    if (argc != 4){
        LOG<<"USAGE : convertPointToIndex3D targetimage pointsFile affineTransform " << endl;
        exit(0);
    }

        
 
    ImageType::Pointer image = ImageUtils<ImageType>::readImage(argv[1]);
    typedef TransfUtils<ImageType>::AffineTransformPointerType  AffineTransformPointerType;
    AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(argv[3]);
    ImageType::PointType point,newPoint;
    ImageType::IndexType idx;

    ifstream ifs(argv[2]);
    bool valid=true;
    while ( ! ifs.eof() ) {
        for (int d=0;d<D;++d){
            if (ifs.eof()){
                valid=false;
                break;
            }
            ifs>>point[d];
        }
        newPoint=affine->TransformPoint(point);
        //        std::cout<<point<<endl;
        //std::cout<<affine->GetMatrix()<<endl;
        //std::cout<<affine->GetOffset()<<endl;
        std::cout<<newPoint<<endl;
    }
    return EXIT_SUCCESS;
}

