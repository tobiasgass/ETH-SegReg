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
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;
typedef  LabelImage::Pointer LabelImagePointerType;


int main(int argc, char * argv [])
{

    
    argstream as(argc, argv);
	string segmentationFilename,outputFilename="";
    int verbose=0;
    string labelList="";
    int targetLabel=-1;
    bool binary=false;
	as >> parameter ("s", segmentationFilename, "segmentation image (file name)", true);
	as >> parameter ("o", outputFilename, "output image (file name)", true);
	as >> parameter ("v", verbose, "verbosity level", false);

	as >> help();
	as.defaultErrorHandling();
    logSetVerbosity(verbose);
 
     SegmentationMapper<LabelImage> segmentationMapper;

    LabelImage::Pointer segmentedImg =
        ImageUtils<LabelImage>::readImage(segmentationFilename);

    segmentedImg=segmentationMapper.FindMapAndApplyMap(segmentedImg);

   
        ImageUtils<LabelImage>::writeImage(outputFilename,segmentedImg);
   

	return EXIT_SUCCESS;
}

