#include <stdio.h>
#include <iostream>
#include "Segmentation-Propagation.h"
#include "Segmentation-Propagation-DefWeighting.h"
//#include "Segmentation-Propagation-Modular.h"
#include "Segmentation-Propagation-efficient1Hop.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef unsigned short PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    //SegmentationPropagationDeformationWeighting<ImageType> net;
    //SegmentationPropagation<ImageType> net;
    SegmentationPropagationModular<ImageType,2> net;
    net.run(argc,argv);
    return 1;
}
