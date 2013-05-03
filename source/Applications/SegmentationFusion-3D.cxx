#include <stdio.h>
#include <iostream>

#include "Segmentation-Fusion-Modular.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef short PixelType;
    const unsigned int D=3;
    typedef itk::Image<PixelType,D> ImageType;
    //SegmentationPropagationDeformationWeighting<ImageType> net;
    //SegmentationPropagation<ImageType> net;
    SegmentationFusionModular<ImageType,3> net;
    net.run(argc,argv);
    return 1;
}
