#include <stdio.h>
#include <iostream>
#include "Segmentation-Propagation.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef unsigned short int PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    SegmentationPropagation<ImageType> net;
    net.run(argc,argv);
    return 1;
}
