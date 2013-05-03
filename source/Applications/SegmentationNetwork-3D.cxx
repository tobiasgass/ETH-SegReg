#include <stdio.h>
#include <iostream>
#include "SegmentationNetwork.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef float PixelType;
    const unsigned int D=3;
    typedef itk::Image<PixelType,D> ImageType;
    SegmentationNetwork<ImageType> net;
    net.run(argc,argv);
    return 1;
}
