#include <stdio.h>
#include <iostream>
//#include "SegmentationNetwork.h"
#include "Segmentation-Network-New.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef unsigned short int PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    SegmentationNetwork<ImageType,2> net;
    //    SegmentationNetwork<ImageType> net;
    net.run(argc,argv);
    return 1;
}
