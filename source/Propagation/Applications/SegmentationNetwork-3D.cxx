#include <stdio.h>
#include <iostream>
//#include "SegmentationNetwork.h"
#include "Segmentation-Network-New.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef short PixelType;
    const unsigned int D=3;
    typedef itk::Image<PixelType,D> ImageType;
    //SegmentationNetwork<ImageType> net;
    SegmentationNetwork<ImageType,2> net;
    net.run(argc,argv);
    return 1;
}
