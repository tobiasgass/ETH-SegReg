#include <stdio.h>
#include <iostream>
//#include "Segmentation-Propagation.h"
#include "Segmentation-Propagation-Modular.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef short int PixelType;
    const unsigned int D=3;
    typedef itk::Image<PixelType,D> ImageType;
    SegmentationPropagationModular<ImageType,2> net;
    net.run(argc,argv);
    return 1;
}
