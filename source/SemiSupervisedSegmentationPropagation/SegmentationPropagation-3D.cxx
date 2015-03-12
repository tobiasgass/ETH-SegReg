/**
 * @file   SegmentationPropagation-3D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:30:35 2015
 * 
 * @brief  2D semi-supervised segmentation propagation
 * 
 * 
 */


#include <stdio.h>
#include <iostream>
//#include "Segmentation-Propagation.h"
//#include "Segmentation-Propagation-Modular.h"
#include "Segmentation-Propagation-efficient1Hop.h"



using namespace SSSP;



int main(int argc, char ** argv)
{
    typedef short int PixelType;
    const unsigned int D=3;
    typedef itk::Image<PixelType,D> ImageType;
    SegmentationPropagationModular<ImageType,33> net;
    net.run(argc,argv);
    return 1;
}
