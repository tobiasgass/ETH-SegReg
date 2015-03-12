/**
 * @file   SegmentationPropagation-2D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:30:11 2015
 * 
 * @brief  3D Semi-supervised segmentation propagation (SSSP)
 * 
 * 
 */

#include <stdio.h>
#include <iostream>
#include "Segmentation-Propagation-efficient1Hop.h"



using namespace SSSP;



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
