/**
 * @file   SegmentationFusion-2D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:24:07 2015
 * 
 * @brief  Simple wrapper for segmentation fusion.
 * 
 * 
 */

#include <stdio.h>
#include <iostream>

#include "Segmentation-Fusion-Modular.h"



using namespace SSSP;



int main(int argc, char ** argv)
{
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    //SegmentationPropagationDeformationWeighting<ImageType> net;
    //SegmentationPropagation<ImageType> net;
    SegmentationFusionModular<ImageType,3> net;
    net.run(argc,argv);
    return 1;
}
