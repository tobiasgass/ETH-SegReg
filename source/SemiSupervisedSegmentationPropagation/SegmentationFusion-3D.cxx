/**
 * @file   SegmentationFusion-3D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:29:39 2015
 * 
 * @brief  3D Modular Segmentation Fusion
 * 
 * 
 */

#include <stdio.h>
#include <iostream>

#include "Segmentation-Fusion-Modular.h"




using namespace SSSP;



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
