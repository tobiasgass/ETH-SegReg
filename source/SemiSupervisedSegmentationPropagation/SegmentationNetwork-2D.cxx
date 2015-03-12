/**
 * @file   SegmentationNetwork-2D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:54:39 2015
 * 
 * @brief  Simple wrapper for the segmentation network algorithm
 * 
 * 
 */
#include <stdio.h>
#include <iostream>
#include "SegmentationNetwork.h"




using namespace SSSP;



int main(int argc, char ** argv)
{
    typedef unsigned short int PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    SegmentationNetwork<ImageType> net;
    net.run(argc,argv);
    return 1;
}
