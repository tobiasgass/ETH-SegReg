/**
 * @file   SegmentationNetwork-3D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:57:07 2015
 * 
 * @brief  
 * 
 * 
 */

#include <stdio.h>
#include <iostream>
#include "SegmentationNetwork.h"




using namespace SSSP;



int main(int argc, char ** argv)
{
    typedef short PixelType;
    const unsigned int D=3;
    typedef itk::Image<PixelType,D> ImageType;
    //SegmentationNetwork<ImageType> net;
    SegmentationNetwork<ImageType> net;
    net.run(argc,argv);
    return 1;
}
