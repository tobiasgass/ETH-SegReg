/**
 * @file   RegistrationFusion2D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 11:25:56 2015
 * 
 * @brief  Call to registration fusion, which typically combines n registration hypotheses between two images to generate a new one
 * 
 * 
 */

#include <stdio.h>
#include <iostream>
//#include "Registration-Propagation-Modular.h"
//#include "Registration-UpdateResidual.h"
//#include "Registration-circles.h"
//#include "Registration-Propagation-Indirect.h"
//#include "Registration-Propagation-Indirect-Statismo.h"
//#include "Registration-Propagation-Indirect-CircleWeighting.h"
//#include "Registration-Propagation-LocalSim.h"
#include "Registration-Fusion-MRF.h"

//using namespace std;

using namespace MRegFuse;

int main(int argc, char ** argv)
{
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
  
    RegistrationFusionMRF<ImageType> net;
   
    net.run(argc,argv);
    return 1;
}

