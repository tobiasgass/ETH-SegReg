/**
 * @file   RegistrationPropagation-2D.cxx
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 11:22:56 2015
 * 
 * @brief  Example call to one of the RegistrationPropagation classes
 * Registration propagation will update a set of n*(n-1) pairwise registrations between n images. Some functionality exists to update only subsets of such sets.
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
#include "Registration-Propagation-MRF.h"

//using namespace std;

using namespace MRegFuse;

int main(int argc, char ** argv)
{
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    //RegistrationUpdateResidual<ImageType> net;
    //RegistrationPropagationModular<ImageType> net;
    RegistrationPropagationMRF<ImageType> net;
    //RegistrationPropagationLocalSim<ImageType > net;
    //RegistrationPropagationIndirect<ImageType> net;
    //RegistrationPropagationIndirectCircleWeight<ImageType> net;
    //RegistrationPropagationIndirectStatismo<ImageType> net;
    //RegistrationCircles<ImageType> net;
    net.run(argc,argv);
    return 1;
}
