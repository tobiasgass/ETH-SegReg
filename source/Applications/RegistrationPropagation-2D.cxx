#include <stdio.h>
#include <iostream>
#include "Registration-Propagation-Modular.h"
#include "Registration-UpdateResidual.h"
#include "Registration-circles.h"
#include "Registration-Propagation-Indirect.h"

using namespace std;



int main(int argc, char ** argv)
{
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    //RegistrationUpdateResidual<ImageType> net;
    //RegistrationPropagationModular<ImageType> net;
    //RegistrationPropagationIndirect<ImageType> net;
    RegistrationCircles<ImageType> net;
    net.run(argc,argv);
    return 1;
}
