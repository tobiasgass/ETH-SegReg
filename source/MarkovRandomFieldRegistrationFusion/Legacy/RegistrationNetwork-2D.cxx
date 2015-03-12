#include <stdio.h>
#include <iostream>
#include "Registration-Propagation-Modular.h"



using namespace std;



int main(int argc, char ** argv)
{
    typedef unsigned short int PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
    RegistrationPropagationModular<ImageType> net;
    net.run(argc,argv);
    return 1;
}
