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



int main(int argc, char ** argv)
{
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef itk::Image<PixelType,D> ImageType;
  
    RegistrationFusionMRF<ImageType> net;
   
    net.run(argc,argv);
    return 1;
}

