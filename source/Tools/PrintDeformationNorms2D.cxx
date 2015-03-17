#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>

#include <random>

using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
errors-grid13-scale1/localErrNorms.txt
	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> DisplacementType;
    typedef Image<DisplacementType,D> DisplacementFieldType;
    typedef DisplacementFieldType::Pointer DisplacementFieldPointerType;
    typedef ImageType::IndexType IndexType;

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string target="";
    as->parameter ("target", target, " filename of target image", false);
  
    DisplacementFieldPointerType def=ImageUtils<DisplacementFieldType>::readImage(target);
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(def,def->GetLargestPossibleRegion());
    it.GoToBegin();
    for (;!it.IsAtEnd();++it){
        std::cout<<it.Get().GetNorm()<<std::endl;


    }
	return 1;
}
