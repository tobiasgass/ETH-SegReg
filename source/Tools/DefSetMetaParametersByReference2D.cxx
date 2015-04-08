#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>



using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
    

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    LabelImagePointerType deformation1 = ImageUtils<LabelImageType>::readImage(argv[1]);
    LabelImagePointerType deformationReference = ImageUtils<LabelImageType>::readImage(argv[2]);
    
    deformation1->SetOrigin(deformationReference->GetOrigin());
    LOG<<"Performing linear interpolation"<<endl;
    ImageUtils<LabelImageType>::writeImage(argv[3],  deformation1);
    
	return 1;
}
