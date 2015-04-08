#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <itkWarpImageFilter.h>

#include "TransformationUtils.h"


using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
    

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    
    //LabelImagePointerType deformation1 = TransfUtils<ImageType>::invert(ImageUtils<LabelImageType>::readImage(argv[1]));

    //LabelImagePointerType deformation2 = ImageUtils<LabelImageType>::readImage(argv[2]);
    LabelImagePointerType deformation2 = TransfUtils<ImageType>::invert(ImageUtils<LabelImageType>::readImage(argv[1]));
    
    ImageUtils<LabelImageType>::writeImage(argv[2],deformation2);
    
    LOG<<"deformed image "<<argv[1]<<endl;
	return 1;
}
