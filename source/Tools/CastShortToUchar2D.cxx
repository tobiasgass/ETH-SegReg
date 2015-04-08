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
    typedef unsigned short PixelType;
    typedef unsigned char OutputPixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef Image<OutputPixelType,D> OutputImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef OutputImageType::Pointer OutputImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string moving,target="",def,output;
    bool NN=false;
    int nFrames=1;
    as->parameter ("in", moving, " filename of input image", true);
    as->parameter ("out", output, " output filename", true);
 
    as->parse();
    
    ImagePointerType image = ImageUtils<ImageType>::readImage(moving);

    ImageUtils<ImageType>::multiplyImage(image,1.0*numeric_limits<OutputPixelType>::max()/numeric_limits<PixelType>::max());

    ImageUtils<OutputImageType>::writeImage(output,FilterUtils<ImageType,OutputImageType>::cast(image));

	return 1;
}
