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
    typedef int PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string moving,target="",def1,def2,output;
    bool NN=false;
    as->parameter ("def1", def1, " filename of deformation1", true);
    as->parameter ("out", output, " output filename", true);
    as->option ("NN", NN," use NN interpolation of image");
    as->parse();
    

    typedef TransfUtils<ImageType,float>::AffineTransformType  AffineTransformType;
    typedef TransfUtils<ImageType,float>::AffineTransformPointerType  AffineTransformPointerType;
    AffineTransformPointerType affine1=TransfUtils<ImageType>::readAffine(def1);
 
    AffineTransformPointerType inverse2 = AffineTransformType::New();;
    affine1->GetInverse(inverse2);
    TransfUtils<ImageType,float>::writeAffine(output,inverse2);
	return 1;
}
