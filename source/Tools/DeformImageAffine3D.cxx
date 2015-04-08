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
    string moving,target="",def,output;
    int verbose=0;
    bool NN=false;
    as->parameter ("moving", moving, " filename of moving image", true);
    as->parameter ("target", target, " filename of target image", true);
    as->parameter ("def", def, " filename of deformation", true);
    as->parameter ("out", output, " output filename", true);
    as->parameter ("v", verbose, " verbosity", false);
    as->option ("NN", NN," use NN interpolation of image");
    as->parse();
    
    logSetVerbosity(verbose);
    ImagePointerType image = ImageUtils<ImageType>::readImage(moving);
    typedef TransfUtils<ImageType>::AffineTransformPointerType  AffineTransformPointerType;
    AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(def);
    LOGV(1)<<VAR(affine)<<endl;
    LOGI(1,TransfUtils<ImageType>::writeAffine("test.txt",affine));
    ImagePointerType ref = ImageUtils<ImageType>::readImage(target);
    
  
    ImageUtils<ImageType>::writeImage(output, (TransfUtils<ImageType,Displacement>::affineDeformImage(image,affine,ref,NN) ));
        
    //    LOG<<VAR(deformation->GetSpacing())<<" "<<endl;
	return 1;
}
