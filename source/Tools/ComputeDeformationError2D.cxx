#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <itkSubtractImageFilter.h>



using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
    LOG<<CLOCKS_PER_SEC<<endl;

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned short PixelType;
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
    typedef  ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef  FloatImageType::Pointer FloatImagePointerType;
    double sigma=0.05;
    if (argc>4)
        sigma=atof(argv[4]);
    typedef itk::SubtractImageFilter<LabelImageType,LabelImageType,LabelImageType> DeformationSubtractFilterType;

    DeformationSubtractFilterType::Pointer differ=DeformationSubtractFilterType::New();
    differ->SetInput1(deformation1);
    differ->SetInput2(deformationReference);
    differ->Update();
    LabelImagePointerType difference=differ->GetOutput();
    double norm;
    FloatImagePointerType diff2=TransfUtils<ImageType>::computeLocalDeformationNorm(differ->GetOutput(),sigma,&norm);
    LOG<<VAR(norm)<<endl;
    ImageUtils<ImageType>::writeImage(argv[3],FilterUtils<FloatImageType,ImageType>::truncateCast(diff2));


    

    
	return 1;
}
