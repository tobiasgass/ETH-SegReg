#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <itkSubtractImageFilter.h>



using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{


	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned short PixelType;
    const unsigned int D=3;
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

    ImagePointerType mask=TransfUtils<ImageType>::createEmptyImage(difference);
    mask->FillBuffer(0);
    ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
    ImageType::IndexType offset;
    double fraction=0.9;
    for (int d=0;d<D;++d){
        offset[d]=(1.0-fraction)/2*size[d];
        size[d]=fraction*size[d];
    }
    
    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(offset);
    ImageUtils<ImageType>::setRegion(mask,region,1);

    double norm;
    FloatImagePointerType diff2=TransfUtils<ImageType>::computeLocalDeformationNorm(differ->GetOutput(),sigma,&norm, mask);
    LOG<<VAR(norm)<<endl;
    if (argc>3)
        ImageUtils<ImageType>::writeImage(argv[3],FilterUtils<FloatImageType,ImageType>::truncateCast(diff2));


    

    
	return 1;
}
