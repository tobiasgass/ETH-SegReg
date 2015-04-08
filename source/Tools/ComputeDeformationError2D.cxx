#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>



using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{


	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned short PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Image<double,D> DoubleImageType;
    typedef DoubleImageType::Pointer DoubleImagePointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> DisplacementType;
    typedef Image<DisplacementType,D> DeformationFieldType;
    typedef DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef ImageType::IndexType IndexType;
    DeformationFieldPointerType deformation1 = ImageUtils<DeformationFieldType>::readImage(argv[1]);
    DeformationFieldPointerType deformationReference = ImageUtils<DeformationFieldType>::readImage(argv[2]);
    typedef  ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef  FloatImageType::Pointer FloatImagePointerType;
  
    typedef itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationSubtractFilterType;

    //#define ADDITIVEERROR

#ifdef ADDITIVEERROR
    DeformationSubtractFilterType::Pointer differ=DeformationSubtractFilterType::New();
    differ->SetInput1(deformation1);
    differ->SetInput2(deformationReference);
    differ->Update();
    DeformationFieldPointerType difference=differ->GetOutput();
#else
    DeformationFieldPointerType difference=TransfUtils<ImageType>::composeDeformations(deformation1,deformationReference);
#endif



    ImagePointerType mask=TransfUtils<ImageType>::createEmptyImage(difference);
    mask->FillBuffer(0);
    ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
    ImageType::IndexType offset;
    double fraction=0.8;
    for (int d=0;d<D;++d){
        offset[d]=(1.0-fraction)/2*size[d];
        size[d]=fraction*size[d];
    }
    
    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(offset);
    ImageUtils<ImageType>::setRegion(mask,region,1);

    double ade;
    FloatImagePointerType diff2=TransfUtils<ImageType>::computeLocalDeformationNorm(difference,1,&ade, mask);
    double maxErr=FilterUtils<FloatImageType>::getMax(diff2);
    double minErr=FilterUtils<FloatImageType>::getMin(diff2);
    typedef  itk::DisplacementFieldJacobianDeterminantFilter<DeformationFieldType,float> DisplacementFieldJacobianDeterminantFilterType;
     DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
    jacobianFilter->SetInput(difference);
    jacobianFilter->SetUseImageSpacingOff();
    jacobianFilter->Update();
    FloatImagePointerType jac=jacobianFilter->GetOutput();
    double minJac = FilterUtils<FloatImageType>::getMin(jac,region);
    double stdDevJac=sqrt(FilterUtils<FloatImageType>::getVariance(jac));
    LOG<<VAR(ade)<<" " <<VAR(minErr)<<" "<<VAR(maxErr)<<" "<<VAR(minJac)<<" "<<VAR(stdDevJac)<<endl;

    if (argc>3){
        //ImageUtils<ImageType>::writeImage(argv[3],FilterUtils<FloatImageType,ImageType>::truncateCast(diff2));
        ImageUtils<FloatImageType>::writeImage(argv[3],diff2);
    }
    if (argc>4){
        //ImageUtils<ImageType>::writeImage(argv[3],FilterUtils<FloatImageType,ImageType>::truncateCast(diff2));
        ImageUtils<FloatImageType>::writeImage(argv[4],jac);
    }


    

    
	return 1;
}
