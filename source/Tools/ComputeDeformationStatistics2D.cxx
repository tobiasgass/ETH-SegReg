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
    typedef float Displacement;
    typedef Vector<Displacement,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    LabelImagePointerType deformation1 = ImageUtils<LabelImageType>::readImage(argv[1]);
    typedef  ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef  FloatImageType::Pointer FloatImagePointerType;
  

    double norm;
    FloatImagePointerType diff2=TransfUtils<ImageType>::computeLocalDeformationNorm(deformation1,1,&norm);

    typedef  itk::DisplacementFieldJacobianDeterminantFilter<LabelImageType,float> DisplacementFieldJacobianDeterminantFilterType;
     DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
    jacobianFilter->SetInput(deformation1);
    jacobianFilter->SetUseImageSpacingOff();
    jacobianFilter->Update();
    FloatImagePointerType jac=jacobianFilter->GetOutput();
    if (argc>2){
        ImageUtils<FloatImageType>::writeImage(argv[2],jac);
    }
    double minJac = FilterUtils<FloatImageType>::getMin(jac);

    LOG<<VAR(norm)<<" "<<VAR(FilterUtils<FloatImageType>::getMax(diff2))<<" "<<VAR(minJac)<<endl;

    
    

    
	return 1;
}
