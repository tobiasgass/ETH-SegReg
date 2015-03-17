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
    double resamplingFactor=atof(argv[2]);

    ImagePointerType img1=TransfUtils<ImageType>::createEmptyImage(deformation1);
    ImagePointerType img=FilterUtils<ImageType>::NNResample(img1,resamplingFactor,false);
    LOG<<VAR(img->GetLargestPossibleRegion().GetSize())<<endl;
    //old 'resampling'
    LabelImagePointerType resampledbSpline=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation1,img,false);
    LOG<<"got bspline"<<endl;
    LabelImagePointerType reresampledbSpline=TransfUtils<ImageType>::bSplineInterpolateDeformationField(resampledbSpline,deformation1,false);
    //compute error norm
    double errorOld=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(reresampledbSpline,deformation1));
    LOG<<VAR(errorOld)<<endl;
    //new resampling
    resampledbSpline=TransfUtils<ImageType>::deformationFieldToBSpline(deformation1,img);
    ImageUtils<LabelImageType>::writeImage("bspline.mha",resampledbSpline);
    LOG<<"got bspline"<<endl;
    //reresampledbSpline=TransfUtils<ImageType>::computeDeformationFieldFromBSplineTransform(resampledbSpline,img1);
    
    double errorNew=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(resampledbSpline,deformation1));

    LOG<<VAR(errorNew)<<endl;

    
	return 1;
}
