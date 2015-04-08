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
    typedef short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<double,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    
    LabelImagePointerType deformation1 = ImageUtils<LabelImageType>::readImage(argv[1]);
    double resamplingFactor=atof(argv[2]);

    ImagePointerType refImage=TransfUtils<ImageType,double,double,double>::createEmptyImage(deformation1);
    ImagePointerType refImageLowRes=FilterUtils<ImageType>::LinearResample(refImage,resamplingFactor,false);
    if (0){
        LabelImagePointerType lowResDeformation= TransfUtils<ImageType,double,double,double>::computeBSplineTransformFromDeformationField(deformation1,refImageLowRes) ;
        LabelImagePointerType highResDeformation= TransfUtils<ImageType,double,double,double>::computeDeformationFieldFromBSplineTransform(lowResDeformation,refImage) ;
        LabelImagePointerType difference=TransfUtils<ImageType,double,double,double>::subtract(deformation1,highResDeformation);
        LOG<<"BsplineConversion :"<<TransfUtils<ImageType,double,double,double>::computeDeformationNorm(difference)<<endl;
    }
    if (0){
        typedef TransfUtils<ImageType,double,double,double>::BSplineTransformPointerType BSplineTransformPointerType;
        BSplineTransformPointerType lowResDeformation= TransfUtils<ImageType,double,double,double>::computeITKBSplineTransformFromDeformationField(deformation1,refImageLowRes) ;
      
        ImagePointerType fixedImage=ImageUtils<ImageType>::readImage(argv[4]);
        ImagePointerType movingImage=ImageUtils<ImageType>::readImage(argv[5]);
        ImageUtils<ImageType>::writeImage("warped.nii",TransfUtils<ImageType,double,double,double>::deformImage(movingImage,fixedImage,lowResDeformation));

        LabelImagePointerType highResDeformation= TransfUtils<ImageType,double,double,double>::computeDeformationFieldFromITKBSplineTransform(lowResDeformation,refImage) ;
        LabelImagePointerType difference=TransfUtils<ImageType,double,double,double>::subtract(deformation1,highResDeformation);
        LOG<<"ITKBsplineConversion :"<<TransfUtils<ImageType,double,double,double>::computeDeformationNorm(difference)<<endl;
    }
    if (1){
        LabelImagePointerType lowResDeformation= TransfUtils<ImageType,double,double,double>::bSplineInterpolateDeformationField(deformation1,refImageLowRes,false) ;
        LabelImagePointerType highResDeformation= TransfUtils<ImageType,double,double,double>::bSplineInterpolateDeformationField(lowResDeformation,refImage,false) ;
        LabelImagePointerType difference=TransfUtils<ImageType,double,double,double>::subtract(deformation1,highResDeformation);
        LOG<<"BsplineResampling (without smoothing) :"<<TransfUtils<ImageType,double,double,double>::computeDeformationNorm(difference)<<endl;

    }
    if (1){
        LabelImagePointerType lowResDeformation= TransfUtils<ImageType,double,double,double>::bSplineInterpolateDeformationField(deformation1,refImageLowRes,true) ;
        LabelImagePointerType highResDeformation= TransfUtils<ImageType,double,double,double>::bSplineInterpolateDeformationField(lowResDeformation,refImage,true) ;
        LabelImagePointerType difference=TransfUtils<ImageType,double,double,double>::subtract(deformation1,highResDeformation);
        LOG<<"BsplineResampling (with smoothing) :"<<TransfUtils<ImageType,double,double,double>::computeDeformationNorm(difference)<<endl;
    }
    if (1){
        LabelImagePointerType lowResDeformation= TransfUtils<ImageType,double,double,double>::linearInterpolateDeformationField(deformation1,refImageLowRes,false) ;
        LabelImagePointerType highResDeformation= TransfUtils<ImageType,double,double,double>::linearInterpolateDeformationField(lowResDeformation,refImage,false) ;
        LabelImagePointerType difference=TransfUtils<ImageType,double,double,double>::subtract(deformation1,highResDeformation);
        LOG<<"linear resampling (without smoothing) :"<<TransfUtils<ImageType,double,double,double>::computeDeformationNorm(difference)<<endl;
    
    }
    if (1){
        LabelImagePointerType lowResDeformation= TransfUtils<ImageType,double,double,double>::linearInterpolateDeformationField(deformation1,refImageLowRes,true) ;
        LabelImagePointerType highResDeformation= TransfUtils<ImageType,double,double,double>::linearInterpolateDeformationField(lowResDeformation,refImage,true) ;
        LabelImagePointerType difference=TransfUtils<ImageType,double,double,double>::subtract(deformation1,highResDeformation);
        LOG<<"linear resampling (with smoothing) :"<<TransfUtils<ImageType,double,double,double>::computeDeformationNorm(difference)<<endl;
    
    }
    //ImageUtils<LabelImageType>::writeImage(argv[3],lowResDeformation);
    

    LOG<<"deformed image "<<argv[1]<<endl;
	return 1;
}
