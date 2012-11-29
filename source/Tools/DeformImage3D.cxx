#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include <itkWarpImageFilter.h>

#include "TransformationUtils.h"


using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
    LOG<<CLOCKS_PER_SEC<<endl;

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    
    ImagePointerType image = ImageUtils<ImageType>::readImage(argv[1]);
    LabelImagePointerType deformation = ImageUtils<LabelImageType>::readImage(argv[2]);
  
#if 0
    typedef  itk::WarpImageFilter<ImageType,ImageType,LabelImageType>     WarperType;
    typedef  WarperType::Pointer     WarperPointer;
    WarperPointer warper=WarperType::New();
    warper->SetInput( image);
    warper->SetDeformationField(deformation);
    warper->SetOutputOrigin(  image->GetOrigin() );
    warper->SetOutputSpacing( image->GetSpacing() );
    warper->SetOutputDirection( image->GetDirection() );
    warper->Update();

    ImageUtils<ImageType>::writeImage(argv[3],  (ImageConstPointerType)  warper->GetOutput());
#else
       if (argc == 5){
        LOG<<"Performing NN interpolation"<<endl;
        //ImageUtils<ImageType>::writeImage(argv[3],  (ImageConstPointerType) ImageUtils<ImageType>::deformSegmentationImage((ImageConstPointerType)image,deformation) );
        ImageUtils<ImageType>::writeImage(argv[3],  (ImageConstPointerType) TransfUtils<ImageType>::warpImage((ImageConstPointerType)image,deformation,true) );
    }
    else{
        LOG<<"Performing linear interpolation"<<endl;
        ImageUtils<ImageType>::writeImage(argv[3],  (ImageConstPointerType) TransfUtils<ImageType>::warpImage((ImageConstPointerType)image,deformation) );
    }
#endif
    LOG<<"deformed image "<<argv[1]<<endl;
	return 1;
}
