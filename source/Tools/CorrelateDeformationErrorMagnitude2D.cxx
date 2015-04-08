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
    typedef float PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    
    LabelImagePointerType truedeformation = ImageUtils<LabelImageType>::readImage(argv[1]);

    LabelImagePointerType deformationEstimate = ImageUtils<LabelImageType>::readImage(argv[2]);
    LabelImagePointerType deformationUpdate = ImageUtils<LabelImageType>::readImage(argv[3]);
    
    LabelImagePointerType trueError= TransfUtils<ImageType>::subtract(deformationEstimate,truedeformation) ;
    LabelImagePointerType estimatedError= TransfUtils<ImageType>::subtract(deformationEstimate,deformationUpdate) ;
    LabelImagePointerType trueErrorUpdate= TransfUtils<ImageType>::subtract(deformationUpdate,truedeformation) ;

    ImagePointerType trueErrMags=TransfUtils<ImageType>::computeLocalDeformationNorm(trueError);
    ImagePointerType estimatedErrMags=TransfUtils<ImageType>::computeLocalDeformationNorm(estimatedError);
    ImagePointerType trueErrUpMags=TransfUtils<ImageType>::computeLocalDeformationNorm(trueErrorUpdate);
    
    ImageUtils<ImageType>::ImageIteratorType trueIt(trueErrMags,trueErrMags->GetLargestPossibleRegion());
    ImageUtils<ImageType>::ImageIteratorType estIt(estimatedErrMags,trueErrMags->GetLargestPossibleRegion());
    ImageUtils<ImageType>::ImageIteratorType estItUp(trueErrUpMags,trueErrMags->GetLargestPossibleRegion());
    trueIt.GoToBegin();estIt.GoToBegin();estItUp.GoToBegin();
    for (;!trueIt.IsAtEnd();++trueIt,++estIt,++estItUp){
        std::cout<<trueIt.Get()<<" "<<estIt.Get()<<" "<<estItUp.Get()<<endl;
    }
    
	return 1;
}
