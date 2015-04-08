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


	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned short PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> LabelType;
    typedef Image<LabelType,D> DeformationFieldType;
    typedef DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef ImageType::IndexType IndexType;
    DeformationFieldPointerType truedef = ImageUtils<DeformationFieldType>::readImage(argv[1]);
    DeformationFieldPointerType def1 = ImageUtils<DeformationFieldType>::readImage(argv[2]);
    DeformationFieldPointerType def2 = ImageUtils<DeformationFieldType>::readImage(argv[3]);
    
    DeformationFieldPointerType diff1 = TransfUtils<ImageType>::subtract(truedef,def1);
    DeformationFieldPointerType diff2 = TransfUtils<ImageType>::subtract(truedef,def2);

    typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
    DeformationIteratorType diff1It(diff1,diff1->GetLargestPossibleRegion());
    DeformationIteratorType diff2It(diff2,diff2->GetLargestPossibleRegion());
    diff1It.GoToBegin();
    diff2It.GoToBegin();
    for (;!diff1It.IsAtEnd();++diff1It,++diff2It){

        double d1Norm=diff1It.Get().GetNorm();
        double d2Norm=diff2It.Get().GetNorm();

        if (d1Norm>0.0){

            cout<<d1Norm<<" "<<1.0-d2Norm/d1Norm<<endl;

        }


    }
      
    
    
    

    

    
	return 1;
}
