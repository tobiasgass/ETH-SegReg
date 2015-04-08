#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"

using namespace std;
using namespace itk;


int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short  PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Image<float,D> FloatImageType;
    typedef FloatImageType::Pointer FloatImagePointerType;
    typedef ImageType::IndexType IndexType;
    string inFile, outFile;
  
    if (argc<4){
        LOG<<"Usage: AverageImages3D <outputFile> <image> <segmentation>"<<endl;
        exit(0);
    }
    
    ImagePointerType img = ImageUtils<ImageType>::readImage(argv[2]);
    ImageUtils<ImageType>::ImageIteratorType it1(img,img->GetRequestedRegion());
    ImagePointerType img2 = ImageUtils<ImageType>::readImage(argv[3]);
    //ImageUtils<ImageType>::multiplyImage(img2,label);
    ImageUtils<ImageType>::ImageIteratorType it2(img2,img2->GetRequestedRegion());
    it1.GoToBegin();it2.GoToBegin();
    for (;!it1.IsAtEnd();++it1,++it2){
        PixelType val2=it2.Get();
        if (val2){
            it1.Set(32767);
            
        }
        
        

    }

   


    ImageUtils<ImageType>::writeImage(argv[1],img);

	return 1;
}
