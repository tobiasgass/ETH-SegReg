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
    typedef  int  PixelType;
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
    double variance=1.0,mean=0.0;
  
    if (argc<4){
        LOG<<"Usage: CombineBinarySegmentations3D <outputFile> label1 <input1> label2 <input2> ..."<<endl;
        exit(0);
    }
    
    GaussianEstimatorScalarImage<ImageType> accumulator;
    int label=atoi(argv[2]);
    ImagePointerType img = FilterUtils<ImageType>::binaryThresholdingLow(ImageUtils<ImageType>::readImage(argv[3]),1);
    ImageUtils<ImageType>::multiplyImage(img,label);
    ImageUtils<ImageType>::ImageIteratorType it1(img,img->GetRequestedRegion());
    for (int i=4;i<argc;i+=2){
        label=atoi(argv[i]);
        ImagePointerType img2 = ImageUtils<ImageType>::readImage(argv[i+1]);
        //ImageUtils<ImageType>::multiplyImage(img2,label);
        ImageUtils<ImageType>::ImageIteratorType it2(img2,img2->GetRequestedRegion());
        it1.GoToBegin();it2.GoToBegin();
        for (;!it1.IsAtEnd();++it1,++it2){
            PixelType val2=it2.Get();
            if (val2){
                if (it1.Get()){
                    //cout<<"duplicate labelling for pixel "<<it1.GetIndex()<<" "<<VAR(it1.Get())<<" "<<VAR(val2)<<endl;
                }else{
                    it1.Set(label);
                }
            }
        }
        

    }

   


    ImageUtils<ImageType>::writeImage(argv[1],img);

	return 1;
}
