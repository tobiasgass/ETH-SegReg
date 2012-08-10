#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "argstream.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <fstream>

#include <itkInverseDisplacementFieldImageFilter.h>


using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{
    LOG<<CLOCKS_PER_SEC<<endl;

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned short PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;

    LabelImagePointerType deformation = ImageUtils<LabelImageType>::readImage(argv[1]);
    typedef InverseDisplacementFieldImageFilter<LabelImageType,LabelImageType> Inverter;
    Inverter::Pointer inverter=Inverter::New();
    cout<<VAR(inverter->GetSubsamplingFactor())<<std::endl;
    inverter->SetSubsamplingFactor(8);
    inverter->SetInput(deformation);
    //    inverter->Update();
    //    LabelImagePointerType invertedDeformation=inverter->GetOutput();
    ImagePointerType referenceImage=ImageUtils<ImageType>::readImage(argv[2]);
    ImagePointerType lmImage=ImageUtils<ImageType>::createEmpty(referenceImage);
    ImagePointerType lmImage2=ImageUtils<ImageType>::createEmpty(referenceImage);
    lmImage->FillBuffer(0); lmImage2->FillBuffer(0);
    vector<IndexType> landmarksReference, landmarksTarget;
    ifstream ifs(argv[3]);
    int i=0;
    for (;i<5;++i){
        IndexType point;
        for (int d=0;d<D;++d){
            ifs>>point[d];
        }
        landmarksReference.push_back(point);
        lmImage->SetPixel(point,65535);
    }
    double sumSquareError=0.0;
    ifstream ifs2(argv[4]);
    i=0;
    for (;i<5;++i){
        IndexType idx;
        for (int d=0;d<D;++d){
            ifs2>>idx[d];
        }        
        lmImage2->SetPixel(idx,65535);

        PointType deformedReferencePoint,targetPoint;
        referenceImage->TransformIndexToPhysicalPoint(landmarksReference[i],deformedReferencePoint);
        deformation->TransformIndexToPhysicalPoint(idx,targetPoint);
        std::cout<<VAR(targetPoint)<<endl;
        targetPoint+=deformation->GetPixel(idx);
        //deformedReferencePoint+=invertedDeformation->GetPixel(landmarksReference[i]);

        double localSquaredError=0;
        for (int d=0;d<D;++d){
            localSquaredError+=(targetPoint[d]-deformedReferencePoint[d])*(targetPoint[d]-deformedReferencePoint[d]);
        }
        std::cout<<VAR(targetPoint)<<" "<<VAR(deformedReferencePoint)<<endl;
        std::cout<<"Sq. error for pt"<<i<<" ="<<sqrt(localSquaredError)<<endl;
        sumSquareError+=sqrt(localSquaredError);
    }
    std::cout<<(sumSquareError)/(i+1)<<std::endl;
   
    if (argc>6){
        ImageUtils<ImageType>::writeImage(argv[5],  (ImageConstPointerType) TransfUtils<ImageType>::warpImage((ImageConstPointerType)lmImage,deformation) );
        ImageUtils<ImageType>::writeImage(argv[6],  (ImageConstPointerType) lmImage2 );
    }

	return 1;
}
