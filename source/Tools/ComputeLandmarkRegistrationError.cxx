#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <fstream>

#include <itkInverseDisplacementFieldImageFilter.h>


using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{
    

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
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
    
    PointType p;
    p.Fill(0.0);
    deformation->SetOrigin(0.0);
    ImagePointerType referenceImage=ImageUtils<ImageType>::readImage(argv[2]);
    ImagePointerType targetLandmarkImage=ImageUtils<ImageType>::createEmpty(referenceImage);
    ImagePointerType deformedReferenceLandmarkImage=ImageUtils<ImageType>::createEmpty(referenceImage);
    targetLandmarkImage->FillBuffer(0); deformedReferenceLandmarkImage->FillBuffer(0);
    vector<PointType> landmarksReference, landmarksTarget;
    ifstream ifs(argv[3]);
    int i=0;
    while ( ! ifs.eof() ) {
        PointType point;
        for (int d=0;d<D;++d){
            ifs>>point[d];
        }
        LOG<<point<<endl;
        landmarksReference.push_back(point);
       
    } 
    std::cout<<"read "<<landmarksReference.size()<<" landmarks"<<std::endl;
    double sumSquareError=0.0;
    ifstream ifs2(argv[4]);
    i=0;
    for (;i<landmarksReference.size();++i){
        PointType idx;
        for (int d=0;d<D;++d){
            ifs2>>idx[d];
        }        
        IndexType index;
        targetLandmarkImage->TransformPhysicalPointToIndex(idx,index);
        targetLandmarkImage->SetPixel(index,65535);

        PointType deformedReferencePoint,targetPoint=idx;
        //referenceImage->TransformIndexToPhysicalPoint(landmarksReference[i],deformedReferencePoint);
        //deformation->TransformIndexToPhysicalPoint(idx,targetPoint);
        
        referenceImage->TransformPhysicalPointToIndex(landmarksReference[i],index);
        //std::cout<<VAR(targetPoint)<<endl;
        
        for (int i2 = 0; i2 < D; i2++) {
          deformedReferencePoint[i2] = landmarksReference[i][i2] + deformation->GetPixel(index)[i2];
        }

        //deformedReferencePoint+=invertedDeformation->GetPixel(landmarksReference[i]);
        referenceImage->TransformPhysicalPointToIndex(deformedReferencePoint,index);
        deformedReferenceLandmarkImage->SetPixel(index,65535);
        double localSquaredError=0;
        for (int d=0;d<D;++d){
            localSquaredError+=(targetPoint[d]-deformedReferencePoint[d])*(targetPoint[d]-deformedReferencePoint[d]);
        }
        //std::cout<<VAR(targetPoint)<<" "<<VAR(deformedReferencePoint)<<endl;
        std::cout<<"pt"<<i<<": "<<sqrt(localSquaredError)<<" ";
        sumSquareError+=sqrt(localSquaredError);
    }
    std::cout<<std::endl<<"totalAverage: "<<(sumSquareError)/(i+1)<<std::endl;
   
    if (argc>6){
        ImageUtils<ImageType>::writeImage(argv[5],  (ImageConstPointerType) deformedReferenceLandmarkImage );
        ImageUtils<ImageType>::writeImage(argv[6],  (ImageConstPointerType) targetLandmarkImage );
    }

	return 1;
}
