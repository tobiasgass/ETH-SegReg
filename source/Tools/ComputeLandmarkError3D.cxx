#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <fstream>

#include <itkInverseDisplacementFieldImageFilter.h>
#include "itkVectorLinearInterpolateImageFunction.h"

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string refLandmarks,targetLandmarks,target="",def,output;
    bool linear=false;

    as->parameter ("refLandmarks", refLandmarks, " filename...", true);
    as->parameter ("targetLandmarks", targetLandmarks, " filename...", true);
    as->option ("linear", linear, " use linear upsampling of deformation");

    as->parse();
    
    
  


   
    PointType p;
    p.Fill(0.0);

    //ImagePointerType targetLandmarkImage=ImageUtils<ImageType>::createEmpty(referenceImage);
    //ImagePointerType deformedReferenceLandmarkImage=ImageUtils<ImageType>::createEmpty(referenceImage);
    vector<PointType> landmarksReference, landmarksTarget;
    ifstream ifs(refLandmarks.c_str());
    int i=0;
    
    while ( ! ifs.eof() ) {
        PointType point;
        for (int d=0;d<D;++d){
            ifs>>point[d];
            point[d]=point[d];
        }
        //LOG<<point<<endl;
        landmarksReference.push_back(point);
       
    } 
    //std::cout<<"read "<<landmarksReference.size()<<" landmarks"<<std::endl;
    double sumSquareError=0.0;
    ifstream ifs2(targetLandmarks.c_str());
    i=0;
    int count = 0;
    for (;i<landmarksReference.size()-1;++i){
        PointType pointTarget;
        for (int d=0;d<D;++d){
            ifs2>>pointTarget[d];
            pointTarget[d]=pointTarget[d];
        }        
        //LOG<< VAR(pointTarget) << endl;
        double localSquaredError=(pointTarget - landmarksReference[i]).GetNorm();
        std::cout<<"pt"<<i<<": "<<(localSquaredError)<<" ";
        sumSquareError+=localSquaredError;
        ++count;
    }
    
    std::cout<<" "<<"totalAverage: "<<(sumSquareError)/(count)<<std::endl;
   
    if (argc>6){
        // ImageUtils<ImageType>::writeImage(argv[5],  (ImageConstPointerType) deformedReferenceLandmarkImage );
        //ImageUtils<ImageType>::writeImage(argv[6],  (ImageConstPointerType) targetLandmarkImage );
    }

	return 1;
}
