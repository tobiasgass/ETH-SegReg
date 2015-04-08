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
#include "itkThinPlateSplineKernelTransform.h"
// Software Guide : EndCodeSnippet
#include "itkPointSet.h"
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
    typedef   double                                           CoordinateRepType;
    typedef   itk::ThinPlateSplineKernelTransform< CoordinateRepType,
                                                   D>                                      TransformType;
    typedef   itk::Point< CoordinateRepType,
                          D >           PointType;
    typedef   TransformType::PointSetType                      PointSetType;
    typedef   PointSetType::PointIdentifier                    PointIdType;
    


    ArgumentParser * as=new ArgumentParser(argc,argv);
    string sourceLandmarks,targetLandmarks,target="",def,output="",intLandmarks="", defST,defIT,defSI;

    as->parameter ("sourceLandmarks", sourceLandmarks, " filename...", false);
    as->parameter ("intLandmarks", intLandmarks, " filename...", false);
    as->parameter ("targetLandmarks", targetLandmarks, " filename...", false);
    as->parameter ("defST", defST, " filename of deformation", true);
    as->parameter ("defSI", defSI, " filename of deformation", true);
    as->parameter ("defIT", defIT, " filename of deformation", true);
    as->parameter ("output", output, " TPS interpolation of registration error", false);
    as->parameter ("target", target, " filename of target image", false);

    as->parse();
    
    
    LabelImagePointerType deformST = ImageUtils<LabelImageType>::readImage(defST);
    LabelImagePointerType deformIT = ImageUtils<LabelImageType>::readImage(defIT);
    LabelImagePointerType deformSI = ImageUtils<LabelImageType>::readImage(defSI);

    //LabelImagePointerType deformSIT=TransfUtils<ImageType>::composeDeformations(deformSI,deformIT);
    LabelImagePointerType deformSIT=TransfUtils<ImageType>::composeDeformations(deformIT,deformSI);
    LabelImagePointerType diff=TransfUtils<ImageType>::subtract(deformSIT,deformST);
    double inconsistency=TransfUtils<ImageType>::computeDeformationNorm(diff);
                                                                         
    LOG<<VAR(inconsistency)<<endl;
    if (output!=""){
        ImageUtils<LabelImageType>::writeImage(output,diff);
    }
    

    return 0;
}
#if 0
    ImageConstPointerType targetImage;
    if (target!=""){
        targetImage=(ImageConstPointerType) ImageUtils<ImageType>::readImage(target);
        
        if (deformation->GetLargestPossibleRegion().GetSize() != targetImage->GetLargestPossibleRegion().GetSize()){
            if (linear){
                deformation=TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,targetImage);
            }else{
                deformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation,targetImage);
            }
        }
    }
    
 
    
    DirectionType targetDir=deformation->GetDirection();
    typedef itk::VectorLinearInterpolateImageFunction<LabelImageType> DefInterpolatorType;
    DefInterpolatorType::Pointer defInterpol=DefInterpolatorType::New();
    defInterpol->SetInputImage(deformation);
    
    typedef DefInterpolatorType::ContinuousIndexType CIndexType;

    PointType p;
    p.Fill(0.0);
    DirectionType sourceDir=targetImage->GetDirection();

    //ImagePointerType targetLandmarkImage=ImageUtils<ImageType>::createEmpty(targetImage);
    //ImagePointerType deformedReferenceLandmarkImage=ImageUtils<ImageType>::createEmpty(targetImage);
    vector<PointType> landmarksSource, landmarksTarget;
    ifstream ifs(sourceLandmarks.c_str());
    int i=0;
    
    while ( ! ifs.eof() ) {
        PointType point;
        for (int d=0;d<D;++d){
            ifs>>point[d];
            point[d]=point[d]*sourceDir[d][d];
        }
        //LOG<<point<<endl;
        landmarksSource.push_back(point);

       
    } 
    
    ifstream ifs2(intLandmarks.c_str());
    i=0;
    
    while ( ! ifs2.eof() ) {
        PointType point;
        for (int d=0;d<D;++d){
            ifs2>>point[d];
            point[d]=point[d]*refDir[d][d];
        }
        //LOG<<point<<endl;
        landmarksInt.push_back(point);

       
    } 
    //std::cout<<"read "<<landmarksSource.size()<<" landmarks"<<std::endl;
    double sumSquareError=0.0;
    ifstream ifs2(targetLandmarks.c_str());
    i=0;
    int count = 0;
    id = itk::NumericTraits< PointIdType >::ZeroValue();

    for (;i<landmarksSource.size()-1;++i){
        PointType pointTarget;
        for (int d=0;d<D;++d){
            ifs2>>pointTarget[d];
            pointTarget[d]=pointTarget[d]*targetDir[d][d];

        }        
        IndexType indexTarget,indexSource;
        deformation->TransformPhysicalPointToIndex(pointTarget,indexTarget);
        //LOG<<VAR(deformation->GetOrigin())<<endl;
        //LOG<<VAR(pointTarget)<<" "<<VAR(indexTarget)<<endl;
        PointType deformedSourcePoint;
        targetImage->TransformPhysicalPointToIndex(landmarksSource[i],indexSource);
        
        //std::cout<<VAR(targetPoint)<<endl;
        //deformedSourcePoint= pointTarget+deformation->GetPixel(indexTarget);
        CIndexType cindex;
        deformation->TransformPhysicalPointToContinuousIndex(pointTarget,cindex);
        //LOG<<VAR(landmarksSource[i])<<" "<<VAR(indexSource)<<" "<<VAR(cindex)<<endl;
        if (deformation->GetLargestPossibleRegion().IsInside(cindex)){
            deformedSourcePoint= pointTarget+defInterpol->EvaluateAtContinuousIndex(cindex);
            
            if (snap){
                targetImage->TransformPhysicalPointToIndex(deformedSourcePoint,indexSource);
                targetImage->TransformIndexToPhysicalPoint(indexSource,deformedSourcePoint);

            }
            //LOG<< VAR(pointTarget) << endl;
            double localSquaredError=(deformedSourcePoint - landmarksSource[i]).GetNorm();
            targetLandMarkContainer->InsertElement( id++, deformedSourcePoint );

            for (int d=0;d<D;++d){
                deformedSourcePoint[d]=deformedSourcePoint[d]*targetDir[d][d];
            }    
            //LOG<< VAR(deformedSourcePoint) << endl;

            std::cout<<"pt"<<i<<": "<<(localSquaredError)<<" ";
            sumSquareError+=localSquaredError;
            ++count;
        }
    }
    
    std::cout<<" "<<"totalAverage: "<<(sumSquareError)/(count)<<std::endl;
    if (output!=""){

        TransformType::Pointer tps = TransformType::New();
        tps->SetSourceLandmarks(sourceLandMarks);
        tps->SetTargetLandmarks(targetLandMarks);
        tps->ComputeWMatrix();
        LabelImagePointerType interpError = ImageUtils<LabelImageType>::duplicate(deformation);
        
        typedef  itk::ImageRegionIterator<LabelImageType> DefIterator;
        DefIterator defIt(interpError,interpError->GetLargestPossibleRegion());
        defIt.GoToBegin();
        IndexType index;
        PointType p1,p2;
        for (;!defIt.IsAtEnd();++defIt){
            index=defIt.GetIndex();
            interpError->TransformIndexToPhysicalPoint(index,p1);
            p2=tps->TransformPoint(p1);
            defIt.Set(p2-p1);



        }
        ImageUtils<LabelImageType>::writeImage(output,interpError);


    }
    

	return 1;
}
#endif
