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
    string refLandmarks,targetLandmarks,target="",def="",output="";
    bool linear=false;
    bool snap=false;

    as->parameter ("refLandmarks", refLandmarks, " filename...", true);
    as->parameter ("targetLandmarks", targetLandmarks, " filename...", true);
    as->parameter ("def", def, " filename of deformation", false);
    as->parameter ("output", output, " TPS interpolation of registration error", false);
    as->parameter ("target", target, " filename of target image", true);
    as->option ("linear", linear, " use linear upsampling of deformation");
    as->option ("snap", snap, " snap deformed landmarks to voxel");

    as->parse();
    
    
    LabelImagePointerType deformation;
    if (def!=""){
        deformation= ImageUtils<LabelImageType>::readImage(def);
    }
    ImagePointerType targetImage;
    if (target!=""){
        targetImage= ImageUtils<ImageType>::readImage(target);
        
        if (deformation.IsNotNull() && deformation->GetLargestPossibleRegion().GetSize() != targetImage->GetLargestPossibleRegion().GetSize()){
            if (linear){
                deformation=TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,targetImage);
            }else{
                deformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation,targetImage);
            }
        }
    }
    if (deformation.IsNull()){
        LOG<<"creating zero deformation"<<endl;
        deformation=TransfUtils<ImageType>::createEmpty(targetImage);
        LabelType zd;zd.Fill(0);deformation->FillBuffer(zd);
    }
    PointSetType::Pointer sourceLandMarks = PointSetType::New();
    PointSetType::Pointer targetLandMarks = PointSetType::New();
    PointType p1;     PointType p2;
    PointSetType::PointsContainer::Pointer sourceLandMarkContainer =
        sourceLandMarks->GetPoints();
    PointSetType::PointsContainer::Pointer targetLandMarkContainer =
        targetLandMarks->GetPoints();
    // Software Guide : EndCodeSnippet
    PointIdType id = itk::NumericTraits< PointIdType >::ZeroValue();
    
    DirectionType targetDir=deformation->GetDirection();
    typedef itk::VectorLinearInterpolateImageFunction<LabelImageType> DefInterpolatorType;
    DefInterpolatorType::Pointer defInterpol=DefInterpolatorType::New();
    defInterpol->SetInputImage(deformation);
    
    typedef DefInterpolatorType::ContinuousIndexType CIndexType;

    PointType p;
    p.Fill(0.0);
    DirectionType refDir=targetImage->GetDirection();

    //ImagePointerType targetLandmarkImage=ImageUtils<ImageType>::createEmpty(targetImage);
    //ImagePointerType deformedReferenceLandmarkImage=ImageUtils<ImageType>::createEmpty(targetImage);
    vector<PointType> landmarksReference, landmarksTarget;
    ifstream ifs(refLandmarks.c_str());
    int i=0;
    
    while ( ! ifs.eof() ) {
        PointType point;
        for (int d=0;d<D;++d){
            ifs>>point[d];
            point[d]=point[d]*refDir[d][d];
        }
        //LOG<<point<<endl;
        landmarksReference.push_back(point);
        sourceLandMarkContainer->InsertElement( id++, point );

       
    } 
    //std::cout<<"read "<<landmarksReference.size()<<" landmarks"<<std::endl;
    double sumSquareError=0.0;
    ifstream ifs2(targetLandmarks.c_str());
    i=0;
    int count = 0;
    id = itk::NumericTraits< PointIdType >::ZeroValue();

    for (;i<landmarksReference.size()-1;++i){
        PointType pointTarget;
        for (int d=0;d<D;++d){
            ifs2>>pointTarget[d];
            pointTarget[d]=pointTarget[d]*targetDir[d][d];

        }        
        PointType deformedReferencePoint;
        IndexType indexTarget,indexReference;
        deformation->TransformPhysicalPointToIndex(pointTarget,indexTarget);
        //LOG<<VAR(deformation->GetOrigin())<<endl;
        //LOG<<VAR(pointTarget)<<" "<<VAR(indexTarget)<<endl;
        
        targetImage->TransformPhysicalPointToIndex(landmarksReference[i],indexReference);
        
        //std::cout<<VAR(targetPoint)<<endl;
        //deformedReferencePoint= pointTarget+deformation->GetPixel(indexTarget);
        CIndexType cindex;
        deformation->TransformPhysicalPointToContinuousIndex(pointTarget,cindex);
        //LOG<<VAR(landmarksReference[i])<<" "<<VAR(indexReference)<<" "<<VAR(cindex)<<endl;
        if (deformation->GetLargestPossibleRegion().IsInside(cindex)){
            deformedReferencePoint= pointTarget+defInterpol->EvaluateAtContinuousIndex(cindex);
                
            if (snap){
                targetImage->TransformPhysicalPointToIndex(deformedReferencePoint,indexReference);
                targetImage->TransformIndexToPhysicalPoint(indexReference,deformedReferencePoint);
                    
            }
            //LOG<< VAR(pointTarget) << endl;
        }
        
        double localSquaredError=(deformedReferencePoint - landmarksReference[i]).GetNorm();
        targetLandMarkContainer->InsertElement( id++, deformedReferencePoint );
        
        for (int d=0;d<D;++d){
            deformedReferencePoint[d]=deformedReferencePoint[d]*targetDir[d][d];
        }    
        //LOG<< VAR(deformedReferencePoint) << endl;
        
        std::cout<<"pt"<<i<<": "<<(localSquaredError)<<" ";
        sumSquareError+=localSquaredError;
        ++count;
    }
    
    
    std::cout<<" "<<"totalAverage: "<<(sumSquareError)/(count)<<std::endl;
    if (output!=""){

        TransformType::Pointer tps = TransformType::New();
        LOG<<VAR(tps->GetStiffness())<<endl;
        //fix image borders
        IndexType border1;
        PointType point;
        for (int d=0;d<D;++d){
            ImageType::SizeType size=deformation->GetLargestPossibleRegion().GetSize();
            for (int s=0;s<size[d];s+=5){
                border1.Fill(0);
                border1[d]=s;
                //main axis
                deformation->TransformIndexToPhysicalPoint(border1,point);
                sourceLandMarkContainer->InsertElement( id, point );
                targetLandMarkContainer->InsertElement( id++, point );
                //xy +xz
                for (int d2=0;d2<D;++d2){
                    if (d2!=d){
                        border1.Fill(0);
                        border1[d]=s;
                        border1[d2]=size[d2]-1;
                        deformation->TransformIndexToPhysicalPoint(border1,point);
                        sourceLandMarkContainer->InsertElement( id, point );
                        targetLandMarkContainer->InsertElement( id++, point );
                    }
                }
                //xyz
                border1.Fill(0);
                border1[d]=s;
                for (int d2=0;d2<D;++d2){
                    if (d2!=d){
                        border1[d2]=size[d2]-1;
                    }
                }
                deformation->TransformIndexToPhysicalPoint(border1,point);
                sourceLandMarkContainer->InsertElement( id, point );
                targetLandMarkContainer->InsertElement( id++, point );
                
            }
        }
        tps->SetSourceLandmarks(sourceLandMarks);
        tps->SetTargetLandmarks(targetLandMarks);
        tps->ComputeWMatrix();
        LabelImagePointerType interpError = ImageUtils<LabelImageType>::duplicate(deformation);
        
        typedef itk::Vector<PixelType,D> VectorType;
        typedef  itk::ImageRegionIterator<LabelImageType> DefIterator;
        DefIterator defIt(interpError,interpError->GetLargestPossibleRegion());
        defIt.GoToBegin();
        IndexType index;
        PointType p1,p2;
        for (;!defIt.IsAtEnd();++defIt){
            index=defIt.GetIndex();
            interpError->TransformIndexToPhysicalPoint(index,p1);
            p2=tps->TransformPoint(p1);
            const VectorType constVector = p2-p1;
            defIt.Set(constVector);



        }
        ImageUtils<LabelImageType>::writeImage(output,interpError);


    }
    

	return 1;
}
