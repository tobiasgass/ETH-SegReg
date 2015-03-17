#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>

#include <random>

using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{

	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef Image<double,D> FloatImageType;
    typedef FloatImageType::Pointer FloatImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> DisplacementType;
    typedef Image<DisplacementType,D> DisplacementFieldType;
    typedef DisplacementFieldType::Pointer DisplacementFieldPointerType;
    typedef ImageType::IndexType IndexType;

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string moving,target="",def,output,previousDef="";
    bool NN=false;
    int nFrames=1;
    int nPoints=5;
    double scale=1.0;
    double length=-1.0;
    double freq=1.0;
    as->parameter ("target", target, " filename of target image", false);
    as->parameter ("out", output, " output filename", true);
    as->parameter ("prev", previousDef, "optional previous deformation. the new def is composed to the previous one, and the result is scaled such that the total magnitude of the def. does not change", false);
    as->parameter ("n", nPoints, "number of grid control points per axis", false);
    as->parameter ("s", scale, "std dev relative to grid spacing", false);
    as->parameter ("l", length, "maximum error (overrides scale)", false);
    as->parameter ("f", freq, "probability of error at each grid point", false);

    as->parse();
    
    ImagePointerType image = ImageUtils<ImageType>::readImage(target);
    
    ImagePointerType coarseImg=FilterUtils<ImageType>::NNResample(image,1.0*nPoints/image->GetLargestPossibleRegion().GetSize()[0],false);

   
    DisplacementFieldPointerType coarseDef =TransfUtils<ImageType>::createEmpty(coarseImg);

    ImageType::SpacingType space=coarseDef->GetSpacing();
    ImageType::SizeType size=coarseDef->GetLargestPossibleRegion().GetSize();
    std::random_device rd;
    std::default_random_engine generator(rd());
    double maxErr=0.4*scale*space[0];
    if (length>=0){
        if (length>maxErr){
            LOG<<"WARNING: "<<VAR(length)<<" is larger than 0.4 grid spacing, folding may occur."<<endl;
        }
        maxErr=length;
    }
    std::uniform_real_distribution<float> distribution(-maxErr,maxErr);
    //std::normal_distribution<float> distribution(0.0,0.4*scale*space[0]);
    std::uniform_real_distribution<float> chanceDistribution(0.0,1.0);

    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(coarseDef,coarseDef->GetLargestPossibleRegion());
    it.GoToBegin();
    for (;!it.IsAtEnd();++it){
        IndexType idx=it.GetIndex();
        bool testBorder=false;
        for (int d=0;d<D;++d){
            if (idx[d] == 0 || idx[d] == size[d]-1){
                testBorder=true;
                continue;
            }
        }
        if (!testBorder){
            DisplacementType l;
            l.Fill(0.0);
            if (chanceDistribution(generator)<freq){
            for (int d=0;d<D;++d){
                l[d] = distribution(generator);  // generates number in the range 1..6 

            }
            }
            it.Set(l);
        }

    }
    DisplacementFieldPointerType interpolatedDef=TransfUtils<ImageType>::bSplineInterpolateDeformationField(coarseDef,image);
    double ade=TransfUtils<ImageType>::computeDeformationNorm(interpolatedDef);
    //LOG<<"Average deformation error: "<<newMag<<endl;
    typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DisplacementFieldType,double> DisplacementFieldJacobianDeterminantFilterType;
    typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
    jacobianFilter->SetInput(interpolatedDef);
    jacobianFilter->SetUseImageSpacingOff();
    jacobianFilter->Update();
    FloatImagePointerType jac=jacobianFilter->GetOutput();
    double minJac = FilterUtils<FloatImageType>::getMin(jac);
    double stdDevJac=sqrt(FilterUtils<FloatImageType>::getVariance(jac));
    LOG<<VAR(ade)<<" "<<VAR(minJac)<<" "<<VAR(stdDevJac)<<endl;

    if (previousDef!=""){
        DisplacementFieldPointerType previous=ImageUtils<DisplacementFieldType>::readImage(previousDef);
        double mag=TransfUtils<ImageType>::computeDeformationNorm(previous);
        interpolatedDef=TransfUtils<ImageType>::composeDeformations(interpolatedDef,previous);
        double newMag=TransfUtils<ImageType>::computeDeformationNorm(interpolatedDef);
        LOG<<VAR(mag)<< " " <<VAR(ade)<<endl;
        if (mag>0.0)
            ImageUtils<DisplacementFieldType>::multiplyImage(interpolatedDef,(mag/ade));
        

    }
    ImageUtils<DisplacementFieldType>::writeImage(output,interpolatedDef);
	return 1;
}
