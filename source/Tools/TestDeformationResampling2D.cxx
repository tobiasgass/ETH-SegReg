#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>



using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{

	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef float Displacement;
    typedef Vector<Displacement,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;

    ArgumentParser * as=new ArgumentParser(argc,argv);
    string moving,target="",def,output;
    bool NN=false;
    double resample=1.0;
    as->parameter ("moving", moving, " filename of moving image", true);
    as->parameter ("target", target, " filename of target image", false);
    as->parameter ("def", def, " filename of deformation", true);
    as->parameter ("resample", resample, " resampling factor", true);
    
    as->parameter ("out", output, " output filename", true);
    as->option ("NN", NN," use NN interpolation");
    as->parse();
    
    ImagePointerType image = ImageUtils<ImageType>::readImage(moving);
    
    ImagePointerType downsampledImage = FilterUtils<ImageType>::LinearResample(image,1.0/resample);
    LOG<<VAR(downsampledImage->GetLargestPossibleRegion())<<endl;
    LabelImagePointerType deformation = ImageUtils<LabelImageType>::readImage(def);

    LabelImagePointerType downsampledDefLinear = TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,downsampledImage);
    LabelImagePointerType downsampledDefBSpline = TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation,downsampledImage);
    LabelImagePointerType downsampledDefNN = TransfUtils<ImageType>::nearestNeighborInterpolateDeformationField(deformation,downsampledImage);

    LabelImagePointerType resampledDefLinear = TransfUtils<ImageType>::bSplineInterpolateDeformationField(downsampledDefLinear,image);
    LabelImagePointerType resampledDefBSpline = TransfUtils<ImageType>::bSplineInterpolateDeformationField(downsampledDefBSpline,image);
    LabelImagePointerType resampledDefNN = TransfUtils<ImageType>::bSplineInterpolateDeformationField(downsampledDefNN,image);
    
    ImageUtils<ImageType>::writeImage("linearbspline.png",  TransfUtils<ImageType,Displacement>::warpImage(image,resampledDefLinear) );
    ImageUtils<ImageType>::writeImage("bsplinebspline.png",  TransfUtils<ImageType,Displacement>::warpImage(image,resampledDefBSpline) );
    ImageUtils<ImageType>::writeImage("nnbspline.png",  TransfUtils<ImageType,Displacement>::warpImage(image,resampledDefNN) );

    ImageUtils<LabelImageType>::writeImage("linearbspline.mha",  resampledDefBSpline );
    ImageUtils<LabelImageType>::writeImage("linearlinear.mha",  resampledDefLinear );



	return 1;
}
