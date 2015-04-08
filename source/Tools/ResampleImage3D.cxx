#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;
    typedef  ImageType::SpacingType SpacingType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
 
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile,refFile="";
    double factor=-1;
    bool noSmoothing=false;
    bool nnResampling=false;
    bool rectifyAlignment=false;
    double spacing=-1;
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("ref", refFile, " filename...", false);
    as->option ("NN", nnResampling, " use NN resampling instead of linear resampling");
    as->option ("noSmoothing", noSmoothing, " do not smooth image when linearly downsampling..");
    as->option ("rectify", rectifyAlignment, " set origin to zero and direction matrix to identity.");
    as->parameter ("f", factor, "resample image by factor", false,true);
    as->parameter ("spacing,s", spacing, "resample image to uniform spacing", false);

    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    ImagePointerType outImage=img;
    if (refFile!=""){
        ImagePointerType reference=ImageUtils<ImageType>::readImage(refFile);
        if (nnResampling){
            outImage=FilterUtils<ImageType>::NNResample(img,reference,false);
        }else{
            outImage=FilterUtils<ImageType>::LinearResample(img,reference,!noSmoothing);
        }
    }else if (factor>0.0){
        if (nnResampling){
            outImage=FilterUtils<ImageType>::NNResample(img,factor,false);
        }else{
            outImage=FilterUtils<ImageType>::LinearResample(img,factor,!noSmoothing);
        }
    }else if (spacing>0.0){
        
        SpacingType spacingVec;
        spacingVec.Fill(spacing);
        outImage=FilterUtils<ImageType>::ResampleIsotropic(img,spacing,!noSmoothing,nnResampling);

    }else{
        LOG<<"No resampling directive given, image will be unchanged!"<<endl;
    }
    
    if (rectifyAlignment){
        PointType org; org.Fill(0.0); outImage->SetOrigin(org);
        ImageType::DirectionType dir;
        for (unsigned int d=0;d<D;++d){
            for (unsigned int d2=0;d2<D;++d2){
                if (d==d2){
                    dir(d,d2)=1.0;
                }else
                    dir(d,d2)=0.0;
            }
        }
        outImage->SetDirection(dir);


    }
    ImageUtils<ImageType>::writeImage(outFile,outImage);

	return 1;
}
