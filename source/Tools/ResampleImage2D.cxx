#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include <sstream>

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
 
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile,refFile="";
    double factor=-1;
    bool noSmoothing=false;
    bool nnResampling=false;
    string size="";
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("ref", refFile, " filename...", false);
    as->parameter ("size", size, "new resolution (in #pixels)", false);
    as->option ("NN", nnResampling, " use NN resampling instead of linear resampling");
    as->option ("noSmoothing", noSmoothing, " do not smooth image when linearly downsampling..");
    as->parameter ("f", factor, "resample image by factor", false);

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
    }else if (size!=""){
        ImageType::SizeType imageSize;
        ImageType::SpacingType imageSpacing;
        std::stringstream ss(size);
        int i;
        int c=0;
        while (ss >> i && c<D)
            {
                imageSize[c]=i;
                ++c; 
                if (ss.peek() == 'x')
                    ss.ignore();
            }
        ImageType::SpacingType origSpace=img->GetSpacing();
        ImageType::SizeType origSize=img->GetLargestPossibleRegion().GetSize();

        ImageType::RegionType region;
        region.SetSize(imageSize);
        for (int d=0;d<D;++d){
            imageSpacing[d]=origSpace[d]*origSize[d]/imageSize[d];
        }
        ImagePointerType refImg=ImageType::New();
        refImg->SetSpacing(imageSpacing);
        refImg->SetRegions(region);
        refImg->Allocate();
        if (nnResampling){
            outImage=FilterUtils<ImageType>::NNResample(img,refImg,false);
        }else{
            outImage=FilterUtils<ImageType>::LinearResample(img,refImg,!noSmoothing);
        }


    }
    
    ImageUtils<ImageType>::writeImage(outFile,outImage);

	return 1;
}
