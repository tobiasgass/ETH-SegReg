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

	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
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
    bool spacing=false,origin=false,direction=false;
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("ref", refFile, " filename...", true);
    as->option ("spacing", spacing, "copy spacing of reference");
    as->option ("origin", origin, " copy origin of reference.");
    as->option ("direction", direction, " copy direction of reference.");
  
    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    ImagePointerType outImage=img;
  
    ImagePointerType reference=ImageUtils<ImageType>::readImage(refFile);

    if (spacing){
      outImage->SetSpacing(reference->GetSpacing());
    }
    if (direction){
      outImage->SetDirection(reference->GetDirection());
    }
    if (origin){
      outImage->SetOrigin(reference->GetOrigin());
    }
    
    ImageUtils<ImageType>::writeImage(outFile,outImage);

	return 1;
}
