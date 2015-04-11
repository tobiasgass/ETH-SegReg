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
    bool copyRefSpacing=false,copyRefOrigin=false,copyRefDirection=false;
    string direction="";
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("ref", refFile, " filename...", false);
    as->option ("copyRefSpacing", copyRefSpacing, "copy spacing of reference");
    as->option ("copyRefOrigin", copyRefOrigin, " copy origin of reference.");
    as->option ("copyRefDirection", copyRefDirection, " copy copyRefDirection of reference.");
    as->parameter("direction",direction,"set direction matrix",false);
    as->parse();
    

    ImagePointerType img = ImageUtils<ImageType>::readImage(inFile);

    ImagePointerType outImage=img;

    if (copyRefSpacing || copyRefOrigin || copyRefDirection){
    ImagePointerType reference=ImageUtils<ImageType>::readImage(refFile);

    if (copyRefSpacing){
      outImage->SetSpacing(reference->GetSpacing());
    }
    if (copyRefDirection){
      outImage->SetDirection(reference->GetDirection());
    }
    if (copyRefOrigin){
      outImage->SetOrigin(reference->GetOrigin());
    }
    }
    if (direction!=""){
      string delim(",");
      std::vector<std::string> entries=split(direction,delim[0]);
      DirectionType dir;
      for (int d1=0;d1<D;++d1){
	for (int d2=0;d2<D;++d2){
	  std::cout<<" "<<entries[d1*(D)+d2];
	  const char * c=entries[d1*(D)+d2].c_str();
	  dir[d2][d1]=atoi(c);
	}}
      std::cout<<std::endl;
      outImage->SetDirection(dir);
    }
    ImageUtils<ImageType>::writeImage(outFile,outImage);

	return 1;
}
