#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"

using namespace std;
using namespace itk;


int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  int  PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Image<float,D> FloatImageType;
    typedef FloatImageType::Pointer FloatImagePointerType;
    typedef ImageType::IndexType IndexType;
    string inFile, outFile;
  
    if (argc<4){
        LOG<<"Usage: FuseSegmentationsMaxVote3D <outputFile>  <input1>  <input2> ..."<<endl;
        exit(0);
    }
    
    
    ImagePointerType img = ImageUtils<ImageType>::readImage(argv[2]);
    LOG<<"allocating counts structure"<<endl;
    std::vector< map<int,unsigned char > * > votes(img->GetRequestedRegion().GetNumberOfPixels(),NULL);
    std::vector< unsigned char > backGroundVotes(img->GetRequestedRegion().GetNumberOfPixels(),0);
    LOG<<"done allocating counts structure"<<endl;
    //for relative weighing
    map<int,unsigned char > totalLabelCountPerImage;

    //accumulate counts
    for (int i=2;i<argc;++i){
        img = ImageUtils<ImageType>::readImage(argv[i]);
        LOG<<"Reading img "<<argv[i]<<endl;
        ImageUtils<ImageType>::ImageIteratorType it2(img,img->GetRequestedRegion());
        it2.GoToBegin();
        for (int c=0;!it2.IsAtEnd();++it2,++c){
            PixelType val2=it2.Get();
            if (val2==0){
                ++backGroundVotes[c];
            }else{
                if (votes[c] == NULL){
                    votes[c]=new  map<int,unsigned char >;
                }
                ++(*(votes[c]))[val2];
            }
        }
    }
    ImageUtils<ImageType>::ImageIteratorType it2(img,img->GetRequestedRegion());
    it2.GoToBegin();
    for (int c=0;!it2.IsAtEnd();++it2,++c){
        int maxVote=backGroundVotes[c]; int maxLabel=0;
        if (votes[c]){
            map<int,unsigned char >::iterator mapIt=(*(votes[c])).begin();
            for (;mapIt!=votes[c]->end();++mapIt){
                if (mapIt->second>maxVote){
                    maxVote=mapIt->second;
                    maxLabel=mapIt->first;
                }
            }
            delete votes[c];
        }
        it2.Set(maxLabel);
    }

    ImageUtils<ImageType>::writeImage(argv[1],img);

	return 1;
}
