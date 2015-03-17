#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"
#include "Metrics.h"
#include "GCoptimization.h"

using namespace std;
using namespace itk;


int main(int argc, char ** argv)
{

	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
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

    typedef float EnergyType;
    typedef GCoptimizationGeneralGraph MRFType;


    string inFile, outFile;
    double variance=1.0,mean=0.0;
  
    if (argc<4){
        LOG<<"Usage: FuseSegmentationsMaxVote3D <outputFile> <targetImage> <deformedImage1> <seg1>  <deformedImage2> <input2> ..."<<endl;
        exit(0);
    }
    
    
    ImagePointerType targetImage = ImageUtils<ImageType>::readImage(argv[2]);
    ImagePointerType downSampledTargetImage=FilterUtils<ImageType>::LinearResample(targetImage,0.3,true);
    LOG<<"allocating counts structure"<<endl;
    std::vector< map<int,float > * > votes(targetImage->GetRequestedRegion().GetNumberOfPixels(),NULL);
    std::vector< float > backGroundVotes(targetImage->GetRequestedRegion().GetNumberOfPixels(),0);
    LOG<<"done allocating counts structure"<<endl;
    //for relative weighing
    map<int,unsigned char > totalLabelCountPerImage;
    
    typedef  map<int,unsigned char >::iterator CountMapIterator;
    
    //accumulate counts
    for (int i=3;i<argc;i+=2){
        ImagePointerType deformedAtlasImage=FilterUtils<ImageType>::LinearResample(ImageUtils<ImageType>::readImage(argv[i]),downSampledTargetImage,true);
        LOG<<"Reading img "<<argv[i]<<endl;
        ImagePointerType img = ImageUtils<ImageType>::readImage(argv[i+1]);
        FloatImagePointerType lncc=Metrics<ImageType,FloatImageType>::efficientLNCC(downSampledTargetImage,deformedAtlasImage,10,10);

        ImageUtils<ImageType>::ImageIteratorType it2(img,img->GetRequestedRegion());

        typedef  itk::LinearInterpolateImageFunction<FloatImageType, double> ImageInterpolatorType;
        ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New();
        interpolator->SetInputImage(lncc);
        map<int,unsigned char > labelCountPerImage;

        it2.GoToBegin();      
        for (int c=0;!it2.IsAtEnd();++it2,++c){
            PixelType val2=it2.Get();
            PointType pt;
            targetImage->TransformIndexToPhysicalPoint(it2.GetIndex(),pt);
            float weight=interpolator->Evaluate(pt)+std::numeric_limits<float>::epsilon();
            if (val2==0){
                backGroundVotes[c]+=weight;
            }else{
                if (votes[c] == NULL){
                    votes[c]=new  map<int,float >;
                }
                (*(votes[c]))[val2]+=weight;
            }
            if (labelCountPerImage[val2] == 0) labelCountPerImage[val2]=1;
        }

        CountMapIterator countMapIt=labelCountPerImage.begin();
        for (;countMapIt!=labelCountPerImage.end();++countMapIt){
            ++totalLabelCountPerImage[countMapIt->first];
            LOG<<VAR(int(countMapIt->first))<<" "<<VAR(int(totalLabelCountPerImage[countMapIt->first]))<<endl;
        }
    }

    int maxCount=1;
    CountMapIterator countMapIt=totalLabelCountPerImage.begin();
    for (;countMapIt!=totalLabelCountPerImage.end();++countMapIt){
        LOG<<VAR(int(countMapIt->second)) <<" "<<VAR(countMapIt->first)<<endl;
        maxCount=max(maxCount,int(countMapIt->second));
    }
    
    MRFType  * m_optimizer;
    m_optimizer= new MRFType(targetImage->GetRequestedRegion().GetNumberOfPixels(),maxCount);

    ImageUtils<ImageType>::ImageIteratorType it2(targetImage,targetImage->GetRequestedRegion());
    it2.GoToBegin();
    for (int c=0;!it2.IsAtEnd();++it2,++c){
        float maxVote=1.0*backGroundVotes[c]/maxCount; int maxLabel=0;
        if (votes[c]){
            map<int,float >::iterator mapIt=(*(votes[c])).begin();
            for (;mapIt!=votes[c]->end();++mapIt){
                float vote=1.0*mapIt->second/(totalLabelCountPerImage[mapIt->first]);
                if (vote>maxVote){
                    maxVote=vote;
                    maxLabel=mapIt->first;
                }
            }
            delete votes[c];
        }
        it2.Set(maxLabel);
    }

    ImageUtils<ImageType>::writeImage(argv[1],targetImage);

	return 1;
}
